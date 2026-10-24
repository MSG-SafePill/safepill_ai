"""Real measurements only. Missing dependencies/models are recorded as 측정 불가.

Use --source-root to run this same script against an unmodified baseline checkout.
No dependencies are installed, no model is downloaded by this script, and no DB is written.
API mode permits only the existing image-analysis endpoints.
"""
import argparse
import hashlib
import importlib.metadata
import json
import mimetypes
import platform
import statistics
import sys
from functools import partial
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

API_PATHS = ("/classify-pills", "/prescription-ocr-upload", "/api/vision/identify", "/api/ocr/prescription")
EXPECTED_LABELS = {
    "디아셀캡슐50밀리그램.png": "40792",
    "써스펜데이정.png": "34342",
    "오피큐탄연질캡슐 .png": "29002",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def api_call(base_url, endpoint, path, timeout):
    boundary = uuid4().hex
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    # ASCII transport filename avoids platform-dependent non-ASCII HTTP headers.
    payload = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="image{path.suffix}"\r\n'
        f'Content-Type: {content_type}\r\n\r\n'
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = Request(base_url.rstrip("/") + endpoint, data=payload,
                      headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def build_runner(args):
    if args.mode == "api":
        if not args.api_url:
            raise ValueError("--api-url is required in API mode")
        return lambda path: api_call(args.api_url, args.endpoint, path, args.timeout), {}
    if args.mode == "ocr":
        from python_service.ocr_pipeline import OcrPipeline
        # For benchmarking, refuse network downloads when OCR weights are missing.
        import os
        model_dir = Path(os.getenv("EASYOCR_MODULE_PATH", os.getenv("MODULE_PATH", str(Path.home() / ".EasyOCR")))) / "model"
        if not all((model_dir / name).is_file() for name in ("craft_mlt_25k.pth", "korean_g2.pth")):
            raise FileNotFoundError(f"Existing EasyOCR CRAFT/korean_g2 weights required in {model_dir}")
        pipeline = OcrPipeline(use_gpu=args.ocr_gpu)
        import easyocr
        reader_factory = easyocr.Reader
        try:
            easyocr.Reader = partial(reader_factory, download_enabled=False)
            pipeline._reader()
        finally:
            easyocr.Reader = reader_factory
        return lambda path: pipeline.extract(path), {"ocr_gpu": args.ocr_gpu}
    if args.model is None or not args.model.is_file():
        raise FileNotFoundError("--model must point to existing, trained weights")
    metadata = {"model": str(args.model), "model_sha256": sha256(args.model), "device": args.device}
    if args.mode == "detection":
        from python_service.inference import YoloInferenceEngine
        engine = YoloInferenceEngine(args.model, device=args.device)
        metadata["model_classes"] = engine._model.names
        return engine.detect, metadata
    from python_service.pill_classifier import PillClassificationService
    from python_service.multi_pill_classifier import MultiPillClassificationService
    classifier = PillClassificationService(args.model, args.mapping, args.device)
    metadata["model_classes"] = classifier.model.names
    # Check baseline as well: do not benchmark generic ImageNet labels as pill classes.
    if classifier.model.task != "classify" or set(map(str, classifier.model.names.values())) != set(classifier.mapping):
        raise ValueError("Model class labels must exactly match the medicine mapping for this comparison")
    return MultiPillClassificationService(classifier).classify, metadata


def synchronize():
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(runner, path):
    synchronize()
    started = perf_counter()
    result = runner(path)
    synchronize()
    return (perf_counter() - started) * 1000, result


def summarize_result(result, mode):
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True)
    summary = {"result_sha256": hashlib.sha256(serialized.encode()).hexdigest()}
    if mode == "classification":
        summary["top1_labels"] = [pill["candidates"][0]["classLabel"] for pill in result if pill.get("candidates")]
    elif mode == "ocr":
        summary["text_count"] = len(result)
    elif mode == "detection":
        summary["detected_labels"] = [box["pillName"] for box in result]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("detection", "classification", "ocr", "api"), required=True)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", type=Path)
    parser.add_argument("--mapping", type=Path, default=Path("mappings/pill_class_mapping.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ocr-gpu", action="store_true")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--api-url")
    parser.add_argument("--endpoint", choices=API_PATHS, default="/classify-pills")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    sys.path.insert(0, str(args.source_root.resolve()))
    report = {
        "mode": args.mode, "source_root": str(args.source_root.resolve()),
        "python": sys.version, "platform": platform.platform(), "repeat": args.repeat,
        "note": "Inference results are real when measured. No mock fallback. Cold call and warm calls are separate.",
        "packages": {}, "images": [],
    }
    for name in ("ultralytics", "torch", "easyocr", "opencv-python-headless", "numpy"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = "missing"
    try:
        started = perf_counter()
        runner, metadata = build_runner(args)
        report.update(metadata)
        report["setup_ms"] = (perf_counter() - started) * 1000
    except Exception as exc:
        report.update(status="측정 불가", reason=f"{type(exc).__name__}: {exc}")
        runner = None
    for path in args.images:
        record = {"image": str(path), "expected_label": EXPECTED_LABELS.get(path.name)}
        try:
            record["image_sha256"] = sha256(path)
            if runner is None:
                record.update(status="측정 불가", reason=report["reason"])
            else:
                cold_ms, _ = measure(runner, path)
                runs = [measure(runner, path) for _ in range(args.repeat)]
                timings = [elapsed for elapsed, _ in runs]
                record.update(status="measured", first_call_ms=cold_ms, mean_ms=statistics.mean(timings),
                              min_ms=min(timings), max_ms=max(timings), samples_ms=timings)
                record["results"] = [summarize_result(result, args.mode) for _, result in runs]
                if args.mode == "classification" and record["expected_label"]:
                    record["expected_label_present_each_run"] = all(
                        record["expected_label"] in result["top1_labels"] for result in record["results"])
        except Exception as exc:
            record.update(status="측정 불가", reason=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, HTTPError):
                record["http_status"] = exc.code
        report["images"].append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}; measured {sum(r.get('status') == 'measured' for r in report['images'])}/{len(report['images'])} images")


if __name__ == "__main__":
    main()
