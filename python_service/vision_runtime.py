"""Small, dependency-free helpers shared by vision endpoints and benchmarks."""
import logging
import re
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from time import perf_counter
from uuid import uuid4

logger = logging.getLogger("uvicorn.error")
request_id = ContextVar("vision_request_id", default="startup")
MAX_IMAGE_BYTES = 30 * 1024 * 1024
VISION_PATHS = frozenset({
    "/infer", "/ocr", "/identify", "/identify-upload", "/classify-pill",
    "/classify-pills", "/prescription-ocr", "/prescription-ocr-upload",
})


@contextmanager
def timed_stage(stage: str):
    started = perf_counter()
    outcome = "ok"
    try:
        yield
    except BaseException:
        outcome = "error"
        raise
    finally:
        logger.info("vision request_id=%s stage=%s elapsed_ms=%.2f outcome=%s",
                    request_id.get(), stage, (perf_counter() - started) * 1000, outcome)


class ImageTooLarge(ValueError):
    pass


def save_upload(stream, suffix: str = ".jpg", max_bytes: int = MAX_IMAGE_BYTES) -> Path:
    """Copy a spooled upload in bounded chunks; remove partial files on all failures."""
    path = None
    try:
        with timed_stage("upload_save"), tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as target:
            path = Path(target.name)
            total = 0
            while chunk := stream.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise ImageTooLarge("Image must be 30 MB or smaller.")
                target.write(chunk)
            if total == 0:
                raise ValueError("Image is empty.")
        return path
    except BaseException:
        if path is not None:
            path.unlink(missing_ok=True)
        raise


class VisionTimingMiddleware:
    """Measure the entire ASGI exchange, including multipart receipt and response send."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] not in VISION_PATHS:
            await self.app(scope, receive, send)
            return
        incoming = dict(scope.get("headers", [])).get(b"x-request-id", b"").decode("ascii", errors="ignore")
        trace = incoming if re.fullmatch(r"[A-Za-z0-9-]{1,64}", incoming) else str(uuid4())
        token = request_id.set(trace)
        started = perf_counter()
        received_bytes = 0
        receive_ms = 0.0
        status = 500

        async def measured_receive():
            nonlocal received_bytes, receive_ms
            before = perf_counter()
            message = await receive()
            receive_ms += (perf_counter() - before) * 1000
            received_bytes += len(message.get("body", b""))
            return message

        async def measured_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message = {**message, "headers": [*message.get("headers", []), (b"x-request-id", trace.encode())]}
            await send(message)

        try:
            await self.app(scope, measured_receive, measured_send)
        finally:
            logger.info("vision request_id=%s stage=api_total path=%s status=%s elapsed_ms=%.2f upload_bytes=%s receive_wait_ms=%.2f",
                        trace, scope["path"], status, (perf_counter() - started) * 1000, received_bytes, receive_ms)
            request_id.reset(token)
