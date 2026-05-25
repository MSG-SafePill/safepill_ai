from difflib import SequenceMatcher


class PillFusionService:
    def __init__(self, ocr_weight: float = 0.8, detect_weight: float = 0.2) -> None:
        self.ocr_weight = ocr_weight
        self.detect_weight = detect_weight

    def rank_candidates(
        self,
        detections: list[dict[str, float | int | str]],
        ocr_candidates: list[dict[str, float | int | str]],
        pill_catalog: list[dict[str, str | int | None]],
        top_k: int = 5,
    ) -> list[dict[str, float | str | int | None]]:
        if not pill_catalog:
            return []

        detection_confidence = 0.0
        if detections:
            detection_confidence = max(float(det["confidence"]) for det in detections)

        ocr_tokens = [self._normalize(str(item["normalizedText"])) for item in ocr_candidates]
        token_scores: list[tuple[str, float]] = []
        for item in ocr_candidates:
            token_scores.append((self._normalize(str(item["normalizedText"])), float(item["confidence"])))

        ranked: list[dict[str, float | str | int | None]] = []
        for pill in pill_catalog:
            pill_name = str(pill["pill_name"])
            imprint = str(pill["imprint_text"] or "")
            normalized_targets = [self._normalize(pill_name), self._normalize(imprint)]

            ocr_similarity, matched_token = self._best_ocr_score(ocr_tokens, token_scores, normalized_targets)
            final_score = (ocr_similarity * self.ocr_weight) + (detection_confidence * self.detect_weight)
            ranked.append(
                {
                    "pillName": pill_name,
                    "itemId": int(pill["item_id"]) if pill.get("item_id") is not None else None,
                    "itemType": str(pill.get("item_type") or "MEDICINE"),
                    "manufacturer": str(pill.get("manufacturer")) if pill.get("manufacturer") else None,
                    "confidence": round(final_score, 4),
                    "ocrScore": round(ocr_similarity, 4),
                    "detectionScore": round(detection_confidence, 4),
                    "matchedText": matched_token,
                }
            )

        ranked.sort(key=lambda item: float(item["confidence"]), reverse=True)
        return ranked[:top_k]

    def _best_ocr_score(
        self, tokens: list[str], token_scores: list[tuple[str, float]], targets: list[str]
    ) -> tuple[float, str | None]:
        if not tokens or not targets:
            return 0.0, None

        best = 0.0
        matched: str | None = None
        confidence_by_token = {token: confidence for token, confidence in token_scores}
        for token in tokens:
            for target in targets:
                if not token or not target:
                    continue
                ratio = SequenceMatcher(a=token, b=target).ratio()
                if token in target or target in token:
                    ratio = max(ratio, 0.92)
                weighted = ratio * confidence_by_token.get(token, 1.0)
                if weighted > best:
                    best = weighted
                    matched = token
        return best, matched

    def _normalize(self, text: str) -> str:
        return "".join(ch for ch in text.upper().strip() if ch.isalnum())
