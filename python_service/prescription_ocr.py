import re


class PrescriptionOcrParser:
    dosage_pattern = re.compile(r"(\d+(?:\.\d+)?\s?(?:mg|g|mcg|ug|µg|ml|정|캡슐|포|회))", re.IGNORECASE)
    days_pattern = re.compile(r"(\d+\s?(?:일|days?))", re.IGNORECASE)
    frequency_pattern = re.compile(r"(?:1일|하루)\s?(\d+)\s?회|(\d+)\s?회\s?(?:복용|투여)")
    meal_pattern = re.compile(r"(식전|식후|식간|취침\s?전|아침|점심|저녁)")

    def parse(self, ocr_candidates: list[dict[str, float | str | int]]) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        seen: set[str] = set()

        for candidate in ocr_candidates:
            raw_text = str(candidate["text"]).strip()
            if not raw_text or raw_text in seen:
                continue
            seen.add(raw_text)

            medicine_name = self._extract_medicine_name(raw_text)
            if len(medicine_name) < 2:
                continue

            items.append(
                {
                    "medicineName": medicine_name,
                    "rawText": raw_text,
                    "dosage": self._first_match(self.dosage_pattern, raw_text),
                    "frequency": self._extract_frequency(raw_text),
                    "mealTiming": self._first_match(self.meal_pattern, raw_text),
                    "days": self._first_match(self.days_pattern, raw_text),
                    "confidence": float(candidate.get("confidence", 0)),
                }
            )

        return items

    def _extract_medicine_name(self, text: str) -> str:
        cleaned = self.dosage_pattern.sub(" ", text)
        cleaned = self.days_pattern.sub(" ", cleaned)
        cleaned = self.frequency_pattern.sub(" ", cleaned)
        cleaned = self.meal_pattern.sub(" ", cleaned)
        cleaned = re.sub(r"[^\w가-힣\s()\-]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or text.strip()

    def _extract_frequency(self, text: str) -> str | None:
        match = self.frequency_pattern.search(text)
        if not match:
            return None
        count = match.group(1) or match.group(2)
        return f"1일 {count}회" if count else match.group(0)

    def _first_match(self, pattern: re.Pattern[str], text: str) -> str | None:
        match = pattern.search(text)
        return match.group(1).replace(" ", "") if match else None
