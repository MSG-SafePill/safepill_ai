import re
from typing import Any


class PrescriptionOcrParser:
    dosage_pattern = re.compile(r"(\d+(?:\.\d+)?\s?(?:mg|g|mcg|ug|µg|ml|정|캡슐|포))", re.IGNORECASE)
    days_pattern = re.compile(r"(\d+\s?(?:일|days?))", re.IGNORECASE)
    frequency_pattern = re.compile(r"(?:1일|하루)\s?(\d+)\s?회|(\d+)\s?회\s?(?:복용|투여|씩)")
    meal_pattern = re.compile(r"(식전|식후|식간|취침\s?전|아침|점심|저녁)")
    split_pattern = re.compile(r"[\n\r]+|[;,]")
    noise_pattern = re.compile(r"(처방|조제|약국|병원|의원|환자|성명|복용법|일수|투약|보험|영수|전화)")
    hard_noise_pattern = re.compile(
        r"(교부번호|발행기관|신분확인번호|승인번호|약품명|성분|급여|투약량|횟수|주의사항|보관방법|유효기간|주소)"
    )
    medicine_form_pattern = re.compile(r"(정|정제|캡|캡슐|시럽|현탁액|과립|산|연질캡슐|장용정|서방정)$")
    ingredient_suffix_pattern = re.compile(r"(염산염|수화물|무수물|수화산|복합체)$")
    meal_only_pattern = re.compile(r"^(아침|점심|저녁|취침전|식전|식후|식간)$")

    def parse(self, ocr_candidates: list[dict[str, float | str | int]]) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        seen: set[str] = set()

        for candidate in ocr_candidates:
            raw_text = str(candidate["text"]).strip()
            if not raw_text:
                continue

            for line in self._split_candidate(raw_text):
                if line in seen:
                    continue
                seen.add(line)

                if self._is_noise(line):
                    continue

                medicine_name = self._extract_medicine_name(line)
                if len(medicine_name) < 2:
                    continue

                dosage = self._first_match(self.dosage_pattern, line)
                frequency = self._extract_frequency(line)
                meal_timing = self._first_match(self.meal_pattern, line)
                days = self._first_match(self.days_pattern, line)
                if not self._is_medicine_line(
                    raw_text=line,
                    medicine_name=medicine_name,
                    dosage=dosage,
                    frequency=frequency,
                    meal_timing=meal_timing,
                ):
                    continue
                items.append(
                    {
                        "medicineName": medicine_name,
                        "rawText": line,
                        "dosage": dosage,
                        "frequency": frequency,
                        "mealTiming": meal_timing,
                        "days": days,
                        "scheduleSuggestions": self._build_schedule_suggestions(dosage, frequency, meal_timing),
                        "confidence": float(candidate.get("confidence", 0)),
                    }
                )

        return items

    def _split_candidate(self, text: str) -> list[str]:
        return [
            re.sub(r"\s+", " ", item).strip()
            for item in self.split_pattern.split(text)
            if re.sub(r"\s+", " ", item).strip()
        ]

    def _is_noise(self, text: str) -> bool:
        if len(text) < 2:
            return True
        if self.noise_pattern.search(text) and not self.dosage_pattern.search(text):
            return True
        digit_ratio = sum(char.isdigit() for char in text) / max(len(text), 1)
        return digit_ratio > 0.65

    def _extract_medicine_name(self, text: str) -> str:
        cleaned = self.frequency_pattern.sub(" ", text)
        cleaned = self.dosage_pattern.sub(" ", cleaned)
        cleaned = self.days_pattern.sub(" ", cleaned)
        cleaned = self.meal_pattern.sub(" ", cleaned)
        cleaned = re.sub(r"[^\w가-힣\s()\-]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned or text.strip()

    def _is_medicine_line(
        self,
        raw_text: str,
        medicine_name: str,
        dosage: str | None,
        frequency: str | None,
        meal_timing: str | None,
    ) -> bool:
        normalized_name = re.sub(r"\s+", "", medicine_name)
        normalized_line = re.sub(r"\s+", "", raw_text)

        if self.hard_noise_pattern.search(normalized_line):
            return False
        if self.meal_only_pattern.fullmatch(normalized_name):
            return False
        if len(normalized_name) < 2:
            return False
        if len(normalized_name) > 30 and not self.medicine_form_pattern.search(normalized_name):
            return False
        if not re.search(r"[가-힣A-Za-z]", normalized_name):
            return False

        has_medicine_shape = bool(
            self.medicine_form_pattern.search(normalized_name)
            or self.ingredient_suffix_pattern.search(normalized_name)
        )
        if has_medicine_shape:
            return True

        # Keep rare lines that include dosage/frequency but lost suffix by OCR.
        return bool(dosage or frequency or meal_timing)

    def _extract_frequency(self, text: str) -> str | None:
        match = self.frequency_pattern.search(text)
        if not match:
            return None
        count = match.group(1) or match.group(2)
        return f"1일 {count}회" if count else match.group(0)

    def _first_match(self, pattern: re.Pattern[str], text: str) -> str | None:
        match = pattern.search(text)
        return match.group(1).replace(" ", "") if match else None

    def _build_schedule_suggestions(
        self,
        dosage: str | None,
        frequency: str | None,
        meal_timing: str | None,
    ) -> list[dict[str, Any]]:
        times = self._suggest_times(frequency, meal_timing)
        return [
            {
                "takeTime": take_time,
                "daysOfWeek": ["EVERYDAY"],
                "dosage": dosage or "처방전 확인 필요",
                "mealTiming": meal_timing,
            }
            for take_time in times
        ]

    def _suggest_times(self, frequency: str | None, meal_timing: str | None) -> list[str]:
        if meal_timing:
            normalized = meal_timing.replace(" ", "")
            if normalized == "아침":
                return ["08:00"]
            if normalized == "점심":
                return ["13:00"]
            if normalized == "저녁":
                return ["19:00"]
            if normalized == "취침전":
                return ["22:00"]

        count = 1
        if frequency:
            matches = re.findall(r"(\d+)", frequency)
            if matches:
                count = max(1, min(int(matches[-1]), 4))
        return {
            1: ["08:00"],
            2: ["08:00", "19:00"],
            3: ["08:00", "13:00", "19:00"],
            4: ["08:00", "13:00", "19:00", "22:00"],
        }[count]
