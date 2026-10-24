from difflib import SequenceMatcher

from python_service.db import fetch_medication_catalog


class MedicationMatcher:
    def match(self, keywords: list[str], top_k: int = 5) -> list[dict[str, object]]:
        keywords = [keyword for keyword in keywords if keyword and keyword.strip()]
        if not keywords:
            return []
        catalog = fetch_medication_catalog()
        bounded_top_k = max(1, min(top_k, 20))
        prepared = [(item, *self._search_terms(item)) for item in catalog]
        return [self._match_one(keyword, prepared, bounded_top_k) for keyword in keywords]

    def _search_terms(self, item):
        searchable = " ".join(
            str(value)
            for value in [item.get("item_name"), item.get("code"), item.get("manufacturer"), item.get("searchable_text")]
            if value
        )
        return self._normalize(searchable), self._normalize(str(item.get("item_name") or ""))

    def _match_one(
        self,
        keyword: str,
        catalog: list[tuple],
        top_k: int,
    ) -> dict[str, object]:
        normalized_keyword = self._normalize(keyword)
        candidates: list[dict[str, object]] = []
        for item, searchable, normalized_name in catalog:
            item_name = str(item.get("item_name") or "")
            score = self._score(normalized_keyword, searchable, normalized_name)
            if score <= 0.2:
                continue
            candidates.append(
                {
                    "itemId": int(item["item_id"]),
                    "itemType": str(item["item_type"]),
                    "itemName": item_name,
                    "manufacturer": item.get("manufacturer"),
                    "score": round(score, 4),
                }
            )

        candidates.sort(key=lambda candidate: float(candidate["score"]), reverse=True)
        return {"keyword": keyword.strip(), "candidates": candidates[:top_k]}

    def _score(self, keyword: str, searchable: str, item_name: str) -> float:
        if not keyword or not searchable:
            return 0.0
        if keyword == item_name:
            return 1.0
        if item_name.startswith(keyword):
            return 0.92
        if keyword in searchable:
            return 0.82
        return SequenceMatcher(a=keyword, b=item_name or searchable).ratio()

    def _normalize(self, text: str) -> str:
        return "".join(ch for ch in text.lower().strip() if ch.isalnum())
