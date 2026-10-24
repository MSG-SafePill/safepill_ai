import unittest
from unittest.mock import patch

from python_service.medication_matcher import MedicationMatcher

# Fixed test fixtures, never used as API inference results.
CATALOG = [
    {"item_id": 1, "item_type": "MEDICINE", "item_name": "디아셀캡슐50밀리그램", "code": "40792", "manufacturer": "제약사", "searchable_text": "CLP DE1"},
    {"item_id": 2, "item_type": "MEDICINE", "item_name": "오피큐탄연질캡슐", "code": "29002", "manufacturer": None},
    {"item_id": 3, "item_type": "MEDICINE", "item_name": "써스펜데이정", "code": "34342", "manufacturer": None},
]


class MatcherTests(unittest.TestCase):
    def test_empty_keywords_do_not_query_db(self):
        with patch("python_service.medication_matcher.fetch_medication_catalog") as fetch:
            self.assertEqual(MedicationMatcher().match(["", " "]), [])
            fetch.assert_not_called()

    def test_exact_prefix_code_and_imprint_scores_are_preserved(self):
        with patch("python_service.medication_matcher.fetch_medication_catalog", return_value=CATALOG):
            result = MedicationMatcher().match(["써스펜데이정", "디아셀", "29002", "CLP DE1"])
        self.assertEqual([r["candidates"][0]["itemId"] for r in result], [3, 1, 2, 1])
        self.assertEqual([r["candidates"][0]["score"] for r in result], [1.0, 0.92, 0.82, 0.82])

    def test_catalog_normalization_is_once_per_item(self):
        matcher = MedicationMatcher()
        with patch("python_service.medication_matcher.fetch_medication_catalog", return_value=CATALOG), \
             patch.object(matcher, "_search_terms", wraps=matcher._search_terms) as normalize:
            result = matcher.match(["디아셀", "써스펜", "오피큐탄", "디아셀"])
        self.assertEqual(normalize.call_count, len(CATALOG))
        self.assertEqual(result[0], result[3])


if __name__ == "__main__":
    unittest.main()
