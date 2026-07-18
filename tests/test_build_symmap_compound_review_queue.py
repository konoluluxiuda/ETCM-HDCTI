import unittest

from tools.build_symmap_compound_review_queue import build_queue


class SymMapCompoundReviewQueueTest(unittest.TestCase):
    def test_queue_separates_coverage_and_quality_tracks(self):
        worklist = [
            {"local_entity_id": "1", "canonical_name": "A"},
            {"local_entity_id": "2", "canonical_name": "B"},
            {"local_entity_id": "3", "canonical_name": "C"},
            {"local_entity_id": "4", "canonical_name": "D"},
        ]
        attributes = [
            {"entity_id": "1", "canonical_smiles": "CCO",
             "formula_match": "exact", "resolution_status": "resolved_direct_cid"},
            {"entity_id": "2", "canonical_smiles": "CCC",
             "formula_match": "mismatch", "resolution_status": "resolved_direct_cid"},
            {"entity_id": "3", "canonical_smiles": "",
             "formula_match": "", "resolution_status": "pending_tcmsp_cross_reference"},
            {"entity_id": "4", "canonical_smiles": "",
             "formula_match": "", "resolution_status": "pending_manual_name_review"},
        ]

        queue, report = build_queue(
            worklist, attributes, minimum_coverage=0.75, buffer=1
        )

        self.assertEqual(report["coverage_gap"], 1)
        self.assertEqual(report["coverage_candidates"], 2)
        self.assertEqual(report["quality_conflicts"], 1)
        self.assertEqual(report["threshold_review_batch_size"], 2)
        self.assertEqual(queue[0]["entity_id"], "3")
        self.assertEqual(queue[0]["threshold_review_batch"], "yes")
        self.assertEqual(queue[-1]["review_track"], "quality_conflict")


if __name__ == "__main__":
    unittest.main()
