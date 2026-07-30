import tempfile
import unittest
from pathlib import Path

from tools.audit_support_complete_cold_start import (
    audit_dataset,
    audit_datasets,
    build_markdown,
)


class SupportCompleteColdStartAuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_dataset(self, name, missing_pd=None):
        dataset = self.root / name
        dataset.mkdir()
        compounds = ["c%d" % index for index in range(12)]
        proteins = ["p%d" % index for index in range(10)]
        cp_edges = set()
        for compound_index, compound in enumerate(compounds):
            cp_edges.add(
                (compound, proteins[compound_index % len(proteins)])
            )
            cp_edges.add(
                (compound, proteins[(compound_index + 3) % len(proteins)])
            )
        (dataset / "C_P.txt").write_text(
            "".join("%s\t%s\n" % edge for edge in sorted(cp_edges)),
            encoding="utf-8",
        )
        (dataset / "H_C.txt").write_text(
            "".join(
                "h%d\t%s\n" % (index, compound)
                for index, compound in enumerate(compounds)
            ),
            encoding="utf-8",
        )
        missing_pd = set(missing_pd or [])
        (dataset / "P_D.txt").write_text(
            "".join(
                "%s\td%d\n" % (protein, index)
                for index, protein in enumerate(proteins)
                if protein not in missing_pd
            ),
            encoding="utf-8",
        )
        return dataset, cp_edges

    def test_double_cold_grid_covers_each_supported_edge_once(self):
        dataset, cp_edges = self.make_dataset("complete")
        row = audit_dataset(
            "complete",
            dataset,
            folds=2,
            seed=17,
            thresholds={
                "minimum_supported_edge_coverage": 0.0,
                "minimum_supported_targets": 2,
                "minimum_target_fold_positives": 1,
                "minimum_target_state_purity": 0.0,
                "minimum_double_cell_positives": 0,
            },
        )

        double = row["double_cold"]
        self.assertEqual(double["grid_shape"], [2, 2])
        self.assertEqual(double["evaluation_cells"], 4)
        self.assertEqual(double["covered_supported_positives"], len(cp_edges))
        self.assertEqual(double["positive_coverage"], 1.0)
        self.assertTrue(double["all_cells_entity_disjoint"])
        self.assertTrue(double["all_cells_have_1to1_negative_capacity"])

    def test_target_cold_folds_are_disjoint_and_deterministic(self):
        dataset, _ = self.make_dataset("deterministic")
        first = audit_dataset(
            "deterministic", dataset, folds=2, seed=29
        )
        second = audit_dataset(
            "deterministic", dataset, folds=2, seed=29
        )

        self.assertEqual(first, second)
        self.assertTrue(
            first["target_cold"]["all_folds_entity_disjoint"]
        )
        self.assertEqual(len(first["target_cold"]["folds"]), 2)
        for fold in first["target_cold"]["folds"]:
            self.assertEqual(fold["train_test_protein_overlap"], 0)
            self.assertLessEqual(
                fold["state_valid_test_positives"],
                fold["raw_supported_test_positives"],
            )

    def test_missing_pd_support_is_reported_not_hidden(self):
        dataset, cp_edges = self.make_dataset(
            "partial", missing_pd={"p0", "p1"}
        )
        row = audit_dataset("partial", dataset, folds=2, seed=17)

        self.assertEqual(row["pd_supported_proteins"], 8)
        self.assertLess(row["both_supported_positive_edges"], len(cp_edges))
        self.assertLess(row["both_supported_edge_coverage"], 1.0)

    def test_multidataset_report_and_markdown(self):
        first, _ = self.make_dataset("first")
        second, _ = self.make_dataset("second")
        report = audit_datasets(
            {"first": first, "second": second},
            folds=2,
            seed=17,
            thresholds={
                "minimum_supported_edge_coverage": 0.0,
                "minimum_supported_targets": 2,
                "minimum_target_fold_positives": 1,
                "minimum_target_state_purity": 0.0,
                "minimum_double_cell_positives": 0,
            },
        )
        markdown = build_markdown(report)

        self.assertEqual(report["passed_datasets"], 2)
        self.assertEqual(
            report["decision"], "GO_state_complete_four_dataset_pilot"
        )
        self.assertIn("Target-cold", markdown)
        self.assertIn("Double-cold", markdown)
        self.assertIn("2×2=4", markdown)


if __name__ == "__main__":
    unittest.main()
