import tempfile
import unittest
from pathlib import Path

from tools.audit_multidataset_cold_start import audit_datasets


class MultiDatasetColdStartAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_dataset(self, name, supported=True, compounds=6):
        dataset = self.root / name
        dataset.mkdir()
        cp_rows = []
        hc_rows = []
        for compound in range(1, compounds + 1):
            cp_rows.append("%d\t%d\n" % (compound, 10 + compound % 2))
            if supported or compound > 1:
                hc_rows.append("%d\t%d\n" % (100 + compound, compound))
        (dataset / "C_P.txt").write_text("".join(cp_rows), encoding="utf-8")
        (dataset / "H_C.txt").write_text("".join(hc_rows), encoding="utf-8")
        (dataset / "P_D.txt").write_text("10\t20\n11\t21\n", encoding="utf-8")
        return dataset

    def test_all_supported_datasets_pass_common_protocol(self):
        datasets = {
            "a": self.make_dataset("a"),
            "b": self.make_dataset("b"),
        }
        report = audit_datasets(
            datasets,
            folds=2,
            thresholds={
                "minimum_hc_support": 0.95,
                "minimum_compounds": 2,
                "minimum_fold_positives": 2,
            },
        )

        self.assertEqual(
            report["decision"],
            "supports_multidataset_cold_start_and_uniform_CHCR",
        )
        self.assertEqual(report["passed_datasets"], 2)

    def test_low_hc_support_blocks_one_dataset(self):
        dataset = self.make_dataset("low", supported=False, compounds=6)
        report = audit_datasets(
            {"low": dataset},
            folds=2,
            thresholds={
                "minimum_hc_support": 0.90,
                "minimum_compounds": 2,
                "minimum_fold_positives": 2,
            },
        )

        self.assertEqual(report["decision"], "partial_compound_cold_start_support")
        self.assertFalse(
            report["datasets"][0]["cold_start_criteria"]["hc_support"]
        )

    def test_shared_herb_blocks_chcr_despite_full_hc_support(self):
        dataset = self.make_dataset("shared", compounds=6)
        (dataset / "H_C.txt").write_text(
            "".join("100\t%d\n" % compound for compound in range(1, 7)),
            encoding="utf-8",
        )
        report = audit_datasets(
            {"shared": dataset},
            folds=2,
            thresholds={
                "minimum_hc_support": 0.95,
                "minimum_chcr_coverage": 0.90,
                "minimum_compounds": 2,
                "minimum_fold_positives": 2,
            },
        )

        row = report["datasets"][0]
        self.assertTrue(row["cold_start_criteria"]["hc_support"])
        self.assertFalse(row["chcr_criteria"]["compound_coverage"])
        self.assertEqual(
            report["decision"],
            "supports_multidataset_compound_cold_start_with_selective_CHCR",
        )
        self.assertEqual(
            row["counterfactual_donors"]["eligible_compounds"], 0
        )


if __name__ == "__main__":
    unittest.main()
