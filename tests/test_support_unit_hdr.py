import tempfile
import unittest
from pathlib import Path

from HDR import HDR
from tools.prepare_support_complete_splits import prepare_dataset_manifest


class DummyConf(object):
    def __init__(self, values):
        self.config = dict(values)

    def contains(self, key):
        return key in self.config

    def __getitem__(self, key):
        return self.config[key]


class SupportUnitHDRTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
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
        (self.dataset / "C_P.txt").write_text(
            "".join("%s\t%s\n" % edge for edge in sorted(cp_edges)),
            encoding="utf-8",
        )
        (self.dataset / "H_C.txt").write_text(
            "".join(
                "h%d\t%s\n" % (index, compound)
                for index, compound in enumerate(compounds)
            ),
            encoding="utf-8",
        )
        (self.dataset / "P_D.txt").write_text(
            "".join(
                "%s\td%d\n" % (protein, index)
                for index, protein in enumerate(proteins)
            ),
            encoding="utf-8",
        )
        self.datapath = self.dataset / "ONE_indices.txt"
        self.datapath.write_text(
            "".join(
                "%s\t%s\t1\n" % edge for edge in sorted(cp_edges)
            ),
            encoding="utf-8",
        )
        self.split_dir = self.root / "splits"
        prepare_dataset_manifest(
            "synthetic",
            self.dataset,
            self.split_dir,
            folds=2,
            seed=17,
        )
        self.manifest_path = self.split_dir / "manifest.json"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def config(self, **overrides):
        values = {
            "datapath": str(self.datapath),
            "ratings.setup": "-columns 0 1 2",
            "evaluation.setup": "-support-unit",
            "experiment.protocol": "strict",
            "support.manifest": str(self.manifest_path),
            "support.mode": "target_cold",
            "support.target.fold": "0",
            "early.stopping": "False",
            "negative.strategy": "random",
        }
        values.update(overrides)
        return DummyConf(values)

    def test_target_unit_is_loaded_without_model_initialization(self):
        runner = HDR(self.config())

        self.assertEqual(
            runner.splitStrategy, "support_complete_target_cold"
        )
        self.assertEqual(
            runner.supportUnitMetadata["unit_key"], "target_fold_0"
        )
        self.assertEqual(
            sum(row[2] > 0 for row in runner.trainingData),
            sum(row[2] == 0 for row in runner.trainingData),
        )
        self.assertEqual(
            runner.supportUnitMetadata["train_test_protein_overlap"], 0
        )

    def test_double_unit_is_loaded_without_entity_overlap(self):
        runner = HDR(self.config(**{
            "support.mode": "double_cold",
            "support.compound.group": "0",
            "support.protein.group": "0",
        }))

        self.assertEqual(
            runner.splitStrategy, "support_complete_double_cold"
        )
        self.assertEqual(
            runner.supportUnitMetadata["unit_key"], "double_c0_p0"
        )
        self.assertEqual(
            runner.supportUnitMetadata["train_test_compound_overlap"], 0
        )
        self.assertEqual(
            runner.supportUnitMetadata["train_test_protein_overlap"], 0
        )

    def test_target_support_unit_builds_state_matched_inner_validation(self):
        runner = HDR(self.config(**{
            "early.stopping": "True",
            "validation.ratio": "0.3",
            "validation.seed": "117",
        }))

        self.assertTrue(runner.supportValidationData)
        train_compounds = {row[0] for row in runner.trainingData}
        train_proteins = {row[1] for row in runner.trainingData}
        validation_compounds = {
            row[0] for row in runner.supportValidationData
        }
        validation_proteins = {
            row[1] for row in runner.supportValidationData
        }
        self.assertTrue(validation_compounds <= train_compounds)
        self.assertFalse(train_proteins & validation_proteins)
        self.assertEqual(
            runner.supportInnerValidationMetadata["mode"],
            "target_cold",
        )

    def test_support_unit_rejects_mixed_negative_sampling(self):
        with self.assertRaisesRegex(
                ValueError, "negative.strategy must be random"):
            HDR(self.config(**{
                "negative.strategy": "mixed",
                "negative.hard.ratio": "0.25",
            }))

    def test_support_unit_rejects_mismatched_dataset(self):
        other = self.root / "other"
        other.mkdir()
        other_datapath = other / "ONE_indices.txt"
        other_datapath.write_text("c0\tp0\t1\n", encoding="utf-8")

        with self.assertRaisesRegex(
                ValueError, "does not match datapath"):
            HDR(self.config(**{"datapath": str(other_datapath)}))

    def test_support_unit_cannot_be_combined_with_cv(self):
        with self.assertRaisesRegex(
                ValueError, "exactly one"):
            HDR(self.config(**{
                "evaluation.setup": "-cv 2 -support-unit",
            }))


if __name__ == "__main__":
    unittest.main()
