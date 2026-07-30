import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_support_complete_splits import (
    prepare_dataset_manifest,
)
from util.support_complete_split import load_support_complete_unit


class SupportCompleteSplitManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_dataset(self):
        dataset = self.root / "dataset"
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
        (dataset / "P_D.txt").write_text(
            "".join(
                "%s\td%d\n" % (protein, index)
                for index, protein in enumerate(proteins)
            ),
            encoding="utf-8",
        )
        return dataset, cp_edges

    @staticmethod
    def read_records(path):
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            left_id, right_id, label = line.split()
            records.append((left_id, right_id, int(label)))
        return records

    def test_manifest_freezes_balanced_target_and_double_tests(self):
        dataset, cp_edges = self.make_dataset()
        output_dir = self.root / "splits"
        manifest = prepare_dataset_manifest(
            "synthetic",
            dataset,
            output_dir,
            folds=2,
            seed=17,
        )

        self.assertEqual(len(manifest["target_cold"]["folds"]), 2)
        self.assertEqual(len(manifest["double_cold"]["cells"]), 4)
        self.assertTrue(
            manifest["strict_guarantees"][
                "double_grid_covers_each_supported_positive_once"
            ]
        )
        self.assertEqual(
            manifest["double_cold"]["test_positives_total"],
            len(cp_edges),
        )

        seen_positive_pairs = set()
        for cell in manifest["double_cold"]["cells"]:
            records = self.read_records(output_dir / cell["test_path"])
            positives = {
                (left_id, right_id)
                for left_id, right_id, label in records if label == 1
            }
            negatives = {
                (left_id, right_id)
                for left_id, right_id, label in records if label == 0
            }
            self.assertFalse(positives & negatives)
            self.assertFalse(negatives & cp_edges)
            self.assertEqual(len(positives), len(negatives))
            self.assertFalse(seen_positive_pairs & positives)
            seen_positive_pairs.update(positives)
        self.assertEqual(seen_positive_pairs, cp_edges)

    def test_existing_manifest_is_reused_byte_for_byte(self):
        dataset, _ = self.make_dataset()
        output_dir = self.root / "splits"
        first = prepare_dataset_manifest(
            "synthetic", dataset, output_dir, folds=2, seed=17
        )
        first_bytes = (output_dir / "manifest.json").read_bytes()
        second = prepare_dataset_manifest(
            "synthetic", dataset, output_dir, folds=2, seed=17
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first_bytes, (output_dir / "manifest.json").read_bytes()
        )

    def test_changed_source_rejects_reuse(self):
        dataset, _ = self.make_dataset()
        output_dir = self.root / "splits"
        prepare_dataset_manifest(
            "synthetic", dataset, output_dir, folds=2, seed=17
        )
        with (dataset / "C_P.txt").open("a", encoding="utf-8") as handle:
            handle.write("c0\tp9\n")

        with self.assertRaisesRegex(ValueError, "sources"):
            prepare_dataset_manifest(
                "synthetic", dataset, output_dir, folds=2, seed=17
            )

    def test_tampered_artifact_rejects_reuse(self):
        dataset, _ = self.make_dataset()
        output_dir = self.root / "splits"
        manifest = prepare_dataset_manifest(
            "synthetic", dataset, output_dir, folds=2, seed=17
        )
        artifact = next(iter(manifest["artifacts"].values()))
        path = output_dir / artifact["path"]
        path.write_text("tampered\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "artifact"):
            prepare_dataset_manifest(
                "synthetic", dataset, output_dir, folds=2, seed=17
            )

    def test_manifest_file_is_valid_json(self):
        dataset, _ = self.make_dataset()
        output_dir = self.root / "splits"
        prepare_dataset_manifest(
            "synthetic", dataset, output_dir, folds=2, seed=17
        )
        stored = json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(stored["protocol"], "support_complete_cold_start")

    def test_loader_reconstructs_target_and_double_units(self):
        dataset, _ = self.make_dataset()
        output_dir = self.root / "splits"
        prepare_dataset_manifest(
            "synthetic", dataset, output_dir, folds=2, seed=17
        )

        target_train, target_test, target_info = (
            load_support_complete_unit(
                output_dir / "manifest.json",
                "target_cold",
                fold=0,
            )
        )
        double_train, double_test, double_info = (
            load_support_complete_unit(
                output_dir / "manifest.json",
                "double_cold",
                compound_group=0,
                protein_group=0,
            )
        )

        self.assertEqual(
            sum(row[2] > 0 for row in target_train),
            sum(row[2] == 0 for row in target_train),
        )
        self.assertEqual(
            sum(row[2] > 0 for row in double_train),
            sum(row[2] == 0 for row in double_train),
        )
        self.assertEqual(
            sum(row[2] > 0 for row in target_test),
            sum(row[2] == 0 for row in target_test),
        )
        self.assertEqual(
            sum(row[2] > 0 for row in double_test),
            sum(row[2] == 0 for row in double_test),
        )
        self.assertEqual(target_info["train_test_protein_overlap"], 0)
        self.assertEqual(double_info["train_test_compound_overlap"], 0)
        self.assertEqual(double_info["train_test_protein_overlap"], 0)

    def test_loader_rejects_changed_training_negative_hash(self):
        dataset, _ = self.make_dataset()
        output_dir = self.root / "splits"
        prepare_dataset_manifest(
            "synthetic", dataset, output_dir, folds=2, seed=17
        )
        manifest_path = output_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["target_cold"]["folds"][0][
            "training_negatives_sha256"
        ] = "0" * 64
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
                ValueError, "training negatives do not match"):
            load_support_complete_unit(
                manifest_path, "target_cold", fold=0
            )


if __name__ == "__main__":
    unittest.main()
