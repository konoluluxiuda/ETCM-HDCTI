import tempfile
import unittest
from pathlib import Path

from tools.prepare_support_complete_splits import prepare_dataset_manifest
from util.support_complete_split import build_four_state_support_unit


class SupportCompleteFourStateTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_manifest(self):
        dataset = self.root / "dataset"
        dataset.mkdir()
        compounds = ["c%d" % index for index in range(12)]
        proteins = ["p%d" % index for index in range(10)]
        positive_edges = {
            (compound, protein)
            for compound_index, compound in enumerate(compounds)
            for protein_index, protein in enumerate(proteins)
            if (compound_index + protein_index) % 3 == 0
        }
        (dataset / "C_P.txt").write_text(
            "".join(
                "%s\t%s\n" % edge for edge in sorted(positive_edges)
            ),
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
        output_dir = self.root / "splits"
        prepare_dataset_manifest(
            "synthetic",
            dataset,
            output_dir,
            folds=2,
            seed=17,
        )
        return output_dir / "manifest.json", positive_edges

    def test_four_states_share_one_training_graph_without_overlap(self):
        manifest_path, all_positive_edges = self.make_manifest()
        training, states, metadata = build_four_state_support_unit(
            manifest_path,
            compound_group=0,
            protein_group=0,
            warm_holdout_ratio=0.2,
            seed=29,
        )

        self.assertEqual(
            set(states),
            {"warm_warm", "cold_warm", "warm_cold", "cold_cold"},
        )
        train_compounds = {row[0] for row in training}
        train_proteins = {row[1] for row in training}
        train_pairs = {(row[0], row[1]) for row in training}
        seen_test_pairs = set()
        expected_support = {
            "warm_warm": (True, True),
            "cold_warm": (False, True),
            "warm_cold": (True, False),
            "cold_cold": (False, False),
        }
        for state, records in states.items():
            positives = [row for row in records if row[2] > 0]
            negatives = [row for row in records if row[2] <= 0]
            self.assertEqual(len(positives), len(negatives))
            self.assertGreater(len(positives), 0)
            pairs = {(row[0], row[1]) for row in records}
            self.assertFalse(train_pairs & pairs)
            self.assertFalse(seen_test_pairs & pairs)
            seen_test_pairs.update(pairs)
            compound_warm, protein_warm = expected_support[state]
            for compound_id, protein_id, _ in records:
                self.assertEqual(
                    compound_id in train_compounds, compound_warm
                )
                self.assertEqual(
                    protein_id in train_proteins, protein_warm
                )
            self.assertFalse({
                (row[0], row[1]) for row in negatives
            } & all_positive_edges)

        self.assertEqual(
            metadata["training_positive_count"],
            metadata["training_negative_count"],
        )
        self.assertEqual(
            set(metadata["states"]),
            set(states),
        )

    def test_four_state_builder_is_deterministic(self):
        manifest_path, _ = self.make_manifest()
        first = build_four_state_support_unit(
            manifest_path, 0, 0, warm_holdout_ratio=0.2, seed=29
        )
        second = build_four_state_support_unit(
            manifest_path, 0, 0, warm_holdout_ratio=0.2, seed=29
        )

        self.assertEqual(first, second)
        self.assertEqual(
            first[2]["assignments_sha256"],
            second[2]["assignments_sha256"],
        )

    def test_invalid_warm_holdout_ratio_is_rejected(self):
        manifest_path, _ = self.make_manifest()
        with self.assertRaisesRegex(ValueError, "warm_holdout_ratio"):
            build_four_state_support_unit(
                manifest_path, 0, 0, warm_holdout_ratio=0.0, seed=29
            )


if __name__ == "__main__":
    unittest.main()
