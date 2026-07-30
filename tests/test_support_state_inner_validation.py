import unittest

from util.support_complete_split import (
    build_support_state_inner_validation,
)


class SupportStateInnerValidationTest(unittest.TestCase):
    def setUp(self):
        self.compounds = ["c%d" % index for index in range(10)]
        self.proteins = ["p%d" % index for index in range(10)]
        self.positive_pairs = {
            (compound, protein)
            for compound_index, compound in enumerate(self.compounds)
            for protein_index, protein in enumerate(self.proteins)
            if (compound_index + 2 * protein_index) % 7 < 2
        }
        self.outer_training_records = [
            [compound, protein, 1.0]
            for compound, protein in sorted(self.positive_pairs)
        ]

    def assert_balanced_and_known_positive_safe(
            self, inner_train, validation):
        for records in (inner_train, validation):
            positive_count = sum(float(row[2]) > 0 for row in records)
            negative_count = len(records) - positive_count
            self.assertEqual(positive_count, negative_count)
            self.assertGreater(positive_count, 0)
            self.assertFalse(any(
                float(row[2]) <= 0
                and (row[0], row[1]) in self.positive_pairs
                for row in records
            ))

    def test_target_cold_inner_split_is_deterministic_and_state_matched(self):
        first = build_support_state_inner_validation(
            self.outer_training_records,
            self.positive_pairs,
            "target_cold",
            0.34,
            91,
            "target_fold_0",
        )
        second = build_support_state_inner_validation(
            self.outer_training_records,
            self.positive_pairs,
            "target_cold",
            0.34,
            91,
            "target_fold_0",
        )
        inner_train, validation, metadata = first
        self.assertEqual(
            metadata["assignments_sha256"],
            second[2]["assignments_sha256"],
        )
        self.assert_balanced_and_known_positive_safe(
            inner_train, validation
        )
        train_compounds = {row[0] for row in inner_train}
        train_proteins = {row[1] for row in inner_train}
        validation_compounds = {row[0] for row in validation}
        validation_proteins = {row[1] for row in validation}
        self.assertTrue(validation_compounds <= train_compounds)
        self.assertFalse(train_proteins & validation_proteins)
        self.assertEqual(metadata["heldout_compounds"], 0)
        self.assertGreater(metadata["heldout_proteins"], 0)

    def test_double_cold_inner_split_holds_out_both_endpoint_types(self):
        inner_train, validation, metadata = (
            build_support_state_inner_validation(
                self.outer_training_records,
                self.positive_pairs,
                "double_cold",
                0.4,
                29,
                "double_c0_p0",
            )
        )
        self.assert_balanced_and_known_positive_safe(
            inner_train, validation
        )
        train_compounds = {row[0] for row in inner_train}
        train_proteins = {row[1] for row in inner_train}
        validation_compounds = {row[0] for row in validation}
        validation_proteins = {row[1] for row in validation}
        self.assertFalse(train_compounds & validation_compounds)
        self.assertFalse(train_proteins & validation_proteins)
        self.assertGreater(metadata["heldout_compounds"], 0)
        self.assertGreater(metadata["heldout_proteins"], 0)
        self.assertGreater(
            metadata["discarded_buffer_positive_count"], 0
        )

    def test_split_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "target_cold or double_cold"):
            build_support_state_inner_validation(
                self.outer_training_records,
                self.positive_pairs,
                "pair_stratified",
                0.2,
                7,
                "invalid",
            )


if __name__ == "__main__":
    unittest.main()
