import unittest

from tools.audit_loco_dual_hypergraph_transfer import (
    build_transfer_matrices,
)


class LocoDualHypergraphTransferTest(unittest.TestCase):
    def test_channels_transfer_through_side_contexts(self):
        hc = {
            ("h_shared", "c_train"),
            ("h_shared", "c_cold"),
            ("h_other", "c_other"),
        }
        pd = {
            ("p_train", "d_shared"),
            ("p_cold", "d_shared"),
            ("p_other", "d_other"),
        }
        training = [
            ["c_train", "p_train", 1.0],
            ["c_other", "p_other", 1.0],
        ]
        evaluation = [
            ["c_cold", "p_train", 1.0],
            ["c_cold", "p_other", 0.0],
            ["c_train", "p_cold", 1.0],
            ["c_other", "p_cold", 0.0],
            ["c_cold", "p_cold", 1.0],
            ["c_cold", "p_other", 0.0],
        ]
        transfer = build_transfer_matrices(hc, pd, training, evaluation)

        self.assertGreater(
            transfer["scores"]["herb_to_target"][0],
            transfer["scores"]["herb_to_target"][1],
        )
        self.assertGreater(
            transfer["scores"]["disease_to_compound"][2],
            transfer["scores"]["disease_to_compound"][3],
        )
        self.assertGreater(
            transfer["scores"]["dual_transfer"][4],
            transfer["scores"]["dual_transfer"][5],
        )

    def test_self_paths_do_not_copy_training_labels(self):
        hc = {("h0", "c0"), ("h1", "c1")}
        pd = {("p0", "d0"), ("p1", "d1")}
        training = [["c0", "p0", 1.0], ["c1", "p1", 1.0]]
        evaluation = [["c0", "p0", 1.0], ["c1", "p1", 1.0]]
        transfer = build_transfer_matrices(hc, pd, training, evaluation)
        for channel in (
                "herb_to_target", "disease_to_compound", "dual_transfer"):
            values = transfer["scores"][channel]
            self.assertTrue((values == 0.0).all())


if __name__ == "__main__":
    unittest.main()
