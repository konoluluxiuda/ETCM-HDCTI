import unittest
from pathlib import Path

from util.config import ModelConf


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class MultiDatasetColdStartConfigTest(unittest.TestCase):
    DATASETS = {
        "tcmsuite": "./dataset/TCMsuite/ONE_indices.txt",
        "tcmsp": "./dataset/TCMSP/one1.txt",
        "symmap": "./dataset/Symmap/one.txt",
        "etcm_mention10": "./dataset/ETCM2.0_core_mention10/ONE_indices.txt",
    }
    FROZEN_KEYS = (
        "datapath",
        "evaluation.setup",
        "evaluation.fold.limit",
        "evaluation.outer.test",
        "experiment.protocol",
        "random.seed",
        "split.strategy",
        "split.seed",
        "split.dir",
        "split.reuse",
        "early.stopping",
        "validation.ratio",
        "validation.seed",
        "validation.metric",
        "validation.interval",
        "validation.patience",
        "validation.min.delta",
        "pair.decoder",
        "num.factors",
        "num.max.epoch",
        "batch_size",
        "attention.max.nodes",
        "learnRate",
        "reg.lambda",
    )

    def load(self, dataset, variant):
        path = (
            REPOSITORY_ROOT
            / "configs"
            / ("HDCTI_%s_cold_start_%s_pilot.conf" % (dataset, variant))
        )
        self.assertTrue(path.exists(), path)
        return ModelConf(str(path))

    def test_no_context_and_herb_only_pairs_share_frozen_protocol(self):
        for dataset, datapath in self.DATASETS.items():
            with self.subTest(dataset=dataset):
                baseline = self.load(dataset, "no_context")
                herb_only = self.load(dataset, "herb_only")
                for key in self.FROZEN_KEYS:
                    self.assertEqual(baseline[key], herb_only[key], key)
                self.assertEqual(baseline["datapath"], datapath)
                self.assertEqual(baseline["split.strategy"], "compound_cold_start")
                self.assertEqual(baseline["evaluation.fold.limit"], "1")
                self.assertEqual(baseline["attention.max.nodes"], "2000")
                self.assertEqual(baseline["counterfactual.context"], "False")
                self.assertEqual(herb_only["counterfactual.context"], "False")
                self.assertEqual(baseline["context.interaction"], "False")
                self.assertEqual(herb_only["context.interaction"], "True")
                self.assertEqual(herb_only["context.herb_protein"], "True")
                self.assertTrue((REPOSITORY_ROOT / datapath).exists())

    def test_support_router_pilots_reuse_herb_only_protocol(self):
        for dataset in self.DATASETS:
            with self.subTest(dataset=dataset):
                herb_only = self.load(dataset, "herb_only")
                router = self.load(dataset, "support_router")
                for key in self.FROZEN_KEYS:
                    self.assertEqual(herb_only[key], router[key], key)
                self.assertEqual(router["support.router"], "True")
                self.assertEqual(
                    router["support.router.mode"], "monotonic_residual"
                )
                self.assertEqual(
                    router["support.router.pseudo.cold.ratio"], "0.1"
                )
                self.assertEqual(router["support.router.seed"], "62026")
                self.assertEqual(router["support.router.initial.slope"], "1.0")
                self.assertEqual(router["counterfactual.context"], "False")


if __name__ == "__main__":
    unittest.main()
