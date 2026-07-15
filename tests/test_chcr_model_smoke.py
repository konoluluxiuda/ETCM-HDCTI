import os
import tempfile
import unittest
from pathlib import Path

import numpy as np


class DummyConf(object):
    def __init__(self, values):
        self.config = dict(values)

    def __getitem__(self, key):
        return self.config[key]

    def contains(self, key):
        return key in self.config


class ChcrModelSmokeTest(unittest.TestCase):
    def test_chcr_trains_one_epoch_with_exact_disjoint_donors(self):
        os.environ['HDCTI_FORCE_CPU'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow.compat.v1 as tf

        from HDCTI import HDCTI

        tf.reset_default_graph()
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            (dataset_dir / 'H_C.txt').write_text(
                'h0\tc0\nh1\tc1\nh2\tc2\nh3\tc3\n', encoding='utf-8'
            )
            (dataset_dir / 'C_P.txt').write_text(
                'c0\tp0\nc1\tp1\nc2\tp0\nc3\tp1\n', encoding='utf-8'
            )
            (dataset_dir / 'P_D.txt').write_text(
                'p0\td0\np1\td1\n', encoding='utf-8'
            )
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text('c0\tp0\t1\nc1\tp1\t1\n', encoding='utf-8')
            conf = DummyConf({
                'datapath': str(datapath),
                'ratings.setup': '-columns 0 1 2',
                'evaluation.setup': '-cv 2',
                'experiment.protocol': 'strict',
                'model.name': 'HDCTI',
                'num.factors': '4',
                'num.max.epoch': '1',
                'batch_size': '2',
                'learnRate': '-init 0.005 -max 1',
                'reg.lambda': '-u 0.001 -i 0.001 -b 0.2 -s 0.2',
                'weight.reg': '0.01',
                'pair.decoder': 'dot',
                'context.interaction': 'True',
                'context.compound_disease': 'False',
                'context.herb_protein': 'True',
                'context.herb_protein.mode': 'static',
                'context.herb_disease': 'False',
                'counterfactual.context': 'True',
                'counterfactual.match': 'exact_hc_degree_disjoint',
                'counterfactual.weight': '0.05',
                'counterfactual.margin': '0.2',
                'counterfactual.draws': '2',
                'counterfactual.seed': '7',
                'attention.max.nodes': '10',
                'output.setup': 'off -dir ./results/',
                'gpu.allow_growth': 'False',
                'gpu.log_device_placement': 'False',
            })
            training = [
                ['c0', 'p0', 1.0], ['c0', 'p1', 0.0],
                ['c1', 'p1', 1.0], ['c1', 'p0', 0.0],
            ]
            test = [['c2', 'p0', 1.0], ['c2', 'p1', 0.0]]
            previous_directory = os.getcwd()
            try:
                os.chdir(temporary_directory)
                model = HDCTI(conf, training, test, '[1]')
                model.readConfiguration()
                model.initModel()
                self.assertTrue(model.counterfactual_context['enabled'])
                self.assertEqual(
                    int(np.sum(model.counterfactual_donor_eligible)), 4
                )
                for source_index in range(model.num_compounds):
                    source_herbs = set(model.compound_herbs[source_index])
                    for donor_index in model.counterfactual_donor_indices[source_index]:
                        self.assertFalse(
                            source_herbs & set(model.compound_herbs[donor_index])
                        )
                model.trainModel()
                self.assertTrue(np.all(np.isfinite(model.u)))
                self.assertGreater(
                    float(np.linalg.norm(model.weight['context_herb_protein'])),
                    0.0,
                )
                model.sess.close()
            finally:
                os.chdir(previous_directory)


if __name__ == '__main__':
    unittest.main()
