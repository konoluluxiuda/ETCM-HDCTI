import json
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


class SupportRouterModelSmokeTest(unittest.TestCase):
    def test_router_trains_with_zero_support_pseudo_compounds(self):
        os.environ['HDCTI_FORCE_CPU'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow.compat.v1 as tf

        from HDCTI import HDCTI

        tf.reset_default_graph()
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            hc_rows = []
            cp_rows = []
            training = []
            for index in range(11):
                compound = 'c%d' % index
                protein = 'p%d' % (index % 2)
                hc_rows.append('h%d\t%s' % (index, compound))
                cp_rows.append('%s\t%s' % (compound, protein))
                if index < 10:
                    training.extend([
                        [compound, protein, 1.0],
                        [compound, 'p%d' % ((index + 1) % 2), 0.0],
                    ])
            (dataset_dir / 'H_C.txt').write_text(
                '\n'.join(hc_rows) + '\n', encoding='utf-8'
            )
            (dataset_dir / 'C_P.txt').write_text(
                '\n'.join(cp_rows) + '\n', encoding='utf-8'
            )
            (dataset_dir / 'P_D.txt').write_text(
                'p0\td0\np1\td1\n', encoding='utf-8'
            )
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text(
                '\n'.join('%s\t%s\t1' % tuple(row.split('\t')) for row in cp_rows)
                + '\n',
                encoding='utf-8',
            )
            conf = DummyConf({
                'datapath': str(datapath),
                'ratings.setup': '-columns 0 1 2',
                'evaluation.setup': '-cv 2',
                'experiment.protocol': 'strict',
                'model.name': 'HDCTI',
                'num.factors': '4',
                'num.max.epoch': '1',
                'batch_size': '4',
                'learnRate': '-init 0.005 -max 1',
                'reg.lambda': '-u 0.001 -i 0.001 -b 0.2 -s 0.2',
                'weight.reg': '0.01',
                'pair.decoder': 'dot',
                'context.interaction': 'True',
                'context.compound_disease': 'False',
                'context.herb_protein': 'True',
                'context.herb_protein.mode': 'static',
                'context.herb_disease': 'False',
                'counterfactual.context': 'False',
                'support.router': 'True',
                'support.router.mode': 'monotonic_residual',
                'support.router.pseudo.cold.ratio': '0.1',
                'support.router.seed': '17',
                'support.router.initial.slope': '1.0',
                'attention.max.nodes': '0',
                'output.setup': 'off -dir ./results/',
                'gpu.allow_growth': 'False',
                'gpu.log_device_placement': 'False',
            })
            test = [['c10', 'p0', 1.0], ['c10', 'p1', 0.0]]
            previous_directory = os.getcwd()
            try:
                os.chdir(temporary_directory)
                model = HDCTI(conf, training, test, '[1]')
                model.readConfiguration()
                model.initModel()
                selected_id = next(iter(model.data.pseudo_cold_compounds))
                selected_index = model.data.compound[selected_id]
                seen_index = next(
                    model.data.compound['c%d' % index]
                    for index in range(10)
                    if 'c%d' % index not in model.data.pseudo_cold_compounds
                )
                model.trainModel()

                gates = model.supportContextGateValues(
                    {'weights': model.weight},
                    [selected_index, seen_index, model.data.compound['c10']],
                )
                self.assertAlmostEqual(float(gates[0]), 1.0, places=6)
                self.assertLess(float(gates[1]), 1.0)
                self.assertAlmostEqual(float(gates[2]), 1.0, places=6)
                self.assertGreater(
                    model.support_router_summary['learned_slope'], 0.0
                )
                pair_compounds = [selected_index, seen_index]
                pair_proteins = [
                    model.data.protein['p0'], model.data.protein['p1']
                ]
                pair_scores = model.predictForPairs(
                    pair_compounds, pair_proteins
                )
                ranking_scores = model.predictForRanking()
                np.testing.assert_allclose(
                    pair_scores,
                    ranking_scores[pair_compounds, pair_proteins],
                    rtol=1e-6,
                    atol=1e-6,
                )
                metadata_paths = list(
                    (dataset_dir / 'saved_model').rglob('support_router.json')
                )
                self.assertEqual(len(metadata_paths), 1)
                metadata = json.loads(
                    metadata_paths[0].read_text(encoding='utf-8')
                )
                self.assertEqual(
                    metadata['pseudo_cold']['selected_count'], 1
                )
                model.sess.close()
            finally:
                os.chdir(previous_directory)


if __name__ == '__main__':
    unittest.main()
