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


class HPLGAModelSmokeTest(unittest.TestCase):
    def test_hplga_trains_without_quadratic_attention_and_writes_metadata(self):
        os.environ['HDCTI_FORCE_CPU'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow.compat.v1 as tf

        from HDCTI import HDCTI

        tf.reset_default_graph()
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            (dataset_dir / 'H_C.txt').write_text(
                'h0\tc0\nh0\tc1\nh1\tc1\nh1\tc2\n'
                'h2\tc3\nh2\tc4\n',
                encoding='utf-8',
            )
            (dataset_dir / 'C_P.txt').write_text(
                'c0\tp0\nc1\tp1\nc2\tp2\nc3\tp0\nc4\tp1\n',
                encoding='utf-8',
            )
            (dataset_dir / 'P_D.txt').write_text(
                'p0\td0\np0\td1\np1\td1\np2\td2\n',
                encoding='utf-8',
            )
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text(
                'c0\tp0\t1\nc1\tp1\t1\nc2\tp2\t1\n'
                'c3\tp0\t1\nc4\tp1\t1\n',
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
                'batch_size': '2',
                'learnRate': '-init 0.005 -max 1',
                'reg.lambda': '-u 0.001 -i 0.001 -b 0.2 -s 0.2',
                'weight.reg': '0.01',
                'pair.decoder': 'dot',
                'context.interaction': 'False',
                'counterfactual.context': 'False',
                'hyperedge.attention': 'False',
                'global.token.attention': 'False',
                'hplga.enabled': 'True',
                'hplga.hc': 'True',
                'hplga.pd': 'True',
                'hplga.heads': '2',
                'attention.max.nodes': '0',
                'output.setup': 'off -dir ./results/',
                'gpu.allow_growth': 'False',
                'gpu.log_device_placement': 'False',
            })
            training = [
                ['c0', 'p0', 1.0], ['c0', 'p1', 0.0],
                ['c1', 'p1', 1.0], ['c1', 'p2', 0.0],
                ['c2', 'p2', 1.0], ['c2', 'p0', 0.0],
                ['c3', 'p0', 1.0], ['c3', 'p2', 0.0],
            ]
            test = [['c4', 'p1', 1.0], ['c4', 'p2', 0.0]]
            previous_directory = os.getcwd()
            try:
                os.chdir(temporary_directory)
                model = HDCTI(conf, training, test, '[1]')
                model.readConfiguration()
                model.initModel()
                hplga_operations = [
                    operation
                    for operation in tf.get_default_graph().get_operations()
                    if 'hplga' in operation.name.lower()
                ]
                self.assertTrue(any(
                    'hc_hplga_layer_1/residual' in operation.name
                    for operation in hplga_operations
                ))
                self.assertTrue(any(
                    'pd_hplga_layer_1/kernel_state' in operation.name
                    for operation in hplga_operations
                ))
                for operation in hplga_operations:
                    for output in operation.outputs:
                        shape = output.shape.as_list()
                        self.assertNotEqual(shape, [5, 5])
                        self.assertNotEqual(shape, [3, 3])

                model.trainModel()
                gamma_values = [
                    float(np.asarray(model.weight[name]).reshape(-1)[0])
                    for name in model.weight
                    if '_hplga_gamma_' in name
                ]
                self.assertTrue(all(np.isfinite(gamma_values)))
                self.assertIsNotNone(model.hplga_summary)
                metadata_paths = list(
                    (dataset_dir / 'saved_model').rglob('hplga.json')
                )
                self.assertEqual(len(metadata_paths), 1)
                metadata = json.loads(
                    metadata_paths[0].read_text(encoding='utf-8')
                )
                self.assertTrue(metadata['hc_enabled'])
                self.assertTrue(metadata['pd_enabled'])
                self.assertEqual(
                    metadata['structure']['hc'][
                        'quadratic_attention_elements'
                    ],
                    0,
                )
                model.sess.close()
            finally:
                os.chdir(previous_directory)


if __name__ == '__main__':
    unittest.main()
