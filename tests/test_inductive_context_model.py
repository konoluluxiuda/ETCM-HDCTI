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


class InductiveContextModelTest(unittest.TestCase):
    def test_self_excluded_context_and_base_gate_match_numpy_inference(self):
        os.environ['HDCTI_FORCE_CPU'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import tensorflow.compat.v1 as tf

        from HDCTI import HDCTI
        from util.self_excluded_context import (
            build_direct_self_excluded_contexts,
            indexed_hc_memberships,
        )

        tf.reset_default_graph()
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_dir = Path(temporary_directory)
            hc_path = dataset_dir / 'H_C.txt'
            hc_path.write_text(
                'h0\tc0\nh0\tc1\nh1\tc2\n', encoding='utf-8'
            )
            (dataset_dir / 'C_P.txt').write_text(
                'c0\tp0\nc1\tp1\nc2\tp0\n', encoding='utf-8'
            )
            (dataset_dir / 'P_D.txt').write_text(
                'p0\td0\np1\td1\n', encoding='utf-8'
            )
            datapath = dataset_dir / 'ONE_indices.txt'
            datapath.write_text(
                'c0\tp0\t1\nc1\tp1\t1\nc2\tp0\t1\n',
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
                'context.interaction': 'True',
                'context.compound_disease': 'False',
                'context.herb_protein': 'True',
                'context.herb_protein.mode': 'static',
                'context.herb_disease': 'False',
                'counterfactual.context': 'False',
                'context.mask.training': 'False',
                'support.router': 'False',
                'hyperedge.attention': 'False',
                'global.token.attention': 'False',
                'inductive.context': 'True',
                'inductive.context.suppress.base.zero.support': 'True',
                'inductive.context.self.excluded': 'True',
                'attention.max.nodes': '0',
                'output.setup': 'off -dir ./results/',
                'gpu.allow_growth': 'False',
                'gpu.log_device_placement': 'False',
            })
            training = [
                ['c0', 'p0', 1.0],
                ['c0', 'p1', 0.0],
            ]
            test = [
                ['c1', 'p1', 1.0],
                ['c1', 'p0', 0.0],
                ['c2', 'p0', 1.0],
                ['c2', 'p1', 0.0],
            ]
            model = HDCTI(conf, training, test, '[1]')
            model.readConfiguration()
            model.initModel()
            model.sess.run(tf.global_variables_initializer())
            state = model.fetchModelState(include_context_audit=True)

            memberships = indexed_hc_memberships(
                hc_path,
                model.data.herb,
                model.data.compound,
            )
            expected = build_direct_self_excluded_contexts(
                state['herb_edge'],
                state['hc_edge_inputs'],
                memberships['herb_members'],
                memberships['compound_herbs'],
                model.num_compounds,
            )
            np.testing.assert_allclose(
                state['compound_context'],
                expected['self_excluded_contexts'],
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                state['inclusive_compound_context'],
                expected['inclusive_contexts'],
                rtol=1e-5,
                atol=1e-5,
            )

            compound_indices = np.asarray([
                model.data.compound['c0'],
                model.data.compound['c1'],
                model.data.compound['c2'],
            ])
            np.testing.assert_array_equal(
                model.inductiveBaseGateValues(compound_indices),
                [1.0, 0.0, 1.0],
            )

            protein_indices = np.asarray([
                model.data.protein['p0'],
                model.data.protein['p1'],
                model.data.protein['p0'],
            ])
            graph_logits = model.sess.run(
                model.buildPairLogits(),
                feed_dict={
                    model.u_idx: compound_indices,
                    model.v_idx: protein_indices,
                    model.isTraining: 0,
                },
            )
            model.u = state['compound']
            model.i = state['protein']
            model.u_context = state['compound_context']
            model.i_context = state['protein_context']
            model.herb_edge = state['herb_edge']
            model.weight = state['weights']
            numpy_logits = model.predictForPairs(
                compound_indices, protein_indices
            )
            model.sess.close()

        np.testing.assert_allclose(
            numpy_logits, graph_logits, rtol=1e-5, atol=1e-5
        )


if __name__ == '__main__':
    unittest.main()
