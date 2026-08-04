import unittest

from util.config import ModelConf
from util.model_components import resolve_hplga


PAIRS = [
    (
        'configs/HDCTI_tcmsuite_pair_stratified_herb_only_pilot.conf',
        'configs/HDCTI_tcmsuite_pair_stratified_hplga_pilot.conf',
    ),
    (
        'configs/HDCTI_tcmsp_pair_stratified_herb_only_pilot.conf',
        'configs/HDCTI_tcmsp_pair_stratified_hplga_pilot.conf',
    ),
    (
        'configs/HDCTI_symmap_pair_stratified_herb_only_pilot.conf',
        'configs/HDCTI_symmap_pair_stratified_hplga_pilot.conf',
    ),
    (
        'configs/HDCTI_etcm_mention10_pair_stratified_herb_only_pilot.conf',
        'configs/HDCTI_etcm_mention10_pair_stratified_hplga_pilot.conf',
    ),
]

HPLGA_KEYS = {
    'hplga.enabled',
    'hplga.mode',
    'hplga.kernel',
    'hplga.hc',
    'hplga.pd',
    'hplga.heads',
    'hplga.pagerank.alpha',
    'hplga.pagerank.max.iter',
    'hplga.pagerank.tol',
    'hplga.epsilon',
}


class HPLGAConfigTest(unittest.TestCase):
    def test_candidates_change_only_variant_and_hplga_settings(self):
        for baseline_path, candidate_path in PAIRS:
            baseline = ModelConf(baseline_path)
            candidate = ModelConf(candidate_path)
            baseline_values = dict(baseline.config)
            candidate_values = dict(candidate.config)
            baseline_values.pop('model.variant')
            candidate_values.pop('model.variant')
            for key in HPLGA_KEYS:
                candidate_values.pop(key)
            self.assertEqual(
                baseline_values,
                candidate_values,
                msg=candidate_path,
            )

            settings = resolve_hplga(candidate)
            self.assertTrue(settings['enabled'])
            self.assertTrue(settings['hc_enabled'])
            self.assertTrue(settings['pd_enabled'])
            self.assertEqual(settings['heads'], 2)
            self.assertEqual(candidate['attention.max.nodes'], '0')
            self.assertEqual(candidate['evaluation.fold.limit'], '1')
            self.assertEqual(candidate['evaluation.outer.test'], 'False')


if __name__ == '__main__':
    unittest.main()
