import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / 'tools' / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chembl = load_script('audit_candidate_chembl.py')
literature = load_script('search_candidate_literature.py')
sampling = load_script('prepare_top_candidate_validation.py')


class CandidateValidationTest(unittest.TestCase):
    def test_target_matching_prefers_exact_human_single_protein(self):
        requested = 'Prostaglandin G/H synthase 2'
        family = {
            'pref_name': 'Cyclooxygenase',
            'target_type': 'PROTEIN FAMILY',
            'organism': 'Homo sapiens',
            'score': 100,
            'target_components': [{'component_description': requested}],
        }
        single = {
            'pref_name': requested,
            'target_type': 'SINGLE PROTEIN',
            'organism': 'Homo sapiens',
            'score': 20,
            'target_components': [],
        }
        self.assertGreater(
            chembl.target_match_score(single, requested)[0],
            chembl.target_match_score(family, requested)[0],
        )

    def test_target_matching_uses_explicit_p450cam_organism_override(self):
        def target(organism):
            return {
                'pref_name': 'Camphor 5-monooxygenase',
                'target_type': 'SINGLE PROTEIN',
                'organism': organism,
                'score': 100,
                'target_components': [
                    {'component_description': 'Cytochrome P450-cam'}
                ],
            }

        bacterial = chembl.target_match_score(
            target('Pseudomonas putida'), 'Cytochrome P450-cam'
        )[0]
        human = chembl.target_match_score(
            target('Homo sapiens'), 'Cytochrome P450-cam'
        )[0]
        self.assertGreater(bacterial, human)

    def test_activity_thresholds_and_direct_negative(self):
        positive = {
            'assay_type': 'B',
            'standard_type': 'IC50',
            'standard_value': '10000',
            'standard_units': 'nM',
            'standard_relation': '=',
        }
        negative = {
            'assay_type': 'B',
            'activity_comment': 'Inhibition < 50% @ 10 uM and thus not active',
        }
        self.assertEqual(chembl.classify_activity(positive), 'credible_positive')
        self.assertEqual(chembl.classify_activity(negative), 'direct_negative')

    def test_wilson_interval_matches_fixed_sample_bounds(self):
        strict = chembl.wilson_interval(0, 30)
        liberal = chembl.wilson_interval(1, 30)
        self.assertEqual(strict[1], 0.11351339317396876)
        self.assertEqual(liberal[1], 0.16670390991409173)

    def test_literature_query_includes_compound_and_target_aliases(self):
        query = literature.build_query(
            'cyanidin 3-glucoside', 'Prostaglandin G/H synthase 2'
        )
        self.assertIn('cyanidin-3-O-glucoside', query)
        self.assertIn('PTGS2', query)

    def test_systematic_sample_uses_first_unlabeled_per_compound(self):
        rows = [
            {'compound_id': '1', 'rank': '1', 'label_status': 'test_positive'},
            {'compound_id': '1', 'rank': '2', 'label_status': 'unlabeled'},
            {'compound_id': '2', 'rank': '2', 'label_status': 'unlabeled'},
            {'compound_id': '2', 'rank': '1', 'label_status': 'unlabeled'},
            {'compound_id': '3', 'rank': '1', 'label_status': 'unlabeled'},
        ]
        population, indices, selected = sampling.systematic_sample(rows, 2)
        self.assertEqual([row['rank'] for row in population], ['2', '1', '1'])
        self.assertEqual(indices, [0, 1])
        self.assertEqual([row['compound_id'] for row in selected], ['1', '2'])


if __name__ == '__main__':
    unittest.main()
