import unittest

from util.etcm_topk_cases import (
    assign_rank_tertiles,
    evidence_classification,
    pair_context_paths,
    rank_candidates,
    resolve_raw_page,
    select_cases,
)


def candidate(compound_id, degree, evidence, herbs=1, identifier=True):
    return {
        'compound_id': str(compound_id),
        'tcmip_id': 'TCMIP-I-%05d' % int(compound_id),
        'compound_name': 'compound-%s' % compound_id,
        'model_train_cp_degree': degree,
        'unseen_confirmed_targets': evidence,
        'confirmed_evidence_count': evidence * 2,
        'independent_herb_count': herbs,
        'mention_count': degree * 10,
        'has_identity_identifier': identifier,
    }


class EtcmTopkCaseTests(unittest.TestCase):

    def test_selection_is_deterministic_and_stratified(self):
        candidates = [
            candidate(index, degree=index // 3, evidence=(index % 4) + 1)
            for index in range(1, 31)
        ]
        eligible_a, selected_a, allocation = select_cases(
            candidates, case_count=5, seed=2026
        )
        eligible_b, selected_b, _ = select_cases(
            list(reversed(candidates)), case_count=5, seed=2026
        )
        self.assertEqual(
            [row['compound_id'] for row in selected_a],
            [row['compound_id'] for row in selected_b],
        )
        self.assertEqual(
            {'low': 2, 'medium': 1, 'high': 2},
            allocation,
        )
        self.assertEqual(
            {'low': 2, 'medium': 1, 'high': 2},
            {
                stratum: sum(
                    row['support_stratum'] == stratum for row in selected_a
                )
                for stratum in ('low', 'medium', 'high')
            },
        )
        self.assertEqual(30, len(eligible_a))

    def test_rank_ties_use_protein_id(self):
        ranked = rank_candidates(['20', '3', '11'], [0.5, 0.5, 0.8])
        self.assertEqual(['11', '20', '3'], [
            row['protein_id'] for row in ranked
        ])
        self.assertGreater(ranked[0]['score'], ranked[1]['score'])

    def test_evidence_levels_keep_potential_separate(self):
        confirmed = {('1', '10'): {}}
        potential = {('1', '11'): {}, ('1', '10'): {}}
        self.assertEqual(
            ('A', 'confirmed_unseen'),
            evidence_classification(('1', '10'), confirmed, potential),
        )
        self.assertEqual(
            ('C', 'potential_target'),
            evidence_classification(('1', '11'), confirmed, potential),
        )
        self.assertEqual(
            ('E', 'no_etcm_target_evidence'),
            evidence_classification(('1', '12'), confirmed, potential),
        )

    def test_rank_tertiles_cover_every_candidate(self):
        rows = assign_rank_tertiles([
            candidate(index, degree=index % 5, evidence=1)
            for index in range(12)
        ])
        self.assertEqual(12, len(rows))
        self.assertEqual(
            {'low', 'medium', 'high'},
            {row['support_stratum'] for row in rows},
        )

    def test_pair_context_paths_intersect_herb_and_target_diseases(self):
        paths = pair_context_paths(
            'c1',
            'p1',
            {'c1': {'h1', 'h2'}},
            {'h1': {'d1', 'd2'}, 'h2': {'d2', 'd3'}},
            {'p1': {'d2', 'd4'}},
        )
        self.assertEqual([('h1', 'd2'), ('h2', 'd2')], paths)

    def test_raw_page_resolution_uses_normalized_names(self):
        index = {'sodiumdependentdopaminetransporter': ['target.json']}
        self.assertEqual(
            'target.json',
            resolve_raw_page(
                index, ['Sodium-dependent dopamine transporter']
            ),
        )


if __name__ == '__main__':
    unittest.main()
