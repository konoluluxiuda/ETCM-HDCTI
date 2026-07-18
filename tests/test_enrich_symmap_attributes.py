import json
import unittest

from tools.enrich_symmap_attributes import (
    filter_routes,
    formula_match,
    parse_formula,
    parse_uniprot_tsv,
    pubchem_candidate_cids,
    pubchem_property_map,
    select_uniprot_candidate,
)


class SymMapAttributeEnrichmentTest(unittest.TestCase):
    def test_pubchem_property_and_candidate_parsing(self):
        properties = json.dumps({
            "PropertyTable": {"Properties": [{
                "CID": 5280445,
                "ConnectivitySMILES": "C1=CC=C(C=C1)O",
                "SMILES": "C1=CC=C(C=C1)O",
                "MolecularFormula": "C6H6O",
            }]}
        }).encode("utf-8")
        candidates = json.dumps({
            "IdentifierList": {"CID": [5280445, 1]}
        }).encode("utf-8")

        self.assertIn("5280445", pubchem_property_map(properties))
        self.assertEqual(pubchem_candidate_cids(candidates), ["5280445", "1"])

    def test_formula_match_accepts_one_source_variant(self):
        self.assertEqual(
            formula_match("C15H10O6|C15H11O6", "C15H10O6"), "exact"
        )
        self.assertEqual(formula_match("C15H10O6", "C15H9O6"), "mismatch")
        self.assertEqual(formula_match("", "C15H10O6"), "not_available")

    def test_formula_match_compares_elemental_composition(self):
        self.assertEqual(
            formula_match("NaCl", "ClNa"), "composition_equivalent"
        )
        self.assertEqual(
            formula_match("Na2SO4", "Na2O4S"), "composition_equivalent"
        )
        self.assertEqual(
            formula_match("C20H14NO4", "C20H14NO4+"),
            "composition_match_charge_diff",
        )
        self.assertEqual(formula_match("c5h5n", "C5H5N"), "exact")

    def test_formula_parser_is_conservative_for_complex_notation(self):
        self.assertEqual(parse_formula("C6H5O7-3"), (
            {"C": 6, "H": 5, "O": 7}, "-3"
        ))
        self.assertIsNone(parse_formula("C6H12O6.H2O"))
        self.assertEqual(
            formula_match("C6H12O6.H2O", "C6H14O7"),
            "invalid_source_formula",
        )

    def test_uniprot_tsv_parser_and_human_reviewed_selection(self):
        payload = (
            "From\tEntry\tEntry Name\tReviewed\tProtein names\tGene Names\t"
            "Organism\tOrganism (ID)\tLength\tSequence\n"
            "ENSG1\tP11111\tA_HUMAN\treviewed\tProtein A\tGENE1\t"
            "Homo sapiens (Human)\t9606\t4\tAAAA\n"
            "ENSG1\tQ22222\tA_MOUSE\treviewed\tProtein A\tGene1\t"
            "Mus musculus (Mouse)\t10090\t4\tBBBB\n"
        ).encode("utf-8")

        rows = parse_uniprot_tsv(payload)
        selected, status, candidates = select_uniprot_candidate(rows, "9606")

        self.assertEqual(selected["accession"], "P11111")
        self.assertEqual(status, "resolved_uniprot")
        self.assertEqual(candidates, ["P11111"])

    def test_uniprot_conflicting_sequences_remain_ambiguous(self):
        rows = [
            {"accession": "P1", "sequence": "AAAA", "reviewed": "reviewed",
             "organism_id": "9606"},
            {"accession": "P2", "sequence": "BBBB", "reviewed": "reviewed",
             "organism_id": "9606"},
        ]

        selected, status, candidates = select_uniprot_candidate(rows, "9606")

        self.assertIsNone(selected)
        self.assertEqual(status, "ambiguous_uniprot_candidates")
        self.assertEqual(candidates, ["P1", "P2"])

    def test_route_filter_is_explicit(self):
        rows = [
            {"enrichment_route": "direct_uniprot"},
            {"enrichment_route": "ensembl_to_uniprot"},
        ]

        self.assertEqual(
            filter_routes(rows, ["ensembl_to_uniprot"]), [rows[1]]
        )
        self.assertEqual(filter_routes(rows, []), rows)


if __name__ == "__main__":
    unittest.main()
