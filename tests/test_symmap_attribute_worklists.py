import csv
import tempfile
import unittest
from pathlib import Path

from tools.prepare_symmap_attribute_worklists import prepare_worklists


class SymMapAttributeWorklistTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.alignment = self.root / "alignment"
        self.alignment.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_csv(self, name, fieldnames, rows):
        with (self.alignment / name).open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_routes_and_threshold_gaps_are_deterministic(self):
        compound_fields = [
            "local_entity_id", "official_entity_id", "canonical_name",
            "molecular_formula", "pubchem_id", "cas_id", "tcmsp_id"
        ]
        self.write_csv("compound_alignment.csv", compound_fields, [
            {"local_entity_id": 1, "official_entity_id": 1,
             "canonical_name": "A", "pubchem_id": "11"},
            {"local_entity_id": 2, "official_entity_id": 2,
             "canonical_name": "B", "molecular_formula": "H2O"},
            {"local_entity_id": 3, "official_entity_id": 3,
             "canonical_name": "C"},
        ])
        protein_fields = [
            "local_entity_id", "official_entity_id", "canonical_name",
            "gene_symbol", "protein_name", "uniprot_id", "ensembl_id",
            "ncbi_id", "genbank_protein_id"
        ]
        self.write_csv("protein_alignment.csv", protein_fields, [
            {"local_entity_id": 10, "official_entity_id": 10,
             "gene_symbol": "P1", "uniprot_id": "P00519"},
            {"local_entity_id": 11, "official_entity_id": 11,
             "gene_symbol": "P2", "ensembl_id": "ENSG2"},
            {"local_entity_id": 12, "official_entity_id": 12,
             "gene_symbol": "P3"},
        ])

        report, _ = prepare_worklists(
            self.alignment, self.root / "output",
            minimum_smiles=2.0 / 3.0, minimum_sequence=2.0 / 3.0
        )

        self.assertEqual(
            report["compound"]["route_counts"]["direct_pubchem_cid"], 1
        )
        self.assertEqual(
            report["compound"]["route_counts"]["name_formula_pubchem_lookup"],
            1,
        )
        self.assertEqual(report["compound"]["additional_entities_needed"], 1)
        self.assertEqual(report["protein"]["ensembl_candidates"], 1)
        self.assertEqual(report["protein"]["minimum_ensembl_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
