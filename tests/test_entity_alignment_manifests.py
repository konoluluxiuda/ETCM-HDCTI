import csv
import tempfile
import unittest
from pathlib import Path

from tools.prepare_entity_alignment_manifests import prepare_manifests


class EntityAlignmentManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_dataset(self, name):
        dataset = self.root / name
        dataset.mkdir(parents=True)
        (dataset / "C_P.txt").write_text(
            "1\t10\n1\t11\n2\t10\n", encoding="utf-8"
        )
        return dataset

    def make_rich_mappings(self, dataset):
        mappings = dataset / "mappings"
        mappings.mkdir()
        with (mappings / "compound_id_map.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "compound_id", "ingredient_names", "tcmip_ids",
                "molecular_formula"
            ])
            writer.writeheader()
            writer.writerows([
                {"compound_id": 1, "ingredient_names": "A",
                 "tcmip_ids": "TCMIP-I-1", "molecular_formula": "C2H6O"},
                {"compound_id": 2, "ingredient_names": "B",
                 "tcmip_ids": "TCMIP-I-2", "molecular_formula": "C3H8O"},
            ])
        with (mappings / "protein_id_map.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "protein_id", "protein_key", "gene_symbols",
                "uniprot_accessions", "target_names", "organisms"
            ])
            writer.writeheader()
            writer.writerows([
                {"protein_id": 10, "protein_key": "ABL1", "gene_symbols": "ABL1",
                 "uniprot_accessions": "P00519", "target_names": "ABL1",
                 "organisms": "Homo sapiens"},
                {"protein_id": 11, "protein_key": "GENE2", "gene_symbols": "GENE2",
                 "uniprot_accessions": "", "target_names": "Target 2",
                 "organisms": "Homo sapiens"},
            ])

    def test_rich_metadata_is_actionable_and_degree_ranked(self):
        dataset = self.make_dataset("rich")
        self.make_rich_mappings(dataset)

        report, rows = prepare_manifests({"rich": dataset})
        compounds = {
            row["local_entity_id"]: row
            for row in rows if row["entity_type"] == "compound"
        }
        proteins = {
            row["local_entity_id"]: row
            for row in rows if row["entity_type"] == "protein"
        }

        self.assertEqual(compounds["1"]["cp_positive_degree"], 2)
        self.assertEqual(compounds["1"]["priority_rank"], 1)
        self.assertEqual(
            compounds["1"]["review_status"], "ready_for_external_enrichment"
        )
        self.assertEqual(proteins["10"]["canonical_identifier"], "P00519")
        self.assertEqual(
            proteins["11"]["review_status"], "needs_uniprot_resolution"
        )
        self.assertEqual(report["datasets"][0]["compound"]["source_ready_entities"], 2)

    def test_anonymous_crosswalk_does_not_become_canonical_identifier(self):
        dataset = self.make_dataset("anonymous")
        (dataset / "compound_id_all.csv").write_text(
            "1,640001\n2,640002\n", encoding="utf-8"
        )
        (dataset / "protein_id_all.csv").write_text(
            "10,17000\n11,17001\n", encoding="utf-8"
        )

        _, rows = prepare_manifests({"anonymous": dataset})

        self.assertTrue(all(not row["canonical_identifier"] for row in rows))
        self.assertTrue(all(
            row["review_status"] == "blocked_unknown_identifier_namespace"
            for row in rows
        ))
        self.assertEqual(rows[0]["mapping_method"], "anonymous_numeric_crosswalk")

    def test_tcmsp_source_labels_make_local_query_ids_actionable(self):
        dataset = self.make_dataset("tcmsp")
        (dataset / "compound_id_all.csv").write_text(
            "1,640001\nmolecule_ID,640002\n2,640003\n", encoding="utf-8"
        )
        (dataset / "protein_id_all.csv").write_text(
            "10,17000\ntarget_ID,17001\n11,17002\n", encoding="utf-8"
        )

        report, rows = prepare_manifests({"TCMSP": dataset})
        compounds = [row for row in rows if row["entity_type"] == "compound"]
        proteins = [row for row in rows if row["entity_type"] == "protein"]

        self.assertTrue(all(
            row["review_status"] == "ready_for_external_enrichment"
            for row in rows
        ))
        self.assertEqual(compounds[0]["source_entity_id"], "1")
        self.assertEqual(compounds[0]["matrix_entity_id"], "640001")
        self.assertEqual(
            compounds[0]["source_identifier_namespace"], "TCMSP:molecule_query"
        )
        self.assertEqual(
            proteins[0]["source_identifier_namespace"], "TCMSP:target_query"
        )
        self.assertEqual(
            report["datasets"][0]["compound"]["source_ready_coverage"], 1.0
        )

    def test_missing_mapping_is_explicitly_blocked(self):
        dataset = self.make_dataset("missing")

        _, rows = prepare_manifests({"missing": dataset})

        self.assertTrue(all(
            row["review_status"] == "blocked_mapping_missing" for row in rows
        ))

    def test_verified_external_alignment_recovers_anonymous_entities(self):
        dataset = self.make_dataset("symmap")
        alignment = self.root / "symmap_alignment"
        alignment.mkdir()
        fields = [
            "local_entity_id", "source_entity_id",
            "source_identifier_namespace", "canonical_identifier",
            "canonical_name", "molecular_formula", "match_method"
        ]
        with (alignment / "compound_alignment.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([
                {"local_entity_id": "1", "source_entity_id": "1",
                 "source_identifier_namespace": "SymMap:Mol_id",
                 "canonical_identifier": "5280445",
                 "canonical_name": "Luteolin", "molecular_formula": "C15H10O6",
                 "match_method": "exact"},
                {"local_entity_id": "2", "source_entity_id": "2",
                 "source_identifier_namespace": "SymMap:Mol_id",
                 "canonical_identifier": "", "canonical_name": "Unknown",
                 "molecular_formula": "", "match_method": "exact"},
            ])
        with (alignment / "protein_alignment.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([
                {"local_entity_id": "10", "source_entity_id": "10",
                 "source_identifier_namespace": "SymMap:Gene_id",
                 "canonical_identifier": "P00519", "canonical_name": "ABL1",
                 "molecular_formula": "", "match_method": "exact"},
                {"local_entity_id": "11", "source_entity_id": "11",
                 "source_identifier_namespace": "SymMap:Gene_id",
                 "canonical_identifier": "", "canonical_name": "GENE2",
                 "molecular_formula": "", "match_method": "exact"},
            ])

        report, rows = prepare_manifests({"SymMap2.0": dataset}, {
            "SymMap2.0": alignment
        })
        indexed = {
            (row["entity_type"], row["local_entity_id"]): row for row in rows
        }

        self.assertEqual(
            indexed[("compound", "1")]["mapping_method"],
            "verified_external_exact",
        )
        self.assertEqual(
            indexed[("protein", "10")]["canonical_identifier"], "P00519"
        )
        self.assertEqual(
            report["datasets"][0]["compound"]["source_ready_coverage"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
