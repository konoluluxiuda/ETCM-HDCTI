import csv
import tempfile
import unittest
from pathlib import Path

from tools.audit_multidataset_attributes import audit_datasets


class MultiDatasetAttributeAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_cp(self, name):
        dataset = self.root / name
        dataset.mkdir(parents=True)
        (dataset / "C_P.txt").write_text("1\t10\n2\t11\n", encoding="utf-8")
        return dataset

    def make_rich_mapping(self, dataset):
        mappings = dataset / "mappings"
        mappings.mkdir()
        with (mappings / "compound_id_map.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "compound_id", "ingredient_names", "tcmip_ids", "molecular_formula"
            ])
            writer.writeheader()
            writer.writerows([
                {"compound_id": 1, "ingredient_names": "A", "tcmip_ids": "T1",
                 "molecular_formula": "C2H6O"},
                {"compound_id": 2, "ingredient_names": "B", "tcmip_ids": "T2",
                 "molecular_formula": "C3H8O"},
            ])
        with (mappings / "protein_id_map.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "protein_id", "uniprot_accessions"
            ])
            writer.writeheader()
            writer.writerows([
                {"protein_id": 10, "uniprot_accessions": "P00519"},
                {"protein_id": 11, "uniprot_accessions": "Q04771"},
            ])

    def test_anonymous_and_missing_mappings_block_cross_dataset_route(self):
        rich = self.make_cp("rich")
        self.make_rich_mapping(rich)
        anonymous = self.make_cp("anonymous")
        (anonymous / "compound_id_all.txt").write_text("1\n2\n", encoding="utf-8")
        (anonymous / "protein_id_all.txt").write_text("10\n11\n", encoding="utf-8")
        missing = self.make_cp("missing")

        report = audit_datasets({
            "rich": rich,
            "anonymous": anonymous,
            "missing": missing,
        })

        self.assertEqual(report["decision"], "blocked_cross_dataset_entity_alignment")
        decisions = {row["name"]: row["decision"] for row in report["datasets"]}
        self.assertEqual(decisions["rich"], "pending_external_enrichment")
        self.assertEqual(
            decisions["anonymous"], "blocked_missing_biological_mapping"
        )
        self.assertEqual(decisions["missing"], "blocked_missing_biological_mapping")

    def test_three_enriched_datasets_enable_shared_pilot(self):
        datasets = {}
        alignment = self.root / "alignment"
        for index in range(3):
            name = "dataset%d" % index
            dataset = self.make_cp(name)
            datasets[name] = dataset
            target = alignment / name
            target.mkdir(parents=True)
            with (target / "compound_attributes.csv").open(
                    "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "entity_id", "canonical_smiles", "formula_match"
                ])
                writer.writeheader()
                writer.writerows([
                    {"entity_id": 1, "canonical_smiles": "CCO", "formula_match": "yes"},
                    {"entity_id": 2, "canonical_smiles": "CCCO", "formula_match": "yes"},
                ])
            with (target / "protein_attributes.csv").open(
                    "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "entity_id", "uniprot_accession", "sequence"
                ])
                writer.writeheader()
                writer.writerows([
                    {"entity_id": 10, "uniprot_accession": "P00519", "sequence": "AAAA"},
                    {"entity_id": 11, "uniprot_accession": "Q04771", "sequence": "BBBB"},
                ])

        report = audit_datasets(datasets, alignment_root=alignment)

        self.assertEqual(report["decision"], "supports_cross_dataset_multimodal_pilot")
        self.assertEqual(report["ready_datasets"], 3)

    def test_formula_verification_excludes_unavailable_source_formulas(self):
        dataset = self.make_cp("formula_audit")
        alignment = self.root / "alignment"
        target = alignment / "formula_audit"
        target.mkdir(parents=True)
        with (target / "compound_attributes.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "entity_id", "canonical_smiles", "formula_match"
            ])
            writer.writeheader()
            writer.writerows([
                {"entity_id": 1, "canonical_smiles": "CCO",
                 "formula_match": "composition_equivalent"},
                {"entity_id": 2, "canonical_smiles": "CCCO",
                 "formula_match": "not_available"},
            ])
        with (target / "protein_attributes.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "entity_id", "uniprot_accession", "sequence"
            ])
            writer.writeheader()
            writer.writerows([
                {"entity_id": 10, "uniprot_accession": "P00519",
                 "sequence": "AAAA"},
                {"entity_id": 11, "uniprot_accession": "Q04771",
                 "sequence": "BBBB"},
            ])

        report = audit_datasets(
            {"formula_audit": dataset}, alignment_root=alignment
        )
        actual = report["datasets"][0]["actual_attributes"]

        self.assertEqual(actual["formula_checkable_entities"], 1)
        self.assertEqual(actual["formula_verified_entities"], 1)
        self.assertEqual(actual["formula_unavailable_entities"], 1)
        self.assertEqual(actual["formula_verification_rate_among_checkable"], 1.0)

    def test_tcmsp_query_labels_are_biological_lookup_ids(self):
        dataset = self.make_cp("tcmsp")
        (dataset / "compound_id_all.csv").write_text(
            "1,640001\nmolecule_ID,640002\n2,640003\n", encoding="utf-8"
        )
        (dataset / "protein_id_all.csv").write_text(
            "10,17000\ntarget_ID,17001\n11,17002\n", encoding="utf-8"
        )

        report = audit_datasets({"TCMSP": dataset})
        row = report["datasets"][0]

        self.assertEqual(
            row["local_mapping"]["mapping_type"], "source_database_query_ids"
        )
        self.assertEqual(
            row["local_mapping"]["compound_biological_lookup_coverage"], 1.0
        )
        self.assertEqual(
            row["local_mapping"]["protein_biological_lookup_coverage"], 1.0
        )
        self.assertEqual(row["decision"], "pending_external_enrichment")

    def test_verified_official_alignment_unblocks_biological_lookup(self):
        dataset = self.make_cp("symmap")
        mapping = self.root / "symmap_mapping"
        mapping.mkdir()
        with (mapping / "compound_alignment.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "local_entity_id", "source_entity_id", "canonical_name",
                "pubchem_id"
            ])
            writer.writeheader()
            writer.writerows([
                {"local_entity_id": 1, "source_entity_id": 1,
                 "canonical_name": "A", "pubchem_id": "1"},
                {"local_entity_id": 2, "source_entity_id": 2,
                 "canonical_name": "B", "pubchem_id": "2"},
            ])
        with (mapping / "protein_alignment.csv").open(
                "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "local_entity_id", "source_entity_id", "gene_symbol",
                "ensembl_id"
            ])
            writer.writeheader()
            writer.writerows([
                {"local_entity_id": 10, "source_entity_id": 10,
                 "gene_symbol": "P1", "ensembl_id": "ENSG1"},
                {"local_entity_id": 11, "source_entity_id": 11,
                 "gene_symbol": "P2", "ensembl_id": "ENSG2"},
            ])

        report = audit_datasets(
            {"SymMap2.0": dataset},
            mapping_overrides={"SymMap2.0": mapping},
        )
        row = report["datasets"][0]

        self.assertEqual(
            row["local_mapping"]["mapping_type"],
            "verified_official_biological_metadata",
        )
        self.assertEqual(
            row["local_mapping"]["protein_biological_lookup_coverage"], 1.0
        )
        self.assertEqual(row["decision"], "pending_external_enrichment")

    def test_three_pending_datasets_move_overall_state_to_enrichment(self):
        datasets = {}
        for index in range(3):
            name = "pending%d" % index
            dataset = self.make_cp(name)
            self.make_rich_mapping(dataset)
            datasets[name] = dataset
        datasets["blocked"] = self.make_cp("blocked")

        report = audit_datasets(datasets)

        self.assertEqual(report["decision"], "pending_cross_dataset_enrichment")


if __name__ == "__main__":
    unittest.main()
