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


if __name__ == "__main__":
    unittest.main()
