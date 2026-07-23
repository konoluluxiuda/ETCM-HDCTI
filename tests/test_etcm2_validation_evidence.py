import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.build_etcm2_validation_evidence import (
    parse_selected_page,
    summary_from_tail,
    write_validation_sets,
)


class Etcm2ValidationEvidenceTest(unittest.TestCase):
    def test_selective_parser_extracts_only_validation_relations(self):
        page = {
            "code": 1,
            "msg": "success",
            "data": [
                {
                    "key": "Basic Information",
                    "id": "base_information",
                    "value": "",
                    "children": [
                        {
                            "key": "Basic Information",
                            "id": "Basic_Information",
                            "value": [
                                {
                                    "key": "Component Name IN English",
                                    "value": "Example",
                                },
                                {
                                    "key": "2D Structure",
                                    "value": "https://example/TCMIP-I-00001.svg",
                                },
                            ],
                            "children": [],
                        }
                    ],
                },
                {
                    "key": "Related Tables",
                    "id": "related_table",
                    "value": [
                        {
                            "label": "Chinese Patent Drugs",
                            "type": "chinese_patent_drug",
                            "value": [{"large": "ignored"}],
                            "count": 1,
                        },
                        {
                            "label": "Herbs",
                            "type": "herb",
                            "value": [{"Herb Name in Pinyin": ["HerbA"]}],
                            "count": 1,
                        },
                        {
                            "label": "Confirmed Targets",
                            "type": "target",
                            "value": [{"Gene Symbol": ["GENE1"]}],
                            "count": 1,
                        },
                        {
                            "label": "Potential Targets",
                            "type": "similar_target",
                            "value": [
                                {"Gene Symbol": ["GENE2"], "Similar Score": 0.9}
                            ],
                            "count": 1,
                        },
                        {
                            "label": "Enriched Diseases",
                            "type": "disease",
                            "value": [{"large": "ignored"}],
                            "count": 1,
                        },
                    ],
                },
            ],
            "complete_fetch_summary": [
                {
                    "section": "related_table",
                    "type": relation,
                    "expected": 1,
                    "actual": 1,
                    "status": "ok",
                }
                for relation in (
                    "chinese_patent_drug",
                    "herb",
                    "target",
                    "similar_target",
                    "disease",
                )
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Example.json"
            path.write_text(
                json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summary = summary_from_tail(path)
            base, relations = parse_selected_page(path, summary)

        self.assertEqual(base["id"], "base_information")
        self.assertEqual(set(relations), {"herb", "target", "similar_target"})
        self.assertEqual(relations["target"]["value"][0]["Gene Symbol"], ["GENE1"])

    def test_validation_sets_separate_overlap_unseen_and_oov(self):
        confirmed = {
            ("TCMIP-I-00001", "GENE1"): self._pair_record(),
            ("TCMIP-I-00001", "GENE2"): self._pair_record(),
            ("TCMIP-I-99999", "GENE1"): self._pair_record(),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            mappings = dataset / "mappings"
            mappings.mkdir(parents=True)
            self._write_csv(
                mappings / "compound_id_map.csv",
                ("compound_id", "tcmip_ids"),
                [{"compound_id": "1", "tcmip_ids": "TCMIP-I-00001"}],
            )
            self._write_csv(
                mappings / "protein_id_map.csv",
                ("protein_id", "protein_key", "gene_symbols"),
                [
                    {
                        "protein_id": "10",
                        "protein_key": "GENE1",
                        "gene_symbols": "GENE1",
                    },
                    {
                        "protein_id": "20",
                        "protein_key": "GENE2",
                        "gene_symbols": "GENE2",
                    },
                ],
            )
            (dataset / "C_P.txt").write_text("1\t10\n", encoding="utf-8")
            staging = root / "output"
            staging.mkdir()
            result = write_validation_sets(
                staging,
                confirmed,
                {"TCMIP-I-00001": "Example"},
                [dataset],
            )

            self.assertEqual(result["dataset"]["training_overlap"], 1)
            self.assertEqual(result["dataset"]["unseen_confirmed"], 1)
            self.assertEqual(result["dataset"]["out_of_vocabulary"], 1)
            unseen_path = (
                staging / "validation" / "dataset" / "unseen_confirmed.tsv"
            )
            with unseen_path.open(encoding="utf-8") as handle:
                unseen = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(unseen[0]["gene_symbol"], "GENE2")

    @staticmethod
    def _pair_record():
        return {
            "evidence_count": 1,
            "activities": {"IC50:1"},
            "reference_urls": {"https://example/reference"},
            "score_count": 0,
            "score_sum": 0.0,
            "score_min": None,
            "score_max": None,
        }

    @staticmethod
    def _write_csv(path, fields, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
