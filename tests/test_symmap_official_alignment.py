import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.audit_symmap_official_alignment import (
    align_ids,
    find_column,
    metadata_coverage,
    normalize_identifier,
    numeric_identifier,
    read_first_sheet,
)


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
</Types>"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""

WORKBOOK = """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""

WORKBOOK_RELS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""

SHEET = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="inlineStr"><is><t>Ingredient_id</t></is></c><c r="B1" t="inlineStr"><is><t>Molecule_name</t></is></c></row>
    <row r="2"><c r="A2" t="inlineStr"><is><t>SMIT00001</t></is></c><c r="B2" t="inlineStr"><is><t>Alpha</t></is></c></row>
    <row r="3"><c r="A3" t="inlineStr"><is><t>SMIT00002</t></is></c><c r="B3" t="inlineStr"><is><t>Beta</t></is></c></row>
  </sheetData>
</worksheet>"""


def make_xlsx(path):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("_rels/.rels", ROOT_RELS)
        archive.writestr("xl/workbook.xml", WORKBOOK)
        archive.writestr("xl/_rels/workbook.xml.rels", WORKBOOK_RELS)
        archive.writestr("xl/worksheets/sheet1.xml", SHEET)


class SymMapOfficialAlignmentTest(unittest.TestCase):
    def test_xlsx_reader_supports_inline_strings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.xlsx"
            make_xlsx(path)
            headers, records = read_first_sheet(path)

        self.assertEqual(headers, ["Ingredient_id", "Molecule_name"])
        self.assertEqual(records[0]["Ingredient_id"], "SMIT00001")
        self.assertEqual(records[1]["Molecule_name"], "Beta")

    def test_unique_numeric_suffix_alignment(self):
        records = [
            {"Ingredient_id": "SMIT00001"},
            {"Ingredient_id": "SMIT00002"},
        ]
        result = align_ids({"1", "2", "3"}, records, "Ingredient_id", "SMIT")

        self.assertEqual(len(result["matched"]), 2)
        self.assertEqual(result["matched"][0][1], "unique_numeric_suffix")
        self.assertEqual(result["unmatched"], ["3"])

    def test_ambiguous_suffix_is_not_accepted(self):
        records = [
            {"Ingredient_id": "SMIT00001"},
            {"Ingredient_id": "1"},
        ]
        result = align_ids({"1"}, records, "Ingredient_id", "SMIT")

        self.assertEqual(result["matched"][0][1], "exact")
        self.assertEqual(result["numeric_suffix_collisions"], 1)

    def test_identifier_and_header_normalization(self):
        self.assertEqual(normalize_identifier("12.0"), "12")
        self.assertEqual(numeric_identifier("SMTT-00094", "SMTT"), "94")
        self.assertEqual(
            find_column(["Target ID", "Gene symbol"], ("Target_id",)),
            "Target ID",
        )
        self.assertEqual(
            find_column(["Mol_id", "Molecule_name"], ("Ingredient_id", "Mol_id")),
            "Mol_id",
        )

    def test_metadata_coverage_accepts_real_v2_formula_header(self):
        records = [{
            "Mol_id": "1", "Molecule_name": "A", "Molecule_formula": "H2O"
        }]
        audit = {
            "entity_type": "compound",
            "official_id_column": "Mol_id",
            "used_alignment": {"matched": [("1", "exact", records[0])]},
        }

        coverage = metadata_coverage(
            audit, ["Mol_id", "Molecule_name", "Molecule_formula"]
        )

        self.assertEqual(coverage["molecular_formula"]["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
