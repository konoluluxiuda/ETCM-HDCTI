import unittest

from tools.audit_tcmsp_etcm_attribute_feasibility import (
    metric,
    systematic_degree_sample,
    table_value,
    wilson_interval,
)


class TcmspEtcmAttributeFeasibilityTest(unittest.TestCase):
    def test_systematic_sample_is_deterministic_and_degree_spread(self):
        degrees = {str(index): index for index in range(1, 101)}

        sample = systematic_degree_sample(degrees, 4)

        self.assertEqual(sample, ["13", "38", "63", "88"])

    def test_wilson_interval_and_decision(self):
        lower, upper = wilson_interval(100, 100)

        self.assertGreater(lower, 0.96)
        self.assertEqual(upper, 1.0)
        self.assertEqual(metric(100, 100, 0.95)["decision"], "go")
        self.assertEqual(metric(10, 100, 0.70)["decision"], "no_go")

    def test_tcmsp_table_parser(self):
        page = (
            "<table><tr><th>Pubchem Cid</th><td><a>6251</a></td></tr>"
            "<tr><th>Target name</th><td>Protein A</td></tr></table>"
        )

        self.assertEqual(table_value(page, "Pubchem Cid"), "6251")
        self.assertEqual(table_value(page, "Target name"), "Protein A")


if __name__ == "__main__":
    unittest.main()
