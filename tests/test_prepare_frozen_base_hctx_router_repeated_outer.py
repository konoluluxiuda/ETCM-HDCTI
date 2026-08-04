import unittest
from pathlib import Path

from tools.prepare_frozen_base_hctx_router_repeated_outer import (
    render_config,
    replace_setting,
)


class RepeatedOuterPreparationTest(unittest.TestCase):
    def test_replace_setting_requires_exactly_one_match(self):
        with self.assertRaises(ValueError):
            replace_setting("other=1\n", "model.variant", "candidate")
        with self.assertRaises(ValueError):
            replace_setting(
                "model.variant=a\nmodel.variant=b\n",
                "model.variant",
                "candidate",
            )

    def test_render_config_changes_only_variant_and_support_manifest(self):
        template = (
            "model.variant=baseline\n"
            "evaluation.setup=-four-state-unit\n"
            "support.four.state.manifest=./old/manifest.json\n"
            "random.seed=2026\n"
            "attention.max.nodes=0\n"
        )
        artifact = (
            Path(__file__).resolve().parents[1]
            / "dataset"
            / "TCMsuite"
            / "unit"
            / "manifest.json"
        )
        rendered = render_config(template, "tcmsuite", 3, artifact)
        self.assertIn(
            "model.variant=tcmsuite_four_state_no_context_c3p3_v1\n",
            rendered,
        )
        self.assertIn(
            "support.four.state.manifest=./dataset/TCMsuite/unit/manifest.json\n",
            rendered,
        )
        self.assertIn("random.seed=2026\n", rendered)
        self.assertIn("attention.max.nodes=0\n", rendered)


if __name__ == "__main__":
    unittest.main()
