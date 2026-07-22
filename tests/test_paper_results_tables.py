import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.build_paper_results_tables import (
    build_markdown,
    check_file,
    parse_summary,
)


def record(dataset, method, value, threshold=False):
    row = {
        'dataset': dataset,
        'method': method,
        'source': '/tmp/source.tsv',
        'config': '/tmp/config.conf',
    }
    for metric in ('AUC', 'AUPR', 'Recall', 'Precision', 'F1-score'):
        row[metric] = value
        row[metric + '_std'] = 0.01
    if threshold:
        row['threshold'] = 0.4
        row['threshold_std'] = 0.02
    return row


class PaperResultsTablesTest(unittest.TestCase):
    def test_parses_both_standard_deviation_formats(self):
        self.assertEqual(parse_summary('0.98(±0.01)'), (0.98, 0.01))
        self.assertEqual(parse_summary('0.98(+-0.01)'), (0.98, 0.01))
        self.assertEqual(parse_summary('0.98'), (0.98, None))

    def test_rejects_changed_frozen_source(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / 'results.tsv'
            path.write_text('first\n', encoding='utf-8')

            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                check_file(path, '0' * 64)

    def test_builds_both_protocol_tables_and_boundaries(self):
        datasets = ['D1', 'D2', 'D3', 'D4']
        random_methods = ['Strict', 'Hctx', 'CHCR']
        cold_methods = ['Hctx', 'SDIS']
        manifest = {
            'datasets': datasets,
            'random_edge': {
                'sources': [{'method': method} for method in random_methods]
            },
            'compound_cold_start': {
                'methods': [{'method': method} for method in cold_methods]
            },
        }
        random_rows = [
            record(dataset, method, 0.8 + index * 0.01)
            for dataset in datasets
            for index, method in enumerate(random_methods)
        ]
        cold_rows = [
            record(dataset, method, 0.7 + index * 0.02)
            for dataset in datasets
            for index, method in enumerate(cold_methods)
        ]
        calibrated_rows = [
            record(dataset, method, 0.75 + index * 0.02, threshold=True)
            for dataset in datasets
            for index, method in enumerate(cold_methods)
        ]

        markdown = build_markdown(
            manifest, random_rows, cold_rows, calibrated_rows
        )

        self.assertIn('普通 Strict 随机边五折', markdown)
        self.assertIn('Compound cold-start 五折', markdown)
        self.assertIn('**+0.010000**', markdown)
        self.assertIn('NoContext 完整五折尚不存在', markdown)

    def test_generator_can_start_as_direct_script(self):
        repository_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                str(repository_root / 'tools' /
                    'build_paper_results_tables.py'),
                '--help',
            ],
            cwd=str(repository_root),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('--manifest', completed.stdout)


if __name__ == '__main__':
    unittest.main()
