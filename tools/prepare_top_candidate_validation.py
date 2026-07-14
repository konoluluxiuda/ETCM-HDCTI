#!/usr/bin/env python3
import argparse
import csv
import hashlib
import html
import json
import math
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare a deterministic sample of top unlabeled TCMSP candidates.'
    )
    parser.add_argument('--input', required=True, help='top_candidates.tsv from checkpoint ranking.')
    parser.add_argument('--sample-size', type=int, default=30)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--delay', type=float, default=0.15)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sortable_id(value):
    value = str(value)
    return (0, int(value)) if value.isdigit() else (1, value)


def systematic_sample(rows, sample_size):
    first_unlabeled = {}
    for row in rows:
        if row['label_status'] != 'unlabeled':
            continue
        compound_id = str(row['compound_id'])
        current = first_unlabeled.get(compound_id)
        if current is None or int(row['rank']) < int(current['rank']):
            first_unlabeled[compound_id] = row
    population = [
        first_unlabeled[key] for key in sorted(first_unlabeled, key=sortable_id)
    ]
    if sample_size <= 0 or sample_size > len(population):
        raise ValueError('sample-size must be between 1 and %d.' % len(population))
    indices = [int(math.floor(index * len(population) / sample_size)) for index in range(sample_size)]
    return population, indices, [population[index] for index in indices]


def fetch(url, attempts=3):
    request = urllib.request.Request(
        url,
        headers={'User-Agent': 'HDCTI-research-audit/1.0'},
    )
    error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode('utf-8', errors='replace')
        except Exception as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(1.0 + attempt)
    raise RuntimeError('Failed to fetch %s: %s' % (url, error))


def clean_html(value):
    value = re.sub(r'<br\s*/?>', '; ', value, flags=re.IGNORECASE)
    value = re.sub(r'<[^>]+>', '', value)
    value = html.unescape(value)
    return re.sub(r'\s+', ' ', value).strip(' ;')


def table_value(page, label):
    pattern = (
        r'<th[^>]*>\s*%s\s*</th>\s*<td[^>]*>(.*?)</td>' %
        re.escape(label)
    )
    match = re.search(pattern, page, flags=re.IGNORECASE | re.DOTALL)
    return clean_html(match.group(1)) if match else ''


def entity_metadata(compound_id, protein_id):
    molecule_url = 'https://tcmsp-e.com/molecule.php?qn=%s' % compound_id
    target_url = 'https://tcmsp-e.com/target.php?qt=%s' % protein_id
    molecule_page = fetch(molecule_url)
    target_page = fetch(target_url)
    return {
        'molecule_tcmsp_id': table_value(molecule_page, 'Molecule ID'),
        'molecule_name': table_value(molecule_page, 'Molecule name'),
        'inchi_key': table_value(molecule_page, 'InChIKey'),
        'pubchem_cid': table_value(molecule_page, 'Pubchem Cid'),
        'cas': table_value(molecule_page, 'CAS'),
        'target_tcmsp_id': table_value(target_page, 'Target ID'),
        'target_name': table_value(target_page, 'Target name'),
        'drugbank_id': table_value(target_page, 'Drugbank ID'),
        'molecule_url': molecule_url,
        'target_url': target_url,
    }


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    population, indices, sample = systematic_sample(rows, args.sample_size)

    output_rows = []
    for sample_index, row in enumerate(sample, start=1):
        metadata = entity_metadata(row['compound_id'], row['protein_id'])
        output_rows.append({
            'sample_index': sample_index,
            'population_index': indices[sample_index - 1],
            'compound_id': row['compound_id'],
            'protein_id': row['protein_id'],
            'candidate_rank': row['rank'],
            'model_score': row['score'],
            **metadata,
            'evidence_level': '',
            'evidence_summary': '',
            'evidence_urls': '',
            'review_status': 'pending',
        })
        print(
            '[%d/%d] %s (%s) -> %s (%s)' % (
                sample_index,
                len(sample),
                metadata['molecule_name'],
                metadata['molecule_tcmsp_id'],
                metadata['target_name'],
                metadata['target_tcmsp_id'],
            )
        )
        time.sleep(max(0.0, args.delay))

    fieldnames = list(output_rows[0].keys())
    sample_path = output_dir / 'validation_sample.tsv'
    with open(sample_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        writer.writerows(output_rows)

    manifest = {
        'created_at': datetime.now().astimezone().isoformat(),
        'input_path': str(input_path),
        'input_sha256': sha256_file(input_path),
        'population_definition': 'highest-ranked unlabeled pair per test-positive compound',
        'population_size': len(population),
        'sampling_method': 'systematic_floor_i_times_N_over_n_after_numeric_compound_sort',
        'sample_size': len(sample),
        'population_indices_zero_based': indices,
        'mapping_source': 'TCMSP official molecule and target pages; mapping only, not evidence',
        'sample_path': str(sample_path),
        'sample_sha256': sha256_file(sample_path),
    }
    (output_dir / 'sample_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print('Wrote %s' % sample_path)


if __name__ == '__main__':
    main()
