#!/usr/bin/env python3
import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


EUROPE_PMC_API = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'

COMPOUND_ALIASES = {
    'MTL': ['mannitol'],
    'o-Anisaldehyde': ['2-methoxybenzaldehyde'],
    'm-Guaiacol': ['3-methoxyphenol'],
    'danshensu': ['salvianic acid A'],
    'cyanidin 3-glucoside': ['cyanidin-3-glucoside', 'cyanidin-3-O-glucoside'],
    '1553-41-9': ['eicosapentaenoic acid'],
}

TARGET_ALIASES = {
    'Prostaglandin G/H synthase 1': ['COX-1', 'PTGS1', 'cyclooxygenase-1'],
    'Prostaglandin G/H synthase 2': ['COX-2', 'PTGS2', 'cyclooxygenase-2'],
    'Nuclear receptor coactivator 2': ['NCOA2', 'SRC-2', 'TIF2'],
    'Coagulation factor Xa': ['factor Xa', 'FXa', 'F10'],
    'Sodium channel protein type 5 subunit alpha': ['SCN5A', 'Nav1.5'],
    'DNA topoisomerase II': ['topoisomerase II', 'TOP2A', 'TOP2B'],
    'Gamma-aminobutyric acid receptor subunit alpha-1': ['GABRA1'],
    'Carbonic anhydrase II': ['carbonic anhydrase 2', 'CA2'],
    'Dipeptidyl peptidase IV': ['DPP4', 'CD26'],
    'Leukotriene A-4 hydrolase': ['LTA4H'],
    'Thrombin': ['coagulation factor II', 'F2'],
    'Cytochrome P450-cam': ['P450cam', 'CYP101A1'],
    'Estrogen receptor': ['ESR1', 'ESR2'],
    'Heat shock protein HSP 90': ['HSP90', 'HSP90AA1'],
    'Progesterone receptor': ['PGR'],
    'Alcohol dehydrogenase 1C': ['ADH1C'],
    'Mineralocorticoid receptor': ['NR3C2'],
    'DNA polymerase beta': ['POLB'],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run reproducible exact-pair literature searches in Europe PMC.'
    )
    parser.add_argument('--sample', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--page-size', type=int, default=20)
    parser.add_argument('--delay', type=float, default=0.1)
    return parser.parse_args()


def quoted_terms(terms):
    return ' OR '.join('TITLE_ABS:"%s"' % term.replace('"', '') for term in terms)


def build_query(compound_name, target_name):
    compounds = [compound_name] + COMPOUND_ALIASES.get(compound_name, [])
    targets = [target_name] + TARGET_ALIASES.get(target_name, [])
    return '(%s) AND (%s)' % (quoted_terms(compounds), quoted_terms(targets))


def fetch_json(url, attempts=3):
    request = urllib.request.Request(
        url,
        headers={'User-Agent': 'HDCTI-research-audit/1.0', 'Accept': 'application/json'},
    )
    error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as exc:
            error = exc
        if attempt + 1 < attempts:
            time.sleep(1.0 + attempt)
    raise RuntimeError('Failed to fetch %s: %s' % (url, error))


def main():
    args = parse_args()
    sample_path = Path(args.sample).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_path, encoding='utf-8') as handle:
        sample = list(csv.DictReader(handle, delimiter='\t'))

    searches = []
    summary_rows = []
    for index, row in enumerate(sample, start=1):
        query = build_query(row['molecule_name'], row['target_name'])
        params = urllib.parse.urlencode({
            'query': query,
            'format': 'json',
            'pageSize': args.page_size,
            'resultType': 'core',
        })
        url = '%s?%s' % (EUROPE_PMC_API, params)
        payload = fetch_json(url)
        results = payload.get('resultList', {}).get('result', [])
        compact_results = []
        for result in results:
            compact_results.append({
                'id': result.get('pmid') or result.get('pmcid') or result.get('id'),
                'pmid': result.get('pmid'),
                'pmcid': result.get('pmcid'),
                'doi': result.get('doi'),
                'title': result.get('title'),
                'author_string': result.get('authorString'),
                'journal': result.get('journalTitle'),
                'year': result.get('pubYear'),
                'abstract': result.get('abstractText'),
            })
        hit_count = int(payload.get('hitCount') or 0)
        searches.append({
            'sample_index': int(row['sample_index']),
            'compound_name': row['molecule_name'],
            'target_name': row['target_name'],
            'query': query,
            'url': url,
            'hit_count': hit_count,
            'results': compact_results,
        })
        summary_rows.append({
            'sample_index': row['sample_index'],
            'compound_name': row['molecule_name'],
            'target_name': row['target_name'],
            'hit_count': hit_count,
            'query': query,
            'search_url': url,
        })
        print('[%d/%d] hits=%d %s -> %s' % (
            index, len(sample), hit_count, row['molecule_name'], row['target_name']
        ), flush=True)
        time.sleep(max(0.0, args.delay))

    (output_dir / 'europe_pmc_search.json').write_text(
        json.dumps(searches, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    with open(output_dir / 'europe_pmc_search.tsv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == '__main__':
    main()
