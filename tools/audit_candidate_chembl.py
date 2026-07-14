#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


CHEMBL_API = 'https://www.ebi.ac.uk/chembl/api/data'
TARGET_ORGANISM_OVERRIDES = {
    'Cytochrome P450-cam': 'Pseudomonas putida',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit sampled compound-target candidates against ChEMBL activities.'
    )
    parser.add_argument('--sample', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--delay', type=float, default=0.1)
    return parser.parse_args()


def fetch_json(url, allow_not_found=False, attempts=3):
    request = urllib.request.Request(
        url,
        headers={'User-Agent': 'HDCTI-research-audit/1.0', 'Accept': 'application/json'},
    )
    error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            if allow_not_found and exc.code == 404:
                return None
            error = exc
        except Exception as exc:
            error = exc
        if attempt + 1 < attempts:
            time.sleep(1.0 + attempt)
    raise RuntimeError('Failed to fetch %s: %s' % (url, error))


def normalized(value):
    return re.sub(r'[^a-z0-9]+', '', str(value).lower())


def expected_target_organism(requested_name):
    return TARGET_ORGANISM_OVERRIDES.get(requested_name, 'Homo sapiens')


def target_match_score(target, requested_name):
    requested = normalized(requested_name)
    preferred_name = normalized(target.get('pref_name') or '')
    component_names = []
    for component in target.get('target_components') or []:
        component_names.append(component.get('component_description') or '')
        component_names.extend(
            synonym.get('component_synonym') or ''
            for synonym in component.get('target_component_synonyms') or []
        )
    component_names = [normalized(name) for name in component_names if name]

    if requested == preferred_name:
        match_type = 'exact_pref_name'
        score = 3000
    elif requested in component_names:
        match_type = 'exact_component_name'
        score = 2000
    elif preferred_name and (requested in preferred_name or preferred_name in requested):
        match_type = 'partial_pref_name'
        score = 1200
    elif any(requested in name or name in requested for name in component_names):
        match_type = 'partial_component_name'
        score = 1000
    else:
        match_type = 'search_only'
        score = 0

    target_type = str(target.get('target_type') or '')
    if target_type == 'SINGLE PROTEIN':
        score += 300
    if target.get('organism') == expected_target_organism(requested_name):
        score += 1000
    score += float(target.get('score') or 0)
    return score, match_type


def resolve_target(target_name):
    query = urllib.parse.urlencode({'q': target_name, 'limit': 100})
    url = '%s/target/search.json?%s' % (CHEMBL_API, query)
    payload = fetch_json(url)
    targets = payload.get('targets') or []
    if not targets:
        return None, 'unmatched', url
    ranked = sorted(
        targets,
        key=lambda target: target_match_score(target, target_name)[0],
        reverse=True,
    )
    selected = ranked[0]
    _, match_type = target_match_score(selected, target_name)
    target_type = str(selected.get('target_type') or 'UNKNOWN').lower().replace(' ', '_')
    confidence = '%s_%s' % (match_type, target_type)
    return selected, confidence, url


def resolve_molecule(inchi_key):
    url = '%s/molecule/%s.json' % (
        CHEMBL_API, urllib.parse.quote(str(inchi_key), safe='-')
    )
    return fetch_json(url, allow_not_found=True), url


def classify_activity(activity):
    if activity.get('assay_type') not in ('B', 'F'):
        return 'excluded_assay_type'
    text = ' '.join(
        str(activity.get(key) or '')
        for key in ('activity_comment', 'standard_text_value', 'text_value')
    ).lower()
    if 'not active' in text or 'inactive' in text:
        return 'direct_negative'

    relation = str(activity.get('standard_relation') or activity.get('relation') or '')
    try:
        pchembl = float(activity['pchembl_value'])
    except (TypeError, ValueError, KeyError):
        pchembl = None
    if pchembl is not None and pchembl >= 5.0 and relation not in ('>', '>='):
        return 'credible_positive'

    standard_type = str(activity.get('standard_type') or '').lower()
    try:
        standard_value = float(activity['standard_value'])
    except (TypeError, ValueError, KeyError):
        standard_value = None
    standard_units = str(activity.get('standard_units') or '').lower()
    if (
            standard_type in ('ic50', 'ec50', 'ki', 'kd', 'potency')
            and standard_value is not None
            and standard_units == 'nm'
            and standard_value <= 10000
            and relation not in ('>', '>=')
    ):
        return 'credible_positive'
    if (
            standard_type in ('inhibition', 'activity')
            and standard_value is not None
            and standard_value >= 50
            and relation not in ('<', '<=')
    ):
        return 'credible_positive'
    return 'direct_ambiguous'


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return 0.0, 0.0
    proportion = successes / float(total)
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def main():
    args = parse_args()
    sample_path = Path(args.sample).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(sample_path, encoding='utf-8') as handle:
        sample = list(csv.DictReader(handle, delimiter='\t'))

    audit_path = output_dir / 'chembl_audit.tsv'
    existing_molecules = {}
    if audit_path.exists():
        with open(audit_path, encoding='utf-8') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                if row.get('inchi_key') and row.get('chembl_molecule_id'):
                    existing_molecules[row['inchi_key']] = row['chembl_molecule_id']

    output_rows = []
    all_positive_documents = set()
    target_cache = {}
    for index, row in enumerate(sample, start=1):
        molecule_url = '%s/molecule/%s.json' % (
            CHEMBL_API, urllib.parse.quote(str(row['inchi_key']), safe='-')
        )
        cached_molecule_id = existing_molecules.get(row['inchi_key'])
        if cached_molecule_id:
            molecule = {'molecule_chembl_id': cached_molecule_id}
        else:
            molecule, molecule_url = resolve_molecule(row['inchi_key'])
        target_name = row['target_name']
        if target_name not in target_cache:
            target_cache[target_name] = resolve_target(target_name)
        target, target_confidence, target_search_url = target_cache[target_name]
        molecule_id = molecule.get('molecule_chembl_id') if molecule else None
        target_id = target.get('target_chembl_id') if target else None
        target_type = str(target.get('target_type') or '') if target else ''
        target_mapping_usable = target_type != 'PROTEIN-PROTEIN INTERACTION'
        activities = []
        activity_url = ''
        if molecule_id and target_id and target_mapping_usable:
            query = urllib.parse.urlencode({
                'molecule_chembl_id': molecule_id,
                'target_chembl_id': target_id,
                'limit': 1000,
            })
            activity_url = '%s/activity.json?%s' % (CHEMBL_API, query)
            activities = fetch_json(activity_url).get('activities') or []

        classifications = [classify_activity(activity) for activity in activities]
        positives = [
            activity for activity, label in zip(activities, classifications)
            if label == 'credible_positive'
        ]
        negatives = [
            activity for activity, label in zip(activities, classifications)
            if label == 'direct_negative'
        ]
        ambiguous = [
            activity for activity, label in zip(activities, classifications)
            if label == 'direct_ambiguous'
        ]
        if positives:
            evidence_class = 'credible_positive'
        elif negatives:
            evidence_class = 'direct_negative'
        elif ambiguous:
            evidence_class = 'direct_ambiguous'
        elif target_id and not target_mapping_usable:
            evidence_class = 'target_mapping_not_specific'
        elif not molecule_id:
            evidence_class = 'compound_not_in_chembl'
        elif not target_id:
            evidence_class = 'target_not_in_chembl'
        else:
            evidence_class = 'no_direct_chembl_record'

        pchembl_values = []
        for activity in positives:
            try:
                pchembl_values.append(float(activity['pchembl_value']))
            except (TypeError, ValueError, KeyError):
                pass
            if activity.get('document_chembl_id'):
                all_positive_documents.add(activity['document_chembl_id'])

        output_rows.append({
            **row,
            'chembl_molecule_id': molecule_id or '',
            'chembl_target_id': target_id or '',
            'chembl_target_name': target.get('pref_name') if target else '',
            'chembl_target_organism': target.get('organism') if target else '',
            'chembl_target_type': target.get('target_type') if target else '',
            'target_mapping_confidence': target_confidence,
            'target_mapping_usable': target_mapping_usable,
            'activity_records': len(activities),
            'credible_positive_records': len(positives),
            'direct_negative_records': len(negatives),
            'ambiguous_records': len(ambiguous),
            'best_pchembl': max(pchembl_values) if pchembl_values else '',
            'positive_document_ids': ';'.join(sorted(
                {
                    activity.get('document_chembl_id') or '' for activity in positives
                } - {''}
            )),
            'chembl_evidence_class': evidence_class,
            'chembl_molecule_url': molecule_url,
            'chembl_target_search_url': target_search_url,
            'chembl_activity_url': activity_url,
        })
        print(
            '[%d/%d] %s -> %s: %s (%d activities)' % (
                index, len(sample), row['molecule_name'], row['target_name'],
                evidence_class, len(activities),
            ),
            flush=True,
        )
        time.sleep(max(0.0, args.delay))

    documents = {}
    for document_id in sorted(all_positive_documents):
        url = '%s/document/%s.json' % (CHEMBL_API, document_id)
        document = fetch_json(url, allow_not_found=True) or {}
        documents[document_id] = {
            'title': document.get('title'),
            'doi': document.get('doi'),
            'pubmed_id': document.get('pubmed_id'),
            'journal': document.get('journal'),
            'year': document.get('year'),
            'url': url,
        }
        time.sleep(max(0.0, args.delay))

    with open(audit_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(output_rows[0].keys()), delimiter='\t'
        )
        writer.writeheader()
        writer.writerows(output_rows)

    counts = {}
    for row in output_rows:
        label = row['chembl_evidence_class']
        counts[label] = counts.get(label, 0) + 1
    positive_count = counts.get('credible_positive', 0)
    lower, upper = wilson_interval(positive_count, len(output_rows))
    summary = {
        'sample_size': len(output_rows),
        'classification_counts': counts,
        'credible_positive_proportion': positive_count / float(len(output_rows)),
        'credible_positive_wilson_95_ci': [lower, upper],
        'substantive_positive_threshold': 0.20,
        'chembl_screen_reaches_substantive_threshold': (
            positive_count / float(len(output_rows)) >= 0.20
        ),
        'positive_definition': (
            'ChEMBL binding/functional assay with pChEMBL >= 5, potency <= 10 uM, '
            'or inhibition/activity >= 50%; docking and predictions excluded'
        ),
        'database_scope_limit': (
            'No ChEMBL record means not found in ChEMBL, not proof of no interaction.'
        ),
        'documents': documents,
        'audit_path': str(audit_path),
    }
    (output_dir / 'chembl_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
