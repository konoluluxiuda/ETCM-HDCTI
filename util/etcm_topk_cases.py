import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def read_csv_index(path, key):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return {
            str(row[key]): row
            for row in csv.DictReader(handle)
        }


def write_tsv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter='\t', extrasaction='ignore'
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


def positive_pairs(records):
    return {
        (str(compound_id), str(protein_id))
        for compound_id, protein_id, label in records
        if float(label) > 0
    }


def relation_pairs(path):
    pairs = set()
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(
                    'Invalid relation row %d in %s.' % (line_number, path)
                )
            pairs.add((str(parts[0]), str(parts[1])))
    return pairs


def relation_adjacency(path, reverse=False):
    adjacency = defaultdict(set)
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(
                    'Invalid relation row %d in %s.' % (line_number, path)
                )
            left, right = str(parts[0]), str(parts[1])
            if reverse:
                left, right = right, left
            adjacency[left].add(right)
    return dict(adjacency)


def compound_degrees(records):
    degrees = defaultdict(int)
    for compound_id, _, label in records:
        if float(label) > 0:
            degrees[str(compound_id)] += 1
    return dict(degrees)


def stable_tie_break(seed, *values):
    text = '|'.join([str(int(seed))] + [str(value) for value in values])
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def assign_rank_tertiles(candidates):
    ordered = sorted(
        candidates,
        key=lambda row: (
            int(row['model_train_cp_degree']),
            str(row['compound_id']),
            str(row['tcmip_id']),
        ),
    )
    size = len(ordered)
    if size < 3:
        raise ValueError('At least three eligible compounds are required.')
    for index, row in enumerate(ordered):
        fraction = (index + 0.5) / float(size)
        if fraction < 1.0 / 3.0:
            row['support_stratum'] = 'low'
        elif fraction < 2.0 / 3.0:
            row['support_stratum'] = 'medium'
        else:
            row['support_stratum'] = 'high'
        row['support_percentile'] = fraction
    return ordered


def stratum_allocation(case_count):
    case_count = int(case_count)
    if case_count < 3 or case_count > 5:
        raise ValueError('Case count must be between 3 and 5.')
    allocations = {
        3: {'low': 1, 'medium': 1, 'high': 1},
        4: {'low': 1, 'medium': 1, 'high': 2},
        5: {'low': 2, 'medium': 1, 'high': 2},
    }
    return allocations[case_count]


def select_cases(candidates, case_count=5, seed=2026):
    ranked = assign_rank_tertiles([dict(row) for row in candidates])
    allocation = stratum_allocation(case_count)
    selected = []
    for stratum in ('low', 'medium', 'high'):
        members = [
            row for row in ranked if row['support_stratum'] == stratum
        ]
        members.sort(key=lambda row: (
            -int(bool(row.get('has_identity_identifier'))),
            -int(row.get('unseen_confirmed_targets', 0)),
            -int(row.get('confirmed_evidence_count', 0)),
            -int(row.get('independent_herb_count', 0)),
            -int(row.get('mention_count', 0)),
            stable_tie_break(seed, row['compound_id'], row['tcmip_id']),
        ))
        needed = allocation[stratum]
        if len(members) < needed:
            raise ValueError(
                'Support stratum %s has %d candidates; %d required.' %
                (stratum, len(members), needed)
            )
        selected.extend(members[:needed])
    selected.sort(key=lambda row: (
        ('low', 'medium', 'high').index(row['support_stratum']),
        int(row['model_train_cp_degree']),
        str(row['compound_id']),
    ))
    for order, row in enumerate(selected, start=1):
        row['selection_order'] = order
    return ranked, selected, allocation


def normalize_name(value):
    return re.sub(r'[^a-z0-9]+', '', str(value).casefold())


def split_values(value):
    return [
        item.strip()
        for item in re.split(r'[|;]+', str(value or ''))
        if item.strip()
    ]


def raw_page_index(directory):
    index = defaultdict(list)
    for path in sorted(Path(directory).glob('*.json')):
        index[normalize_name(path.stem)].append(path)
    return dict(index)


def resolve_raw_page(index, values):
    for value in values:
        matches = index.get(normalize_name(value), [])
        if matches:
            return matches[0]
    return None


def pair_context_paths(
        compound_id,
        protein_id,
        herbs_by_compound,
        diseases_by_herb,
        diseases_by_protein):
    paths = []
    protein_diseases = diseases_by_protein.get(str(protein_id), set())
    for herb_id in sorted(herbs_by_compound.get(str(compound_id), set())):
        shared = diseases_by_herb.get(herb_id, set()) & protein_diseases
        paths.extend((herb_id, disease_id) for disease_id in sorted(shared))
    return paths


def sigmoid(value):
    value = float(value)
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 50.0)))
    exp_value = math.exp(max(value, -50.0))
    return exp_value / (1.0 + exp_value)


def rank_candidates(protein_ids, logits):
    if len(protein_ids) != len(logits):
        raise ValueError('Protein IDs and logits have different lengths.')
    if any(not math.isfinite(float(value)) for value in logits):
        raise ValueError('Candidate logits contain NaN or infinity.')
    order = sorted(
        range(len(protein_ids)),
        key=lambda index: (-float(logits[index]), str(protein_ids[index])),
    )
    rows = []
    for rank, index in enumerate(order, start=1):
        rows.append({
            'rank': rank,
            'protein_id': str(protein_ids[index]),
            'logit': float(logits[index]),
            'score': sigmoid(logits[index]),
        })
    return rows


def evidence_classification(pair, confirmed_pairs, potential_pairs):
    if pair in confirmed_pairs:
        return 'A', 'confirmed_unseen'
    if pair in potential_pairs:
        return 'C', 'potential_target'
    return 'E', 'no_etcm_target_evidence'


def group_rows(rows, keys):
    output = defaultdict(list)
    for row in rows:
        output[tuple(str(row[key]) for key in keys)].append(row)
    return dict(output)
