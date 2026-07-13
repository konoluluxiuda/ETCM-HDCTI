import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from .io import FileIO, NEGATIVE_FILE_CANDIDATES, resolve_optional_dataset_file, sample_negative_records


STRICT_MANIFEST_VERSION = 3
SIDE_RELATION_FILE_CANDIDATES = {
    'H_C': ('H_C.txt', 'herb-compound.txt', 'HI.txt'),
    'C_P': ('C_P.txt', 'compound-protein.txt', 'IT.txt'),
    'P_D': ('P_D.txt', 'target-disease.txt', 'TD.txt'),
}


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _side_relation_sources(dataset_dir):
    sources = {}
    for relation_name, candidates in SIDE_RELATION_FILE_CANDIDATES.items():
        path_value = resolve_optional_dataset_file(str(dataset_dir), candidates)
        path = Path(path_value).resolve() if path_value else None
        sources[relation_name] = {
            'path': str(path) if path else None,
            'sha256': _sha256(path) if path else None,
        }
    return sources


def _deduplicate(records, positive):
    output = []
    seen = set()
    for left_id, right_id, rating in records:
        is_positive = float(rating) > 0
        if is_positive != positive:
            continue
        pair = (str(left_id), str(right_id))
        if pair in seen:
            continue
        seen.add(pair)
        output.append([pair[0], pair[1], 1.0 if positive else 0.0])
    return output


def _sample_negative_file(conf, path, count, positives, rng):
    reservoir = []
    selected = set()
    eligible_count = 0
    for left_id, right_id, _ in FileIO.iterDataSet(conf, path, default_rating=0.0):
        pair = (str(left_id), str(right_id))
        if pair in positives or pair in selected:
            continue
        eligible_count += 1
        record = [pair[0], pair[1], 0.0]
        if len(reservoir) < count:
            reservoir.append(record)
            selected.add(pair)
            continue
        replacement = rng.randrange(eligible_count)
        if replacement < count:
            previous = reservoir[replacement]
            selected.remove((previous[0], previous[1]))
            reservoir[replacement] = record
            selected.add(pair)
    if len(reservoir) < count:
        raise ValueError(
            'Negative file %s yielded only %d eligible pairs; %d are required.' %
            (path, len(reservoir), count)
        )
    return reservoir, eligible_count


def _write_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    temp_path.write_text(content, encoding='utf-8')
    os.replace(str(temp_path), str(path))


class DataSplit(object):

    def __init__(self):
        pass

    @staticmethod
    def crossValidation(data, k, output=True, path='./dataset/TCMsuite', order=1):
        if k <= 1 or k > 10:
            k = 3
        for i in range(k):
            trainingSet = []
            testSet = []
            for ind, line in enumerate(data):
                if ind % k == i:
                    testSet.append(line[:])
                else:
                    trainingSet.append(line[:])

            if output:
                os.makedirs(path, exist_ok=True)
                save_path = os.path.join(path, 'test_fold_%d.txt' % i)
                with open(save_path, 'w', encoding='utf-8') as f:
                    for item in testSet:
                        if isinstance(item, list):
                            f.write('\t'.join(map(str, item)) + '\n')
                        else:
                            f.write(str(item) + '\n')

            yield trainingSet, testSet

    @staticmethod
    def prepareStrictFolds(conf, datapath, k):
        if k <= 1 or k > 10:
            raise ValueError('Strict cross-validation requires k between 2 and 10.')

        seed = int(conf['random.seed']) if conf.contains('random.seed') else 2026
        dataset_path = Path(datapath).resolve()
        dataset_dir = dataset_path.parent
        if conf.contains('split.dir'):
            split_dir = Path(conf['split.dir']).resolve()
        else:
            split_dir = dataset_dir / 'splits' / ('strict_seed_%d_k%d' % (seed, k))
        reuse = _as_bool(conf['split.reuse'], True) if conf.contains('split.reuse') else True
        manifest_path = split_dir / 'manifest.json'
        assignments_path = split_dir / 'fold_assignments.tsv'

        negative_path_value = resolve_optional_dataset_file(str(dataset_dir), NEGATIVE_FILE_CANDIDATES)
        negative_path = Path(negative_path_value).resolve() if negative_path_value else None
        expected_sources = {
            'datapath': str(dataset_path),
            'datapath_sha256': _sha256(dataset_path),
            'negative_path': str(negative_path) if negative_path else None,
            'negative_sha256': _sha256(negative_path) if negative_path else None,
            'side_relations': _side_relation_sources(dataset_dir),
        }

        if reuse and manifest_path.exists() and assignments_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            DataSplit._validateStrictManifest(manifest, expected_sources, seed, k, assignments_path)
            folds = DataSplit._loadStrictAssignments(assignments_path, k)
            print('Reusing strict split manifest: %s' % manifest_path)
            return folds, manifest

        positive_records = _deduplicate(FileIO.readDataSet(conf, str(dataset_path)), positive=True)
        if not positive_records:
            raise ValueError('Strict protocol found no positive records in %s.' % dataset_path)
        positive_pairs = {(row[0], row[1]) for row in positive_records}
        rng = random.Random(seed)

        if negative_path:
            negative_records, candidate_count = _sample_negative_file(
                conf, str(negative_path), len(positive_records), positive_pairs, rng
            )
            negative_mode = 'sampled_file'
        else:
            negative_records = sample_negative_records(
                positive_records, len(positive_records), rng=rng, deterministic=True
            )
            candidate_count = (
                len({row[0] for row in positive_records})
                * len({row[1] for row in positive_records})
                - len(positive_pairs)
            )
            negative_mode = 'generated_cartesian'

        rng.shuffle(positive_records)
        rng.shuffle(negative_records)
        fold_records = [[] for _ in range(k)]
        for index, record in enumerate(positive_records):
            fold_records[index % k].append(record)
        for index, record in enumerate(negative_records):
            fold_records[index % k].append(record)
        for fold_index, records in enumerate(fold_records):
            random.Random(seed + fold_index + 1).shuffle(records)

        split_dir.mkdir(parents=True, exist_ok=True)
        assignment_lines = ['left_id\tright_id\tlabel\tfold\n']
        fold_stats = []
        for fold_index, records in enumerate(fold_records):
            positive_count = sum(float(row[2]) > 0 for row in records)
            negative_count = len(records) - positive_count
            fold_stats.append({
                'fold': fold_index,
                'test_records': len(records),
                'test_positives': positive_count,
                'test_negatives': negative_count,
            })
            test_lines = []
            for left_id, right_id, rating in records:
                label = 1 if float(rating) > 0 else 0
                assignment_lines.append('%s\t%s\t%d\t%d\n' % (left_id, right_id, label, fold_index))
                test_lines.append('%s\t%s\t%d\n' % (left_id, right_id, label))
            _write_atomic(split_dir / ('test_fold_%d.txt' % fold_index), ''.join(test_lines))
        _write_atomic(assignments_path, ''.join(assignment_lines))

        manifest = {
            'version': STRICT_MANIFEST_VERSION,
            'protocol': 'strict',
            'split_algorithm': 'class_stratified_round_robin_v2',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'seed': seed,
            'folds': k,
            'sources': expected_sources,
            'negative_mode': negative_mode,
            'negative_candidate_count': candidate_count,
            'positive_count': len(positive_records),
            'negative_count': len(negative_records),
            'assignments_path': str(assignments_path),
            'assignments_sha256': _sha256(assignments_path),
            'fold_stats': fold_stats,
            'strict_guarantees': {
                'fixed_negative_sample': True,
                'fixed_stratified_folds': True,
                'pair_disjoint_train_test': True,
                'training_graph_must_use_fold_training_positives': True,
                'fixed_hd_side_information': False,
            },
        }
        _write_atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        folds = DataSplit._loadStrictAssignments(assignments_path, k)
        print('Created strict split manifest: %s' % manifest_path)
        return folds, manifest

    @staticmethod
    def _validateStrictManifest(manifest, expected_sources, seed, k, assignments_path):
        errors = []
        if manifest.get('version') != STRICT_MANIFEST_VERSION:
            errors.append('manifest version')
        if manifest.get('protocol') != 'strict':
            errors.append('protocol')
        if manifest.get('seed') != seed:
            errors.append('seed')
        if manifest.get('folds') != k:
            errors.append('folds')
        if manifest.get('sources') != expected_sources:
            errors.append('source files or hashes')
        if manifest.get('assignments_path') != str(assignments_path):
            errors.append('assignments path')
        if manifest.get('assignments_sha256') != _sha256(assignments_path):
            errors.append('assignments hash')
        if errors:
            raise ValueError(
                'Existing strict manifest does not match %s. Use a new split.dir or set split.reuse=False.' %
                ', '.join(errors)
            )

    @staticmethod
    def _loadStrictAssignments(path, k):
        fold_records = [[] for _ in range(k)]
        all_records = []
        with open(path, encoding='utf-8') as handle:
            header = next(handle, '').strip().split('\t')
            if header != ['left_id', 'right_id', 'label', 'fold']:
                raise ValueError('Invalid strict assignment header in %s.' % path)
            for line_number, line in enumerate(handle, start=2):
                parts = line.rstrip('\n').split('\t')
                if len(parts) != 4:
                    raise ValueError('Invalid strict assignment row %d in %s.' % (line_number, path))
                left_id, right_id, label, fold_value = parts
                fold_index = int(fold_value)
                if fold_index < 0 or fold_index >= k:
                    raise ValueError('Invalid fold %d in %s.' % (fold_index, path))
                record = [left_id, right_id, float(label)]
                fold_records[fold_index].append(record)
                all_records.append((fold_index, record))

        folds = []
        for fold_index in range(k):
            test = [record[:] for record in fold_records[fold_index]]
            train = [record[:] for assigned_fold, record in all_records if assigned_fold != fold_index]
            train_pairs = {(row[0], row[1]) for row in train}
            test_pairs = {(row[0], row[1]) for row in test}
            if train_pairs & test_pairs:
                raise ValueError('Strict split contains train/test pair overlap in fold %d.' % fold_index)
            folds.append((train, test))
        return folds
