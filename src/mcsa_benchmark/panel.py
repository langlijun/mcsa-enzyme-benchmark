"""Portable, rank-preserving panel replay; no historical labels enter evaluation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from . import __version__, evaluate, list_cases

STAGES = ('mechanism', 'general', 'joint')
STATUSES = {'PASS', 'FAIL', 'ERROR', 'NOT_RUN'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(rows, data_root, expected_samples=5):
    """Reject incomplete cohorts and invalid paths before creating output."""
    if not isinstance(rows, list) or not rows:
        raise ValueError('Manifest must be a nonempty JSON array')
    if expected_samples < 1:
        raise ValueError('expected_samples must be positive')
    root = Path(data_root).resolve()
    keys, groups, checked = set(), defaultdict(list), []
    required = {'sample_key', 'code', 'model', 'sample', 'native_rank', 'input_file', 'sha256'}
    for row in rows:
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError('Manifest row lacks required fields')
        key = row['sample_key']
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', key) or key in keys:
            raise ValueError('Duplicate or unsafe sample_key')
        keys.add(key)
        if row['code'] not in list_cases() or not isinstance(row['model'], str) or not row['model'].strip():
            raise ValueError(f'Invalid registered primary case or model: {key}')
        if any(type(row[name]) is not int for name in ('sample', 'native_rank')):
            raise ValueError(f'Sample and rank must be integers: {key}')
        if row['sample'] < 0:
            raise ValueError(f'Negative sample index: {key}')
        relative = Path(row['input_file'])
        path = (root/relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing or out-of-root input: {key}')
        digest = row['sha256']
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest) or sha256(path) != digest:
            raise ValueError(f'Input SHA256 mismatch: {key}')
        groups[(row['code'], row['model'])].append(row)
        checked.append((dict(row), path))
    for group, members in groups.items():
        if len(members) != expected_samples or len({x['sample'] for x in members}) != expected_samples:
            raise ValueError(f'Incomplete or duplicate sample group: {group}')
        if sorted(x['native_rank'] for x in members) != list(range(1, expected_samples+1)):
            raise ValueError(f'Native ranks must be unique 1..{expected_samples}: {group}')
    return checked


def summarize(samples):
    groups = defaultdict(list)
    for sample in samples:
        m, g, joint = (sample[s+'_status'] for s in STAGES)
        if not {m, g, joint} <= STATUSES:
            raise ValueError('Unknown stage status')
        if (joint == 'PASS') != (m == g == 'PASS'):
            raise ValueError('Joint PASS must use two passing stages from the same sample')
        groups[(sample['code'], sample['model'])].append(sample)
    case_models = []
    for (code, model), members in sorted(groups.items()):
        ranks = sorted(x['native_rank'] for x in members)
        if ranks != list(range(1, len(members)+1)):
            raise ValueError('Cannot summarize missing or duplicate native ranks')
        top = next(x for x in members if x['native_rank'] == 1)
        row = {'code': code, 'model': model, 'samples': len(members), 'top1_sample': top['sample']}
        for stage in STAGES:
            counts = Counter(x[stage+'_status'] for x in members)
            if not counts.keys() <= STATUSES:
                raise ValueError('Unknown stage status')
            row[stage+'_status_counts'] = dict(counts)
            row[stage+'_pass_count'] = counts['PASS']
            row['top1_'+stage+'_status'] = top[stage+'_status']
            row['top1_'+stage+'_pass'] = int(top[stage+'_status'] == 'PASS')
            row['best5_'+stage+'_pass'] = int(counts['PASS'] > 0)
            row['best5_'+stage+'_status'] = ('PASS' if counts['PASS'] else 'ERROR' if counts['ERROR']
                                            else 'NOT_RUN' if counts['NOT_RUN'] else 'FAIL')
        case_models.append(row)
    return {'package_version': __version__, 'samples': len(samples),
            'cases': len({r['code'] for r in samples}),
            'totals': {s:dict(Counter(x[s+'_status'] for x in samples)) for s in STAGES},
            'case_models': case_models}


def scientific_payload(result):
    """Ignore only location/software metadata, never numeric scientific values."""
    keys = ['case_id', 'variant', 'configuration_sha256', 'structure_sha256',
            'input_status', 'input_problem', 'atom_mapping', 'ignored_water_count',
            'input_state_contract', 'atom_profile', 'mechanism', 'general',
            'mechanism_status', 'general_status', 'joint_status']
    return json.loads(json.dumps({k:result[k] for k in keys if k in result}, allow_nan=False))


def baseline_by_key(path, expected_keys):
    records = read_json(path)
    if not isinstance(records, list):
        raise ValueError('Baseline must be an array of sample_key/result objects')
    mapped = {}
    for record in records:
        if record['sample_key'] in mapped:
            raise ValueError('Duplicate baseline sample key')
        mapped[record['sample_key']] = record['result']
    if set(mapped) != set(expected_keys):
        raise ValueError('Baseline cohort differs from the requested manifest')
    return mapped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args(argv)
    try:
        checked = validate_manifest(read_json(args.manifest), args.data_root)
        baseline = baseline_by_key(args.baseline, [x['sample_key'] for x, _ in checked]) if args.baseline else None
        out = args.output.resolve()
        # A new output directory avoids overwriting either old results or inputs.
        out.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    (out/'samples').mkdir()
    samples, full, comparisons = [], [], []
    for i, (row, path) in enumerate(checked, 1):
        result = evaluate(path, row['code'], atom_profile=row.get('atom_profile'))
        key = row['sample_key']
        dump(out/'samples'/f'{key}.json', result)
        full.append({'sample_key':key, 'result':result})
        sample = {k:row[k] for k in ['sample_key','code','model','sample','native_rank','sha256']}
        sample.update({s+'_status':result[s+'_status'] for s in STAGES})
        sample.update(input_status=result['input_status'], criteria_id=result['mechanism'].get('criteria_id'),
                      mechanism_tags=result['mechanism'].get('failure_tags', []),
                      general_tags=result['general'].get('failure_tags', []))
        samples.append(sample)
        if baseline is not None:
            old, new = scientific_payload(baseline[key]), scientific_payload(result)
            changed = sorted(k for k in old.keys() | new.keys()
                             if k not in old or k not in new or
                             json.dumps(old[k], sort_keys=True) != json.dumps(new[k], sort_keys=True))
            comparisons.append({'sample_key':key, 'identical':not changed, 'changed_fields':changed})
        print(f"{i}/{len(checked)} {key}: {result['mechanism_status']} / {result['general_status']}", flush=True)
    summary = summarize(samples)
    summary['manifest_sha256'] = sha256(args.manifest)
    summary['all_input_hashes_checked'] = True
    summary['new_predictions'] = False
    summary['metal_distance_lower_bound_in_general'] = False
    summary['baseline_compared'] = baseline is not None
    summary['baseline_differences'] = sum(not r['identical'] for r in comparisons)
    dump(out/'manifest.json', [{k:r[k] for k in ['sample_key','code','model','sample','native_rank','input_file','sha256','atom_profile'] if k in r} for r, _ in checked])
    dump(out/'samples.json', samples)
    dump(out/'full_results.json', full)
    dump(out/'summary.json', summary)
    dump(out/'case_models.json', summary['case_models'])
    if baseline is not None:
        dump(out/'regression.json', comparisons)
    print(json.dumps({'totals':summary['totals'],'baseline_differences':summary['baseline_differences']}))
    has_error = any(r[s+'_status'] == 'ERROR' for r in samples for s in STAGES)
    return 2 if has_error or summary['baseline_differences'] else 0


if __name__ == '__main__':
    raise SystemExit(main())

