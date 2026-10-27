"""Strict fixed-cohort long-horizon aggregation; no model or torch imports.

All confidence intervals resample whole matched trajectories. They describe
trajectory sampling variability, not independent training seeds. Incomplete or
inconsistent grids fail closed; no results are selected using their outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

REQUIRED_METRICS = ('psnr', 'ssim', 'lpips', 'mae')
LOWER_IS_BETTER = {'lpips', 'mae', 'mse'}
CONTRAST_BASELINES = ('recent_w6', 'current_only', 'full_w1', 'full_w3', 'full_w12')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def retained_indices(target, window, anchor):
    # Deliberately equivalent to decoder.window_decoder.retained_indices, but
    # kept here to allow CPU aggregation without importing the torch decoder.
    if anchor:
        return [1] + list(range(max(2, target - window), target))
    return list(range(max(1, target - window), target))


def validate_trace(trace, *, horizon, recent_states, retain_anchor):
    require(isinstance(recent_states, int) and not isinstance(recent_states, bool)
            and recent_states >= 1, 'Invalid recent window')
    require(isinstance(retain_anchor, bool), 'Invalid anchor setting')
    require(len(trace) == horizon, f'Expected {horizon} trace steps')
    for target, row in enumerate(trace, 2):
        expected = retained_indices(target, recent_states, retain_anchor)
        stored = retained_indices(target + 1, recent_states, retain_anchor)
        require(row['target'] == target and row['retained'] == expected
                and row['stored_after'] == stored,
                f'Invalid state/action/pruning trace at target {target}')


def validated_records(output, protocol):
    """Read records in frozen protocol order, independently of file order."""
    protocol_hash = sha(output / 'protocol.json')
    records, file_hashes, all_ids, expected_files = {}, {}, [], set()
    metric_keys = None
    for cohort_name, cohort in protocol['cohorts'].items():
        horizon = cohort['horizon']
        require(horizon in (64, 128), f'Unexpected horizon: {horizon}')
        require(cohort['cases'], f'Empty cohort: {cohort_name}')
        indices = [row['index'] for row in cohort['cases']]
        require(len(indices) == len(set(indices)), f'Duplicate indices: {cohort_name}')
        ids = [row['id'] for row in cohort['cases']]
        require(len(ids) == len(set(ids)), f'Duplicate IDs: {cohort_name}')
        all_ids.extend(ids)
        records[cohort_name] = {}
        for arm_name, arm in protocol['arms'].items():
            records[cohort_name][arm_name] = []
            for case in cohort['cases']:
                stem = output / 'results' / cohort_name / arm_name / f"case_{case['index']:03d}"
                record_path, artifact_path = stem.with_suffix('.json'), stem.with_suffix('.npz')
                expected_files.update((record_path, artifact_path))
                require(record_path.is_file(), f'Missing result: {record_path}')
                require(artifact_path.is_file(), f'Missing artifact: {artifact_path}')
                record = read(record_path)
                required = {'id': case['id'], 'index': case['index'], 'cohort': cohort_name,
                            'arm': arm_name, 'horizon': horizon,
                            'protocol_sha256': protocol_hash,
                            'input_sha256': case['sha256'],
                            'checkpoint_digest': protocol['checkpoint_digest']}
                for key, expected in required.items():
                    require(record[key] == expected, f'{record_path}: mismatched {key}')
                artifact_hash = sha(artifact_path)
                require(record['artifact_sha256'] == artifact_hash,
                        f'{record_path}: artifact hash mismatch')
                file_hashes[str(record_path.relative_to(output))] = sha(record_path)
                file_hashes[str(artifact_path.relative_to(output))] = artifact_hash
                keys = tuple(sorted(record['per_frame']))
                require(set(REQUIRED_METRICS) <= set(keys), f'{record_path}: missing metrics')
                if metric_keys is None:
                    metric_keys = keys
                require(keys == metric_keys, f'{record_path}: inconsistent metric schema')
                for key in keys:
                    values = np.asarray(record['per_frame'][key], dtype=np.float64)
                    require(values.shape == (horizon,) and np.isfinite(values).all(),
                            f'{record_path}: invalid per-frame {key}')
                    for prefix in (32, 64, 128):
                        if prefix > horizon:
                            continue
                        declared = float(record['metrics'][str(prefix)][key])
                        require(np.isfinite(declared)
                                and abs(float(values[:prefix].mean()) - declared) <= 1e-5,
                                f'{record_path}: incorrect {key} prefix {prefix}')
                for key in ('inference_seconds', 'peak_allocated_GiB'):
                    value = float(record[key])
                    require(np.isfinite(value) and value >= 0,
                            f'{record_path}: invalid {key}')
                validate_trace(record['trace'], horizon=horizon, **arm)
                records[cohort_name][arm_name].append(record)
    require(len(all_ids) == len(set(all_ids)), 'An episode appears in multiple cohorts')
    actual_files = set((output / 'results').glob('*/*/case_*.json'))
    actual_files.update((output / 'results').glob('*/*/case_*.npz'))
    require(actual_files == expected_files,
            f'Unexpected or missing case files: {sorted(map(str, actual_files ^ expected_files))}')
    return records, file_hashes, metric_keys


def bootstrap_weights(rng, n, resamples):
    indices = rng.integers(0, n, size=(resamples, n))
    # Counts produce the same bootstrap as values[indices].mean(1), without a
    # large [resamples, cases, frames] intermediate.
    counts = np.zeros((resamples, n), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(resamples), n), indices.ravel()), 1)
    return counts / n


def estimate(values, weights):
    values = np.asarray(values, dtype=np.float64)
    boot = weights @ values
    lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
    mean = values.mean(axis=0)
    if values.ndim == 1:
        return {'mean': float(mean), 'ci95_low': float(lower), 'ci95_high': float(upper)}
    return {'mean': mean.tolist(), 'ci95_low': lower.tolist(), 'ci95_high': upper.tolist()}


def summarize_cohort(name, records, horizon, metric_keys, weights, *, scope, components=None):
    n = len(next(iter(records.values())))
    prefixes = [h for h in (32, 64, 128) if h <= horizon]
    intervals = [('1-32', 0, 32), ('33-64', 32, 64)]
    if horizon == 128:
        intervals.append(('65-128', 64, 128))
    slices = [(f'prefix_{h}', 0, h) for h in prefixes]
    slices.extend((f'interval_{label}', first, last) for label, first, last in intervals)
    slices.append(('final_16', horizon - 16, horizon))
    result = dict(cases=n, horizon=horizon, scope=scope,
                  trajectory_ids=[r['id'] for r in next(iter(records.values()))],
                  variants={}, paired_contrasts=[])
    if components:
        result['components'] = components
        result['diagnostic_only'] = True
        result['pooling_note'] = ('Combined long-episode dev and held-out test diagnostics; '
                                  'all configurations are frozen before test inference. '
                                  'This does not replace split-specific results or permit tuning.')
    arrays = {}
    for arm, rows in records.items():
        require([r['id'] for r in rows] == result['trajectory_ids'], f'{name}/{arm}: unpaired order')
        arrays[arm] = {key: np.asarray([r['per_frame'][key] for r in rows], dtype=np.float64)
                       for key in metric_keys}
        statistics = {}
        for label, first, last in slices:
            statistics[label] = {key: estimate(values[:, first:last].mean(1), weights)
                                 for key, values in arrays[arm].items()}
        times = np.asarray([r['inference_seconds'] for r in rows], dtype=np.float64)
        peaks = np.asarray([r['peak_allocated_GiB'] for r in rows], dtype=np.float64)
        result['variants'][arm] = dict(
            cases=n, statistics=statistics,
            per_frame={key: estimate(values, weights) for key, values in arrays[arm].items()},
            timing=dict(mean_seconds_per_case=estimate(times, weights),
                        total_generation_seconds=float(times.sum()),
                        total_generation_GPU_hours=float(times.sum() / 3600),
                        mean_seconds_per_predicted_frame=float(times.mean() / horizon),
                        definition='Measured inference only, one GPU per video; not training or end-to-end job wall time.'),
            memory=dict(mean_peak_allocated_GiB=float(peaks.mean()),
                        max_peak_allocated_GiB=float(peaks.max()),
                        definition='Per-case CUDA peak allocated memory; not device-wide reserved or total process memory.'))
    for baseline in CONTRAST_BASELINES:
        require(baseline in arrays, f'Missing contrast baseline: {baseline}')
        for key in metric_keys:
            differences = arrays['full_w6'][key] - arrays[baseline][key]
            per_frame = estimate(differences, weights)
            summaries = {}
            for label, first, last in slices:
                values = differences[:, first:last].mean(1)
                direction = -1 if key in LOWER_IS_BETTER else 1
                favorable = values * direction
                summaries[label] = dict(estimate(values, weights),
                                        full_w6_better=int((favorable > 0).sum()),
                                        tied=int((values == 0).sum()),
                                        full_w6_worse=int((favorable < 0).sum()))
            result['paired_contrasts'].append(dict(
                treatment='full_w6', baseline=baseline, metric=key,
                delta_definition='full_w6 minus baseline', lower_is_better=key in LOWER_IS_BETTER,
                statistics=summaries, per_frame=per_frame))
    return result


def aggregate(output, *, include_pooled=False, verify_input_files=False):
    output = Path(output)
    protocol = read(output / 'protocol.json')
    records, hashes, metric_keys = validated_records(output, protocol)
    if verify_input_files:
        for cohort in protocol['cohorts'].values():
            for case in cohort['cases']:
                require(sha(case['path']) == case['sha256'], f"Prepared input changed: {case['id']}")
    seed = int(protocol.get('bootstrap_seed', 20261024))
    resamples = int(protocol.get('bootstrap_resamples', 5000))
    require(resamples > 0, 'Invalid bootstrap resample count')
    rng = np.random.default_rng(seed)
    summaries = {}
    for name, cohort in protocol['cohorts'].items():
        weights = bootstrap_weights(rng, len(cohort['cases']), resamples)
        summaries[name] = summarize_cohort(name, records[name], cohort['horizon'], metric_keys,
                                            weights, scope=cohort.get('scope', ''))
    pool_names = ('extended_dev128', 'heldout_test128')
    if include_pooled:
        require(all(name in records for name in pool_names), 'Both declared long cohorts required for pooling')
        require(all(protocol['cohorts'][name]['horizon'] == 128 for name in pool_names), 'Pool requires H128')
        combined = {arm: [r for name in pool_names for r in records[name][arm]]
                    for arm in protocol['arms']}
        components = {name: len(protocol['cohorts'][name]['cases']) for name in pool_names}
        weights = bootstrap_weights(rng, sum(components.values()), resamples)
        summaries['combined_long128'] = summarize_cohort(
            'combined_long128', combined, 128, metric_keys, weights,
            scope='Diagnostic combined dev+test long episodes; not used for tuning.', components=components)
    result = dict(state='SUCCEEDED', experiment=protocol.get('experiment'),
                  protocol_sha256=sha(output / 'protocol.json'), protocol=protocol,
                  checkpoint_digest=protocol['checkpoint_digest'], updates=0,
                  evaluated_videos=sum(len(c['cases']) for c in protocol['cohorts'].values()) * len(protocol['arms']),
                  input_files_verified=verify_input_files, artifact_and_record_sha256=hashes,
                  bootstrap=dict(seed=seed, resamples=resamples, unit='whole matched trajectory',
                                 interval='percentile 95%; per-frame intervals are pointwise, not simultaneous',
                                 training_seed_uncertainty=False),
                  interpretation='Fixed-checkpoint inference ablation. Split cohorts are separate; no independent training-seed claim.',
                  cohorts=summaries)
    write(output / 'summary.json', result)
    fields = ['cohort', 'n', 'horizon', 'arm', 'statistic', 'metric', 'mean', 'ci95_low', 'ci95_high',
              'mean_inference_seconds', 'max_peak_allocated_GiB']
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for name, cohort in summaries.items():
        for arm, variant in cohort['variants'].items():
            for label, values in variant['statistics'].items():
                for key, stats in values.items():
                    writer.writerow(dict(cohort=name, n=cohort['cases'], horizon=cohort['horizon'],
                                         arm=arm, statistic=label, metric=key, **stats,
                                         mean_inference_seconds=variant['timing']['mean_seconds_per_case']['mean'],
                                         max_peak_allocated_GiB=variant['memory']['max_peak_allocated_GiB']))
    (output / 'summary.csv').write_text(stream.getvalue())
    fields = ['cohort', 'n', 'treatment', 'baseline', 'metric', 'statistic',
              'mean', 'ci95_low', 'ci95_high', 'full_w6_better', 'tied', 'full_w6_worse']
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for name, cohort in summaries.items():
        for contrast in cohort['paired_contrasts']:
            for label, stats in contrast['statistics'].items():
                writer.writerow(dict(cohort=name, n=cohort['cases'], treatment=contrast['treatment'],
                                     baseline=contrast['baseline'], metric=contrast['metric'],
                                     statistic=label, **stats))
    (output / 'paired_contrasts.csv').write_text(stream.getvalue())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-pooled', action='store_true', help='Also add explicitly diagnostic dev+test H128 pool')
    parser.add_argument('--verify-input-files', action='store_true', help='Rehash prepared inputs in addition to record input hashes')
    args = parser.parse_args()
    result = aggregate(args.output, include_pooled=args.include_pooled,
                       verify_input_files=args.verify_input_files)
    print('LONG_MEMORY_COMPLETE', json.dumps(dict(videos=result['evaluated_videos'],
          cohorts={name: c['cases'] for name, c in result['cohorts'].items()})), flush=True)


if __name__ == '__main__':
    main()
