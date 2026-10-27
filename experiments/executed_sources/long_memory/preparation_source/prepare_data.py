"""Prepare a length-only subset of the existing RT-1 dev cohort, without GPUs.

Reads original TFRecords and writes new files only inside --output. The historical
34-frame evaluation inputs and original model/experiment files are never changed.
"""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

D = Path('/opt/roboimagine-workspace')
DEFAULT_BASE = D / 'experiments/r2_separate_controls_20261012/rt1_hybrid500'
DEFAULT_READER = D / 'experiments/rt1_mjev_20261011/sft_mjev500_8gpu_v1/code/src/futureworlds/data_readers/rt1.py'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def select(selection, horizon):
    rows = selection['dev']
    assert len(rows) == 128, 'Expected the original 128-episode dev cohort'
    ids = [r['id'] for r in rows]
    assert len(set(ids)) == 128, 'Duplicate evaluation episode ID'
    train = selection['train']
    train_ids = {r['id'] for r in train}
    assert not train_ids.intersection(ids), 'Training/evaluation episode overlap'
    eligible, excluded = [], []
    for index, row in enumerate(rows):
        original = row['original']
        assert row['id'] == original['id']
        assert original['split'] == 'dev', 'Only original dev episodes are allowed'
        assert row['start'] == 0 and row['clip_length'] == 34
        assert original['text'] == row['text'] and row['text'].strip()
        frames = int(original['frames'])
        assert frames >= 34
        if frames >= horizon + 2:
            eligible.append((index, row))
        else:
            excluded.append(dict(original_index=index, id=row['id'], original_frames=frames,
                                 required_frames=horizon + 2,
                                 reason='Original episode shorter than two observations plus requested horizon'))
    return eligible, excluded, dict(train_rows=len(train), unique_train_episodes=len(train_ids),
                                   original_dev_episodes=128, overlap_episodes=0)


def explicit_cohort(cohort_file, selection, horizon):
    """Admit a predeclared held-out cohort; never select on model outputs here."""
    cohort = read(cohort_file)
    assert cohort['version'] == 1 and cohort['dataset'] == 'rt1'
    assert cohort['horizon'] == horizon and cohort['start'] == 0
    assert cohort['selection_uses_model_outcomes'] is False and cohort['selection_rule'].strip()
    source_file = Path(cohort['source_manifest'])
    assert sha(source_file) == cohort['source_manifest_sha256']
    source_rows = [json.loads(s) for s in source_file.read_text().splitlines() if s.strip()]
    lookup = {r['id']: r for r in source_rows}
    source_splits = {r['split'] for r in source_rows}
    assert len(lookup) == len(source_rows) and len(source_splits) == 1
    logical_split = next(iter(source_splits))
    assert logical_split in ('dev', 'test'), 'Only held-out logical dev/test splits are admitted'
    if 'split' in cohort:
        assert cohort['split'] == logical_split
    old_rows = selection['dev']
    assert len(old_rows) == 128 and len({r['id'] for r in old_rows}) == 128
    old_lookup = {r['id']: (i, r) for i, r in enumerate(old_rows)}
    train_ids = {r['id'] for r in selection['train']}
    assert not train_ids.intersection(old_lookup)
    eligible = []
    ids = []
    for expected_index, item in enumerate(cohort['cases']):
        assert item['index'] == expected_index, 'Explicit cohort indices must be contiguous and ordered'
        original = item['original']
        identity = original['id']
        assert identity in lookup and original == lookup[identity], 'Case differs from hashed held-out manifest'
        assert identity not in train_ids, 'Long evaluation episode occurs in RL training cohort'
        assert int(original['frames']) >= horizon + 2 and original['text'].strip()
        old_index, historical = old_lookup.get(identity, (None, None))
        if item.get('historical_index') is not None:
            assert item['historical_index'] == old_index
        row = dict(id=identity, original=original, text=original['text'], start=0,
                   historical=historical, historical_index=old_index)
        eligible.append((expected_index, row))
        ids.append(identity)
    assert len(set(ids)) == len(ids) and ids, 'Explicit cohort is empty or contains duplicates'
    length_eligible = {r['id'] for r in source_rows if int(r['frames']) >= horizon + 2 and r['text'].strip()}
    assert set(ids) == length_eligible, 'Explicit long cohort must include every length-eligible instructed episode'
    excluded = [dict(id=r['id'], original_frames=int(r['frames']), required_frames=horizon + 2,
                     reason='Original episode too short' if int(r['frames']) < horizon + 2 else 'Empty instruction')
                for r in source_rows if r['id'] not in length_eligible]
    provenance = dict(explicit_cohort=str(Path(cohort_file).resolve()), explicit_cohort_sha256=sha(cohort_file),
                      heldout_manifest=str(source_file), heldout_manifest_sha256=sha(source_file),
                      logical_split=logical_split)
    split_check = dict(train_rows=len(selection['train']), unique_train_episodes=len(train_ids),
                       original_dev_episodes=128, complete_heldout_manifest_episodes=len(source_rows),
                       logical_split=logical_split,
                       new_cohort_episodes=len(ids), overlap_episodes=0)
    return eligible, excluded, split_check, provenance, cohort['selection_rule']


def prepare(output, source_base, reader_file, horizon=128, cohort_file=None):
    output = Path(output).resolve()
    source_base = Path(source_base).resolve()
    reader_file = Path(reader_file).resolve()
    assert horizon in (64, 128), 'This experiment admits only explicit H64/H128 datasets'
    assert output != source_base and source_base not in output.parents, 'Keep new cache outside the source experiment'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'prepare.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    selection_file = source_base / 'selection.json'
    prepared_file = source_base / 'prepared.json'
    selection = read(selection_file)
    prepared = read(prepared_file)
    assert prepared['selection_sha256'] == sha(selection_file), 'Historical selection hash mismatch'
    if cohort_file is None:
        eligible, excluded, split_check = select(selection, horizon)
        eligible = [(i, dict(r, historical=r, historical_index=i)) for i, r in eligible]
        cohort_source = {}
        selection_rule = f'Original 128 dev episodes with original.frames >= {horizon + 2}; preserve original order'
        scope = 'A length-filtered subset of the pre-existing 128-case dev cohort'
    else:
        eligible, excluded, split_check, cohort_source, selection_rule = explicit_cohort(cohort_file, selection, horizon)
        scope = f"A new predeclared held-out {cohort_source['logical_split']} cohort; not the historical 128-case evaluation cohort"
    source = dict(selection=str(selection_file), selection_sha256=sha(selection_file),
                  prepared=str(prepared_file), prepared_sha256=sha(prepared_file),
                  reader=str(reader_file), reader_sha256=sha(reader_file), **cohort_source)
    plan = dict(version=1, dataset='rt1', horizon=horizon, initial_observations=2,
                clip_length=horizon + 2, start=0, source=source,
                selection_rule=selection_rule, data_scope=scope,
                selection_uses_model_outcomes=False,
                eligible_indices=[i for i, _ in eligible], eligible_ids=[r['id'] for _, r in eligible],
                excluded=excluded, split_check=split_check,
                temporal_protocol='Original consecutive RT-1 frames and actions; no padding, repetition, resampling, or GT refresh')
    plan_file = output / 'selection_plan.json'
    if plan_file.exists():
        assert read(plan_file) == plan, 'Existing output uses another dataset selection/protocol'
    else:
        write(plan_file, plan)
    assert eligible, f'No original dev episodes contain {horizon + 2} frames; exclusions saved in {plan_file}'
    spec = importlib.util.spec_from_file_location('rt1_original_record_reader', reader_file)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    data_dir = output / 'data'
    data_dir.mkdir(exist_ok=True)
    records = []
    for index, row in eligible:
        historical = row['historical']
        if historical is not None:
            assert sha(historical['path']) == historical['sha256'], f'Historical input checksum changed: {index}'
        images, actions, text = reader.read_images(row['original'], start=0, length=horizon + 2)
        assert images.shape == (horizon + 2, 256, 320, 3) and images.dtype == np.uint8, index
        assert actions.shape == (horizon + 2, 13) and actions.dtype == np.float32, index
        assert np.isfinite(actions).all() and text == row['text'], index
        if historical is not None:
            with np.load(historical['path'], allow_pickle=False) as old:
                assert np.array_equal(images[:34], old['image']), f'Initial 34 RGB frames changed: {index}'
                assert np.array_equal(actions[:34], old['action']), f'Initial 34 actions changed: {index}'
                assert text == str(old['instruction']), f'Instruction changed: {index}'
        dest = data_dir / f'dev_{index:03d}.npz'
        if dest.exists():
            with np.load(dest, allow_pickle=False) as cached:
                assert set(cached.files) == {'image', 'action', 'instruction'}
                assert np.array_equal(cached['image'], images)
                assert np.array_equal(cached['action'], actions)
                assert str(cached['instruction']) == text
        else:
            temp = dest.with_name(dest.name + f'.{os.getpid()}.partial')
            with temp.open('wb') as f:
                np.savez_compressed(f, image=images, action=actions, instruction=np.array(text))
            temp.replace(dest)
        records.append(dict(index=index, original_index=row['historical_index'], id=row['id'], path=str(dest), sha256=sha(dest),
                            text=text, clip_length=horizon + 2, start=0,
                            historical_input_path=historical['path'] if historical is not None else None,
                            historical_input_sha256=historical['sha256'] if historical is not None else None,
                            prefix34_frames_actions_instruction_exact=True if historical is not None else None,
                            original=row['original']))
        write(output / 'status.json', dict(state='PREPARING', cases=len(records), total_cases=len(eligible),
                                          latest_original_index=index, horizon=horizon))
        print('LONG_MEMORY_INPUT_READY', index, len(records), len(eligible), flush=True)
    manifest = dict(state='READY', **plan, selection_plan_sha256=sha(plan_file), cases=records,
                    cases_count=len(records), excluded_count=len(excluded),
                    prediction_inputs='Only images[:2], actions[:horizon+1], and unchanged instruction enter the model; images[2:] are scoring targets only')
    write(output / 'manifest.json', manifest)
    write(output / 'status.json', dict(state='READY', cases=len(records), total_cases=len(eligible),
                                      horizon=horizon, manifest_sha256=sha(output / 'manifest.json')))
    print('LONG_MEMORY_PREPARATION_COMPLETE', len(records), horizon, flush=True)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-base', type=Path, default=DEFAULT_BASE)
    p.add_argument('--reader', type=Path, default=DEFAULT_READER)
    p.add_argument('--horizon', type=int, choices=[64, 128], default=128)
    p.add_argument('--cohort', type=Path, help='Explicit predeclared held-out cohort JSON; otherwise filter the old 128 cases by length')
    args = p.parse_args()
    prepare(args.output, args.source_base, args.reader, args.horizon, args.cohort)
