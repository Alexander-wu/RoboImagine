"""Score all128 matched final predictions, without generating videos or training.

This controller never acquires a fifth physical GPU. The caller must provide the
four UUIDs already assigned to the named Bridge job and one UUID from that set.
Only this controller's own judge subprocess can be terminated by this script.
"""
import argparse
import base64
import hashlib
import json
import os
import signal
import subprocess
import time
import traceback
import urllib.request
import zlib
from pathlib import Path

import numpy as np

D = Path('/opt/roboimagine-workspace')
METHODS = ['sft', 'mcts500', 'ordinary500', 'diverse500']
UPDATES = dict(zip(METHODS, [0, 500, 500, 500]))
INDICES = list(range(128))
EXPECTED_SERVICE_SHA = '64aba33102879a71c74a7cc3346fc814e5a5aa34da038754aed5bc78a32c43be'
EXPECTED_PROMPT_SHA = '394edb1462134167dded4adcf7ea23956b0ba50ba17dea9414fe7d1ba87ded74'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def get_job(address, job):
    # Limited read-only retries do not start/stop/resubmit any job.
    error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(address.rstrip('/') + '/api/jobs/' + job,
                                        timeout=15) as response:
                return json.load(response)
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(2)
    raise RuntimeError('Unable to verify Bridge owner job') from error


def gpu_snapshot(uuids, source_root, own_pid=None):
    """Verify Bridge workers in the container's PID namespace and GPU mapping.

    nvidia-smi exposes host PIDs on this cluster; those PIDs cannot be looked up
    directly in the container /proc. Check worker command+environment instead.
    """
    gpu_rows = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.free,memory.used',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    all_gpus = {}
    for raw in gpu_rows:
        index, actual, free, used = [s.strip() for s in raw.split(',')]
        all_gpus[index] = {'index': int(index), 'uuid': actual,
                          'free_MiB': float(free), 'used_MiB': float(used)}
    def normalize(value):
        return value if value.startswith('GPU-') else all_gpus[value]['uuid']
    processes = {gpu: [] for gpu in uuids}
    for procdir in Path('/proc').iterdir():
        if not procdir.name.isdigit() or int(procdir.name) == own_pid:
            continue
        try:
            cmd = (procdir / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
            if str(source_root) + '/' not in cmd:
                continue
            worker = ('worker_entry.py' in cmd and 'train_retry' in cmd) or 'train_retry.py' in cmd
            judge = 'judge_service.py' in cmd
            if not worker and not judge:
                continue
            pairs = (procdir / 'environ').read_bytes().split(b'\0')
            env = dict(pair.split(b'=', 1) for pair in pairs if b'=' in pair)
            visible = env[b'CUDA_VISIBLE_DEVICES'].decode().split(',')
            if len(visible) == 4:
                if b'LOCAL_RANK' not in env:
                    continue  # torchrun launcher, not a CUDA worker
                assert len(set(visible)) == 4
                assert set(normalize(v) for v in visible) == set(uuids)
                local_rank = int(env[b'LOCAL_RANK'])
                assert 0 <= local_rank < 4
                gpu = normalize(visible[local_rank])
            else:
                assert len(visible) == 1
                gpu = normalize(visible[0])
            assert gpu in processes, ('Bridge worker outside declared four GPUs', cmd, gpu)
            processes[gpu].append({'container_pid': int(procdir.name), 'command': cmd,
                                   'kind': 'worker' if worker else 'judge',
                                   'cuda_visible_devices': visible,
                                   'local_rank': env.get(b'LOCAL_RANK', b'').decode()})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    result = []
    for gpu in uuids:
        row = next(r for r in all_gpus.values() if r['uuid'] == gpu).copy()
        assert sum(p['kind'] == 'worker' for p in processes[gpu]) == 1, ('Missing/duplicate Bridge worker', gpu, processes[gpu])
        assert sum(p['kind'] == 'judge' for p in processes[gpu]) == 1, ('Missing/duplicate Bridge judge', gpu, processes[gpu])
        host_pids = subprocess.check_output([
            'nvidia-smi', '-i', gpu, '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).splitlines()
        host_pids = [int(p.strip()) for p in host_pids if p.strip()]
        # No independent inference process is expected before our launch.
        assert len(host_pids) == 2, ('Unexpected GPU process count', gpu, host_pids)
        row.update(processes=processes[gpu], host_gpu_pids=host_pids,
                   identity_check='Container command/env and four-GPU rank mapping; host PIDs are in a different namespace')
        result.append(row)
    return result


def validate_prepared(prepared, expected_sha):
    assert sha(prepared) == expected_sha, 'Prepared receipt changed'
    prep = read(prepared)
    assert prep['methods'] == METHODS
    assert prep['indices'] == INDICES
    assert prep['generation_updates'] == UPDATES
    cases = prep['per_case']
    assert len(cases) == 128 and [c['index'] for c in cases] == INDICES
    assert len({c['id'] for c in cases}) == 128
    assert len({c['prepared_input_sha256'] for c in cases}) == 128
    source_rows = prep['source_completions']
    assert [row['method'] for row in source_rows] == METHODS
    assert len({row['initial_digest'] for row in source_rows}) == 1
    for row in source_rows:
        assert row['updates'] == UPDATES[row['method']]
        assert sha(row['path']) == row['sha256']
        complete = read(row['path'])
        assert complete['initial_digest'] == row['initial_digest']
        observed = complete['evaluation']
        assert len(observed) == 128
        assert [r['id'] for r in observed] == [c['id'] for c in cases]
        assert [r['prepared_input_sha256'] for r in observed] == [
            c['prepared_input_sha256'] for c in cases]
        method_index = METHODS.index(row['method'])
        for record, case in zip(observed, cases):
            artifact = case['artifacts'][method_index]
            assert artifact['method'] == row['method']
            assert record['artifact_sha256'] == artifact['sha256']
            for metric in ['psnr', 'ssim', 'lpips', 'mae', 'mse']:
                assert abs(record['metrics']['32'][metric] - case['metrics'][row['method']][metric]) < 1e-10
        if row['method'] != 'sft':
            assert complete['updates'] == 500
    # Confirm every source video and request before loading the GPU service.
    for case in cases:
        assert [a['method'] for a in case['artifacts']] == METHODS
        tokens = []
        obs_ref = gt_ref = instruction_ref = None
        request = Path(case['request_path'])
        assert sha(request) == case['request_sha256']
        with np.load(request, allow_pickle=False) as req:
            videos = req['videos']
            initial = req['initial']
            request_instruction = str(req['instruction'])
            assert videos.shape == (4, 32, 256, 320, 3)
            assert videos.dtype == np.uint8
            assert initial.shape == (256, 320, 3)
            assert initial.dtype == np.uint8
            for method_index, artifact in enumerate(case['artifacts']):
                assert sha(artifact['path']) == artifact['sha256']
                with np.load(artifact['path'], allow_pickle=False) as z:
                    t = z['tokens']; pred = z['prediction']
                    obs = z['observations']; gt = z['ground_truth']
                    instruction = str(z['instruction'])
                    assert t.shape == (1, 32, 80) and np.issubdtype(t.dtype, np.integer)
                    assert pred.shape == gt.shape == (32, 256, 320, 3)
                    assert obs.shape == (2, 256, 320, 3)
                    assert np.isfinite(pred).all() and 0 <= pred.min() <= pred.max() <= 1
                    assert hashlib.sha256(t.tobytes()).hexdigest() == artifact['token_sha256']
                    assert np.array_equal(videos[method_index], np.round(pred * 255).astype('uint8'))
                    assert np.array_equal(initial, obs[1])
                    assert request_instruction == instruction == case['instruction']
                    if obs_ref is None:
                        obs_ref, gt_ref, instruction_ref = obs.copy(), gt.copy(), instruction
                    else:
                        assert np.array_equal(obs_ref, obs) and np.array_equal(gt_ref, gt)
                        assert instruction_ref == instruction
                    tokens.append(t.copy())
        distance = np.array([[float(np.mean(a != b)) * 100 for b in tokens] for a in tokens])
        assert np.allclose(distance, case['token_distance_pct'], atol=1e-10, rtol=0)
        for method in METHODS:
            metrics = case['metrics'][method]
            assert all(np.isfinite(metrics[k]) for k in ['psnr', 'ssim', 'lpips', 'mae'])
        if (case['index'] + 1) % 16 == 0:
            print('VERIFIED_INPUTS', case['index'] + 1, 128, flush=True)
    for key, expected in [('judge_service.py', EXPECTED_SERVICE_SHA),
                          ('judge_prompt_snapshot.py', EXPECTED_PROMPT_SHA)]:
        entry = prep['sources'][key]
        assert entry['sha256'] == expected and sha(entry['path']) == expected
    return prep


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--expected-prepared-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--bridge-gpu-uuids', required=True)
    p.add_argument('--bridge-job-id', required=True)
    p.add_argument('--bridge-status', type=Path, required=True)
    p.add_argument('--bridge-source-root', type=Path, required=True)
    p.add_argument('--ray-address', default='http://RAY_DASHBOARD_HOST:8081')
    a = p.parse_args()
    uuids = a.bridge_gpu_uuids.split(',')
    assert len(uuids) == len(set(uuids)) == 4 and a.gpu_uuid in uuids
    assert all(u.startswith('GPU-') for u in uuids)
    # Fail instead of accepting an arbitrary owner supplied through command line.
    bridge = D / 'experiments/roboimagine_bridge_retry_20261026'
    assert a.bridge_job_id == 'roboimagine-bridge-retry-20261026-run-v1'
    assert a.bridge_status.resolve() == bridge / 'status.json'
    assert a.bridge_source_root.resolve() == bridge / 'source_v1'
    a.output.mkdir(parents=True, exist_ok=False)
    proc = None
    log = None
    started = time.time()
    overlap = None
    def write(name, value):
        atomic(a.output / name, value)
    def interrupt(sig, frame):
        raise RuntimeError(f'Interrupted by signal {sig}; preserve partial outputs')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        write('status.json', {'state': 'VERIFYING_INPUTS', 'started': started})
        prep = validate_prepared(a.prepared, a.expected_prepared_sha256)
        owner = get_job(a.ray_address, a.bridge_job_id)
        assert owner['status'] == 'RUNNING'
        owner_status = read(a.bridge_status)
        assert owner_status['state'] == 'RUNNING' and owner_status['stage'].endswith('_train')
        train_dir = bridge / 'results_v1' / owner_status['stage']
        train_status = read(train_dir / 'status.json')
        assert train_status['state'] == 'TRAINING' and train_status['target_updates'] == 500
        resources = gpu_snapshot(uuids, a.bridge_source_root)
        selected = next(row for row in resources if row['uuid'] == a.gpu_uuid)
        assert selected['free_MiB'] >= 48 * 1024, ('Less than48GiB free; defer scoring', selected)
        service = Path(prep['sources']['judge_service.py']['path']).read_text()
        changes = [
            ('video.shape==(10,256,320,3)', 'video.shape==(32,256,320,3)'),
            ('np.linspace(0,9,8)', 'np.linspace(0,31,8)'),
            ("z['videos'].shape==(4,10,256,320,3)", "z['videos'].shape==(4,32,256,320,3)"),
            ("'records_per_model':15600", "'records_per_model':1536"),
            ('torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False',
             'torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False\n'
             'torch.cuda.set_per_process_memory_fraction(16*2**30/torch.cuda.get_device_properties(0).total_memory)')]
        for old, new in changes:
            assert service.count(old) == 1, (old, service.count(old))
            service = service.replace(old, new)
        service_path = a.output / 'judge_service_h32.py'
        service_path.write_text(service)
        prompt_path = a.output / 'judge_prompt_snapshot.py'
        prompt_path.write_bytes(Path(prep['sources']['judge_prompt_snapshot.py']['path']).read_bytes())
        assert sha(prompt_path) == EXPECTED_PROMPT_SHA
        protocol = {
            'methods': METHODS, 'generation_updates': UPDATES, 'indices': INDICES,
            'cases': 128, 'predicted_frames': 32, 'ordered_pair_calls': 1536,
            'displayed_frame_indices_zero_based': np.linspace(0, 31, 8).round().astype(int).tolist(),
            'judge': 'Frozen training mJev4B; not an independent evaluator',
            'candidate_labels_blind_to_method': True,
            'score': 'mean antisymmetric P(A)-P(B) margin against all three other models',
            'source_service_sha256': EXPECTED_SERVICE_SHA,
            'service_h32_sha256': sha(service_path), 'prompt_source_sha256': EXPECTED_PROMPT_SHA,
            'changes': changes, 'prepared_sha256': a.expected_prepared_sha256,
            'physical_gpu_uuid': a.gpu_uuid, 'additional_physical_GPUs': 0,
            'memory_cap_GiB': 16, 'bridge_gpu_uuids': uuids,
            'bridge_job_id': a.bridge_job_id, 'bridge_start_status': owner_status,
            'bridge_training_start_status': train_status, 'gpu_preflight': resources,
            'controller_pid': os.getpid(), 'controller_sha256': sha(__file__),
            'driver_job_id': os.environ.get('RAY_JOB_ID'),
            'source_completions': prep['source_completions']}
        write('protocol.json', protocol)
        overlap = {'start_unix': time.time(), 'end_unix': None,
                   'physical_gpu_uuid': a.gpu_uuid, 'owner_job': a.bridge_job_id,
                   'owner_stage': owner_status['stage'], 'controller_pid': os.getpid(),
                   'note': 'Concurrent inference on one of the four already allocated Bridge devices; this interval is not uncontended training or scoring timing.'}
        write('training_overlap.json', overlap)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu_uuid,
                   PYTHONPATH=str(D / 'experiments/roboalign_r2_text_20261010/judge_mjev_v1/pydeps_transformers_5_14_1'),
                   PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        log = (a.output / 'judge.log').open('x')
        proc = subprocess.Popen([str(D / 'env/bin/python'), '-u', str(service_path),
                                 'mjev4b', str(a.output / 'judgments')],
                                env=env, stdin=subprocess.PIPE, stdout=log,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)
        protocol['judge_pid'] = proc.pid
        write('protocol.json', protocol)
        def wait(path):
            deadline = time.time() + 300
            while not path.exists():
                assert proc.poll() is None, 'Judge exited; inspect judge.log'
                assert time.time() < deadline, 'Judge response timeout'
                assert time.time() - started < 3 * 3600, 'Bounded controller timeout'
                time.sleep(.3)
            return read(path)
        ready = wait(a.output / 'judgments/READY.json')
        assert ready['pid'] == proc.pid
        cases = []
        for step, case in enumerate(prep['per_case'], 1):
            # Confirm ownership persists before each case; no running task is signaled.
            owner_now = read(a.bridge_status)
            assert owner_now['state'] == 'RUNNING' and owner_now['stage'] == owner_status['stage'], (
                'Bridge owner changed stage; stop own scorer', owner_now)
            assert read(train_dir / 'status.json')['state'] == 'TRAINING'
            request = Path(case['request_path'])
            assert sha(request) == case['request_sha256']
            response = a.output / f"case_{case['index']:03d}.reward.json"
            assert not response.exists()
            write('status.json', {'state': 'SCORING', 'completed': len(cases), 'total': 128,
                                  'index': case['index'], 'time_unix': time.time()})
            proc.stdin.write(json.dumps({'step': step, 'path': str(request), 'result': str(response)}) + '\n')
            proc.stdin.flush()
            answer = wait(response)
            assert answer['source_sha256'] == sha(request)
            assert len(answer['scores']) == 4 and len(answer['pairs']) == 6
            scores = np.zeros(4, dtype=np.float64)
            assert [(p['i'], p['j']) for p in answer['pairs']] == [
                (i, j) for i in range(4) for j in range(i + 1, 4)]
            for pair in answer['pairs']:
                pf, pb = pair['forward'], pair['reverse']
                for probability in [pf, pb]:
                    assert set(probability) == set('ABTU')
                    assert all(np.isfinite(v) and 0 <= v <= 1 for v in probability.values())
                    assert abs(sum(probability.values()) - 1) < 1e-5
                margin = .5 * ((pf['A'] - pf['B']) - (pb['A'] - pb['B']))
                assert abs(margin - pair['margin']) < 1e-9
                scores[pair['i']] += margin / 3
                scores[pair['j']] -= margin / 3
            assert np.allclose(scores, answer['scores'], atol=1e-9, rtol=0)
            assert abs(scores.sum()) < 1e-8
            cases.append(dict(case, semantic_feedback=answer))
            print('SCORED', step, 128, case['index'], answer['scores'], flush=True)
        proc.stdin.close()
        assert proc.wait(timeout=60) == 0
        log.close(); log = None
        final = read(a.output / 'judgments/complete.json')
        assert final['frozen_witness_unchanged'] and final['records'] == 1536
        assert final['state'] == 'SUCCEEDED' and final['checks'] and all(
            x['pass_check'] for x in final['checks'])
        raw = [read(p) for p in sorted((a.output / 'judgments').glob('g*_*.json'))]
        assert len(raw) == 1536 and len({r['id'] for r in raw}) == 1536
        assert {r['id'] for r in raw} == {
            f'g{step:04d}_p{i}{j}_{order}' for step in range(1, 129)
            for i in range(4) for j in range(i + 1, 4) for order in ['f', 'b']}
        assert all(not r['errors'] and r['mode'] == 'pair' and r['layout'] == 'candidate'
                   and r['arm'] == 'mjev4b' for r in raw)
        assert sha(a.prepared) == a.expected_prepared_sha256
        assert sha(service_path) == protocol['service_h32_sha256']
        assert sha(prompt_path) == EXPECTED_PROMPT_SHA
        complete = {
            'state': 'SUCCEEDED', 'methods': METHODS, 'generation_updates': UPDATES,
            'indices': INDICES, 'per_case': cases,
            'mean_token_distance_pct': np.mean([c['token_distance_pct'] for c in cases], axis=0).tolist(),
            'mean_semantic': np.mean([c['semantic_feedback']['scores'] for c in cases], axis=0).tolist(),
            'mean_metrics': {method: {k: float(np.mean([c['metrics'][method][k] for c in cases]))
                                      for k in cases[0]['metrics'][method]} for method in METHODS},
            'judge_completion': final, 'protocol': protocol, 'raw_judgments': raw,
            'started_unix': started, 'finished_unix': time.time()}
        write('complete.json', complete)
        write('status.json', {'state': 'SUCCEEDED', 'completed': 128, 'total': 128,
                              'ordered_judgments': 1536, 'time_unix': time.time()})
        print('EXPORT_ZLIB=' + base64.b64encode(zlib.compress(json.dumps(complete).encode())).decode(), flush=True)
    except BaseException:
        write('ERROR.json', {'error': traceback.format_exc(), 'time_unix': time.time()})
        write('status.json', {'state': 'FAILED', 'time_unix': time.time(),
                              'note': 'Partial outputs retained; no automatic retry or resubmission.'})
        raise
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        if log is not None:
            log.close()
        if overlap is not None:
            overlap['end_unix'] = time.time()
            if a.bridge_status.exists():
                overlap['owner_end_status'] = read(a.bridge_status)
            write('training_overlap.json', overlap)


if __name__ == '__main__':
    main()
