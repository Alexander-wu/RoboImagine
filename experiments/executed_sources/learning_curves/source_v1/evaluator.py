"""Frozen-checkpoint RT-1 learning-curve evaluation, separate from training."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

D = Path('/opt/roboimagine-workspace')
HYBRID = D/'experiments/r2_separate_controls_20261012/rt1_hybrid500'
SFT = Path('/opt/roboimagine-scratch/r2_supplement_20261023/sft/rt1_sft500_v1')
BASELINE = D/'experiments/rt1_mjev_20261011/sft_reference_v1/complete.json'
SOURCE = Path(__file__).resolve().parent
INITIAL = 'e6c3a7b5db0d677f55ece747ddec6edf7f60b5b901a95b14a8d575c8c5a27865'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def immutable_json(path,value):
    path=Path(path)
    if path.exists():assert read(path)==value,('Existing immutable JSON differs',str(path))
    else:write(path,value)


def receipt_check(checkpoint, step):
    rec=read(checkpoint/'receipt.json')
    assert rec['updates']==step and rec['initial_digest']==INITIAL
    assert {'model.safetensors','text_adapters.safetensors'}.issubset(rec['files'])
    for name,expected in rec['files'].items():
        assert Path(name).name==name and sha(checkpoint/name)==expected,('Checkpoint file mismatch',name)
    return rec


def verify_sft_preparation(target):
    """Verify the immutable v3 cache provenance and admitted v4 training protocol."""
    admission_path=SFT/'four_gpu_preparation_admission.json'
    admission=read(admission_path);prepared=read(SFT/'prepared.json');legacy_protocol=read(SFT/'protocol.json')
    current=admission['current_protocol']
    assert admission['state']==prepared['state']=='READY'
    assert admission['original_prepared_sha256']==sha(SFT/'prepared.json')
    assert admission['original_protocol_sha256']==prepared['protocol_sha256']==sha(SFT/'protocol.json')
    assert admission['schedule_sha256']==prepared['schedule_sha256']==sha(SFT/'schedule.json')
    assert admission['witnesses']==prepared['witnesses']
    assert prepared['initial_digest']==current['initial_digest']==target['initial_digest']==INITIAL
    assert admission['original_preparation_source']==prepared['source']
    assert admission['current_source']==target['source']
    assert current==target['protocol']
    assert current['version']=='rt1_sft_continuation500_v2_4gpu'
    assert current['updates']==500 and current['world_size']==4
    assert current['global_batch']==8 and current['gradient_accumulation']==2
    provenance={str(SFT/name):sha(SFT/name) for name in [
        'four_gpu_preparation_admission.json','prepared.json','protocol.json','schedule.json']}
    source_match=None
    for candidate in sorted(SFT.parents[1].glob('source_sft_*/sft')):
        actual={p.name:sha(p) for p in candidate.glob('*.py')}
        if actual==target['source']:
            source_match=str(candidate)
            provenance.update({str(candidate/name):value for name,value in actual.items()});break
    assert source_match is not None,'SFT checkpoint training source is not verifiable'
    if admission['hardware_only_migration']:
        legacy=SFT.parents[1]/'source_sft_v3/sft'
        actual={p.name:sha(p) for p in legacy.glob('*.py')}
        assert legacy.is_dir() and actual==prepared['source']
        assert legacy_protocol['version']=='rt1_sft_continuation500_v1'
        assert legacy_protocol['world_size']==8 and legacy_protocol['global_batch']==8
        allowed={'version','world_size','gradient_accumulation','scope'}
        assert {k:v for k,v in legacy_protocol.items() if k not in allowed}=={
            k:v for k,v in current.items() if k not in allowed},'Non-hardware protocol changes'
        provenance.update({str(legacy/name):value for name,value in actual.items()})
    else:
        assert legacy_protocol==current and prepared['source']==target['source']
    return source_match,provenance


def make_protocol(arm,step):
    assert arm in ['hybrid','sft'] and step in [100,250]
    seal=read(HYBRID/'source_manifest.json')
    for name,expected in seal.items():assert sha(HYBRID/name)==expected,('Original source changed',name)
    selection=read(HYBRID/'selection.json');prepared=read(HYBRID/'prepared.json')
    assert sha(HYBRID/'selection.json')==prepared['selection_sha256']
    baseline=read(BASELINE);full=read(HYBRID/'run/complete.json')
    assert full['state']=='SUCCEEDED' and full['updates']==500 and full['initial_digest']==INITIAL
    assert baseline['state']=='SUCCEEDED' and baseline['initial_digest']==INITIAL
    rows=selection['dev']
    assert len(rows)==len({r['id'] for r in rows})==128
    assert [r['id'] for r in rows]==[r['id'] for r in full['evaluation']]==[r['id'] for r in baseline['evaluation']]
    for i,(row,b,f) in enumerate(zip(rows,baseline['evaluation'],full['evaluation'])):
        assert b['index']==f['index']==i
        assert row['sha256']==b['prepared_input_sha256']==f['prepared_input_sha256']
        assert sha(row['path'])==row['sha256']
    reference=HYBRID/'run/checkpoint_000500';ref=receipt_check(reference,500)
    assert ref['digest']==full['final_digest']
    checkpoint=(HYBRID if arm=='hybrid' else SFT)/f'run/checkpoint_{step:06d}'
    target=receipt_check(checkpoint,step)
    source_match=None;training_provenance={}
    if arm=='hybrid':
        assert target['protocol']==read(HYBRID/'protocol.json')
    else:
        source_match,training_provenance=verify_sft_preparation(target)
    return dict(version=1,experiment='rt1_sft50k_posttraining_curve',arm=arm,step=step,
                checkpoint=str(checkpoint),checkpoint_digest=target['digest'],initial_digest=INITIAL,
                checkpoint_receipt_sha256=sha(checkpoint/'receipt.json'),
                reference_checkpoint=str(reference),reference_digest=ref['digest'],
                reference_receipt_sha256=sha(reference/'receipt.json'),
                original_complete=str(HYBRID/'run/complete.json'),original_complete_sha256=sha(HYBRID/'run/complete.json'),
                baseline=str(BASELINE),baseline_sha256=sha(BASELINE),
                selection=str(HYBRID/'selection.json'),selection_sha256=sha(HYBRID/'selection.json'),
                old_source_manifest_sha256=sha(HYBRID/'source_manifest.json'),
                training_source_match=source_match,training_provenance_files=training_provenance,
                source_files={p.name:sha(p) for p in SOURCE.glob('*.py')},
                cases=128,horizon=32,beam_width=4,memory='full',precision='FP32; TF32 disabled',
                primary='mean predicted frames1..32 within each trajectory, then mean all128 trajectories',
                eval_seed='20261011 + 10000 + original_index',inference_inputs='Only two observed images, planned actions and task instruction',
                newly_trained_updates=0,bootstrap_resamples=5000,bootstrap_seed=20261023,
                uncertainty='Paired trajectory bootstrap; not independent training-seed uncertainty',
                parity='One native hybrid500 replay per rank; exact tokens, pixels atol1e-6 and per-frame metrics atol1e-5 before target inference',
                parity_pixel_atol=1e-6,parity_metric_atol=1e-5)


def check_protocol(output):
    p=read(output/'protocol.json')
    for name,value in p['source_files'].items():assert sha(SOURCE/name)==value,name
    for name,value in p['training_provenance_files'].items():assert sha(name)==value,name
    for file,key in [(HYBRID/'source_manifest.json','old_source_manifest_sha256'),
                     (HYBRID/'selection.json','selection_sha256'),(BASELINE,'baseline_sha256'),
                     (HYBRID/'run/complete.json','original_complete_sha256'),
                     (Path(p['checkpoint'])/'receipt.json','checkpoint_receipt_sha256'),
                     (Path(p['reference_checkpoint'])/'receipt.json','reference_receipt_sha256')]:
        assert sha(file)==p[key],str(file)
    return p


def validate_trace(trace):
    assert len(trace)==32
    for t,r in enumerate(trace,2):
        assert r['target']==t and r['retained']==[1]+list(range(max(2,t-6),t)),(t,r)


def validate_scored(scored,np):
    assert {'psnr','ssim','lpips','mae','mse'}.issubset(scored['per_frame'])
    for key,values in scored['per_frame'].items():
        assert len(values)==32 and np.isfinite(values).all()
        for h in [10,20,32]:assert abs(float(np.mean(values[:h]))-scored['metrics'][str(h)][key])<1e-5


def validate_record(rec,row,index,protocol,protocol_sha,artifact,np):
    assert rec['index']==index and rec['id']==row['id']
    assert rec['prepared_input_sha256']==row['sha256']
    assert rec['checkpoint_digest']==protocol['checkpoint_digest']
    assert rec['arm']==protocol['arm'] and rec['step']==protocol['step']
    assert rec['protocol_sha256']==protocol_sha and sha(artifact)==rec['artifact_sha256']
    validate_trace(rec['trace']);validate_scored(rec,np)


def worker(args):
    output=args.output;rank=args.rank;shards=args.workers
    assert shards==4 and 0<=rank<4
    root=output/f'workers4/rank{rank}';root.mkdir(parents=True,exist_ok=True)
    lock=(root/'worker.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=check_protocol(output);protocol_sha=sha(output/'protocol.json')
    assert protocol['arm']==args.arm and protocol['step']==args.step
    sys.path.insert(0,str(HYBRID))
    from common import load_initial,setup,pixels,digest
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from futureworlds.evaluation import Metrics
    own=[(i,r) for i,r in enumerate(read(HYBRID/'selection.json')['dev']) if i%4==rank]
    assert len(own)==32
    def status(state,**kw):write(root/'status.json',dict(state=state,rank=rank,time=time.time(),**kw))
    def load_model(pipe,path,expected):
        receipt=read(path/'receipt.json')
        for name in ['model.safetensors','text_adapters.safetensors']:assert sha(path/name)==receipt['files'][name]
        pipe.model.core.load_state_dict(load_file(str(path/'model.safetensors'),device='cuda:0'),strict=True)
        pipe.model.adapters.load_state_dict(load_file(str(path/'text_adapters.safetensors'),device='cuda:0'),strict=True)
        pipe.model.eval().requires_grad_(False)
        assert digest(pipe.model)==receipt['digest']==expected
        assert all(v.dtype==torch.float32 for v in pipe.model.parameters())
    started=time.monotonic()
    try:
        torch.cuda.set_device(0);setup(20261011)
        status('LOADING',cases=0,total=32)
        pipe=load_initial();assert digest(pipe.model)==INITIAL
        metric=Metrics('cuda:0');torch.cuda.reset_peak_memory_stats()
        load_model(pipe,Path(protocol['reference_checkpoint']),protocol['reference_digest'])
        index,row=own[0];setup(20261011+10000+index)
        images,actions,text=pixels(row)
        native=pipe.predict(images[:2].copy(),actions[:33].copy(),text,horizon=32,beam_width=4,memory='full')
        validate_trace(native['trace'])
        original=read(HYBRID/'run/complete.json')['evaluation'][index]
        original_artifact=HYBRID/f'run/rank{index%8}/dev_{index:03d}.npz'
        assert sha(original_artifact)==original['artifact_sha256']
        with np.load(original_artifact,allow_pickle=False) as z:
            assert np.array_equal(native['tokens'],z['tokens']),'Native baseline token mismatch'
            pixel_error=float(np.max(np.abs(native['prediction']-z['prediction'])))
            assert pixel_error<=protocol['parity_pixel_atol'],('Native baseline pixels differ',pixel_error)
            assert np.array_equal(images[:2],z['observations']) and np.array_equal(images[2:34],z['ground_truth'])
            assert text==str(z['instruction'])
        scored=metric(native['prediction'],images[2:34],[10,20,32]);validate_scored(scored,np)
        metric_errors={k:float(np.max(np.abs(np.asarray(v)-np.asarray(original['per_frame'][k])))) for k,v in scored['per_frame'].items()}
        assert all(v<=protocol['parity_metric_atol'] for v in metric_errors.values()),('Native baseline metrics differ',metric_errors)
        identity=metric(images[2:34].astype('float32')/255,images[2:34],[32])['metrics']['32']
        assert abs(identity['ssim']-1)<1e-5 and abs(identity['lpips'])<1e-5
        parity=dict(state='PASS',rank=rank,index=index,id=row['id'],reference_digest=protocol['reference_digest'],
                    artifact_sha256=original['artifact_sha256'],exact_tokens=True,max_pixel_error=pixel_error,
                    pixel_atol=protocol['parity_pixel_atol'],metric_atol=protocol['parity_metric_atol'],
                    per_frame_metric_max_errors=metric_errors,protocol_sha256=protocol_sha)
        immutable_json(root/'parity.json',parity)
        del native;pipe.text_cache.clear()
        load_model(pipe,Path(protocol['checkpoint']),protocol['checkpoint_digest'])
        records=[];computed=0
        for index,row in own:
            record_path=root/f'dev_{index:03d}.json';artifact=root/f'dev_{index:03d}.npz'
            if record_path.exists():
                record=read(record_path);validate_record(record,row,index,protocol,protocol_sha,artifact,np)
            else:
                setup(20261011+10000+index);images,actions,text=pixels(row)
                torch.cuda.synchronize();tick=time.monotonic()
                result=pipe.predict(images[:2].copy(),actions[:33].copy(),text,horizon=32,beam_width=4,memory='full')
                torch.cuda.synchronize();infer=time.monotonic()-tick
                prediction=np.asarray(result['prediction'],dtype='float32');gt=images[2:34]
                assert prediction.shape==(32,256,320,3) and np.isfinite(prediction).all()
                validate_trace(result['trace']);scored=metric(prediction,gt,[10,20,32]);validate_scored(scored,np)
                values=dict(prediction=prediction,tokens=result['tokens'],ground_truth=gt,observations=images[:2],instruction=np.array(text))
                if artifact.exists():
                    # A crash may leave an atomic NPZ without its JSON commit record.
                    with np.load(artifact,allow_pickle=False) as z:
                        assert set(z.files)==set(values)
                        assert all(np.array_equal(z[k],v) for k,v in values.items()),'Orphan artifact differs'
                else:
                    tmp=artifact.with_name(artifact.name+'.'+uuid.uuid4().hex+'.partial')
                    with tmp.open('wb') as f:
                        np.savez_compressed(f,**values);f.flush();os.fsync(f.fileno())
                    os.replace(tmp,artifact)
                record=dict(id=row['id'],index=index,arm=args.arm,step=args.step,checkpoint_digest=protocol['checkpoint_digest'],
                            protocol_sha256=protocol_sha,prepared_input_sha256=row['sha256'],artifact_sha256=sha(artifact),
                            trace=result['trace'],metrics=scored['metrics'],per_frame=scored['per_frame'],
                            frame32={k:v[-1] for k,v in scored['per_frame'].items()},inference_seconds=infer)
                immutable_json(record_path,record);computed+=1
            records.append(record);pipe.text_cache.clear()
            status('EVALUATING',cases=len(records),total=32,new_cases=computed)
            print('CURVE_EVAL',args.arm,args.step,rank,len(records),32,flush=True)
        assert len(records)==32 and digest(pipe.model)==protocol['checkpoint_digest']
        complete=dict(state='SUCCEEDED',rank=rank,cases=32,arm=args.arm,step=args.step,
                      checkpoint_digest=protocol['checkpoint_digest'],protocol_sha256=protocol_sha,
                      record_hashes={f"dev_{r['index']:03d}.json":sha(root/f"dev_{r['index']:03d}.json") for r in records},
                      parity_sha256=sha(root/'parity.json'),new_training_updates=0)
        immutable_json(root/'complete.json',complete)
        write(root/('attempt_'+uuid.uuid4().hex+'.json'),dict(seconds=time.monotonic()-started,new_cases=computed,
                                                        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))
        status('SUCCEEDED',cases=32,total=32)
    except BaseException:
        write(root/('ERROR_'+uuid.uuid4().hex+'.json'),dict(rank=rank,error=traceback.format_exc(),time=time.time()))
        status('FAILED');raise


def aggregate(output):
    import numpy as np
    protocol=check_protocol(output);protocol_sha=sha(output/'protocol.json')
    rows=read(HYBRID/'selection.json')['dev'];baseline=read(BASELINE)['evaluation'];records=[];parities=[]
    for rank in range(4):
        root=output/f'workers4/rank{rank}';done=read(root/'complete.json');parity=read(root/'parity.json')
        assert done['state']=='SUCCEEDED' and done['cases']==32 and done['rank']==rank
        assert done['protocol_sha256']==protocol_sha and done['checkpoint_digest']==protocol['checkpoint_digest']
        assert done['parity_sha256']==sha(root/'parity.json')
        assert parity['state']=='PASS' and parity['exact_tokens']
        assert parity['pixel_atol']==protocol['parity_pixel_atol'] and parity['metric_atol']==protocol['parity_metric_atol']
        assert parity['max_pixel_error']<=protocol['parity_pixel_atol']
        assert all(v<=protocol['parity_metric_atol'] for v in parity['per_frame_metric_max_errors'].values())
        parities.append(parity)
        for i in range(rank,128,4):
            path=root/f'dev_{i:03d}.json';assert sha(path)==done['record_hashes'][path.name]
            rec=read(path);validate_record(rec,rows[i],i,protocol,protocol_sha,path.with_suffix('.npz'),np)
            records.append(rec)
    records.sort(key=lambda r:r['index'])
    assert [r['index'] for r in records]==list(range(128))
    assert [r['id'] for r in records]==[r['id'] for r in baseline]
    metrics=['psnr','ssim','lpips','mae','mse'];boot=np.random.default_rng(20261023).integers(0,128,size=(5000,128))
    means={str(h):{k:float(np.mean([r['metrics'][str(h)][k] for r in records])) for k in metrics} for h in [10,20,32]}
    contrasts={};mean_ci={}
    for key in metrics:
        values=np.array([r['metrics']['32'][key] for r in records])
        delta=values-np.array([r['metrics']['32'][key] for r in baseline])
        contrasts[key]=dict(delta=float(delta.mean()),ci95=[float(v) for v in np.quantile(delta[boot].mean(1),[.025,.975])])
        mean_ci[key]=[float(v) for v in np.quantile(values[boot].mean(1),[.025,.975])]
    result=dict(state='SUCCEEDED',arm=protocol['arm'],updates=protocol['step'],cases=128,
                initial_digest=INITIAL,final_digest=protocol['checkpoint_digest'],protocol=protocol,
                protocol_sha256=protocol_sha,mean_frames1_32=means['32'],means=means,
                mean_trajectory_ci95=mean_ci,paired_delta_vs_sft=contrasts,evaluation=records,
                reference_parity=parities,new_training_updates=0,
                statistics='All128 trajectories,5000 paired bootstrap resamples; no sample/checkpoint selection; not training-seed uncertainty')
    immutable_json(output/'complete.json',result)
    print('CURVE_COMPLETE',protocol['arm'],protocol['step'],json.dumps(means['32']),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['hybrid','sft'],required=True)
    p.add_argument('--step',type=int,choices=[100,250],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=[4],default=4);p.add_argument('--rank',type=int)
    p.add_argument('--preflight',action='store_true');p.add_argument('--aggregate',action='store_true');args=p.parse_args()
    if args.preflight:
        args.output.mkdir(parents=True,exist_ok=True);immutable_json(args.output/'protocol.json',make_protocol(args.arm,args.step))
        print('CURVE_PREFLIGHT_PASS',args.arm,args.step,flush=True)
    elif args.aggregate:aggregate(args.output)
    else:
        assert args.rank is not None
        worker(args)


if __name__=='__main__':main()
