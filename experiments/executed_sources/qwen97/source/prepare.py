import time,shutil,statistics
import numpy as np
from shared import *
scope=sealed();indices=scope['indices']
old_identity=json.loads((ORIGINAL/'identity.json').read_text())
assert old_identity['evaluation_id']==scope['base_scoring_identity']
mi=json.loads((ORIGINAL/'model_identity.json').read_text())
assert sha(ORIGINAL/'model_identity.json')==old_identity['model_identity_sha256']
for name,digest in mi['files'].items():assert sha(MODEL/name)==digest,name
print('MODEL_VERIFIED',flush=True)
for sub in ['cached_judgments','historical_invalid','judgments','attempts','claims']:(ROOT/sub).mkdir(exist_ok=True)
expected=json.loads((SOURCE/'expected_sources.json').read_text())
rows=[];cached=[];physical=[];contexts={};initial=None
for arm in ARMS:
    src=expected[arm];complete=Path(src['path']);assert sha(complete)==src['sha256']
    d=json.loads(complete.read_text());assert d['initial_digest']==src['initial_digest']
    if initial is None:initial=d['initial_digest']
    assert d['initial_digest']==initial
    if arm!='sft':assert d['updates']==500
    ev={x['index']:x for x in d['evaluation']};assert set(ev)==set(range(128))
    candidates={}
    for p in complete.parent.glob('**/dev_*.npz'):
        i=int(p.stem.split('_')[-1]);assert i not in candidates,(arm,i);candidates[i]=p
    assert set(candidates)==set(range(128))
    for i in indices:
        rec=ev[i];p=candidates[i];assert sha(p)==rec['artifact_sha256']
        with np.load(p,allow_pickle=False) as z:
            obs=z['observations'];pred=z['prediction'];gt=z['ground_truth'];ins=str(z['instruction'].item())
            assert obs.shape==(2,256,320,3) and obs.dtype==np.uint8
            assert pred.shape==gt.shape==(32,256,320,3) and np.isfinite(pred).all()
            context=hashlib.sha256(obs.tobytes()+gt.tobytes()+ins.encode()).hexdigest()
            h=hashlib.sha256(obs[1].tobytes()+np.rint(pred.clip(0,1)*255).astype(np.uint8)[PROTOCOL['sampled_indices']].tobytes()+ins.encode()).hexdigest()
        triple=(rec['id'],rec['prepared_input_sha256'],context)
        if i in contexts:assert contexts[i]==triple,(arm,i)
        else:contexts[i]=triple
        uid=f'rt1_{i:03d}_{arm}'
        row=dict(uid=uid,dataset='rt1',arm=arm,index=i,episode_id=rec['id'],path=str(p),artifact_sha256=rec['artifact_sha256'],prepared_input_sha256=rec['prepared_input_sha256'],context_sha256=context,expected_judge_input_sha256=h,seed=20261014+i,kind='prediction')
        physical.append(dict(arm=arm,index=i,metrics=rec['metrics']['32']))
        old=(OLD if arm in ARMS[:4] else PILOT)/'judgments'/(uid+'.json')
        if old.exists():
            prior=json.loads(old.read_text());assert prior['artifact_sha256']==row['artifact_sha256']
            assert prior['judge_input_sha256']==h and prior['prompt_sha256']==PROTOCOL['prompt_sha256']
            assert prior['episode_id']==row['episode_id']
            assert prior['evaluation_id']==(old_identity['evaluation_id'] if arm in ARMS[:4] else json.loads((PILOT/'identity.json').read_text())['evaluation_id'])
            if prior['valid']:
                good=next(a for a in prior['attempts'] if a.get('error') is None)
                assert parse(good['raw_response'])['scores']==prior['judgment']['scores']
                dest=ROOT/'cached_judgments'/old.name;shutil.copy2(old,dest)
                cached.append(dict(uid=uid,source=str(old),path=str(dest),sha256=sha(old)))
                continue
            assert arm in ['sample','ordinary'] and len(prior['attempts'])==2
            dest=ROOT/'historical_invalid'/old.name;shutil.copy2(old,dest)
            row['prior_invalid_path']=str(dest);row['prior_invalid_sha256']=sha(dest)
        else:assert arm in ['sample','ordinary']
        rows.append(row)
    print('SOURCE_VERIFIED',arm,len(indices),flush=True)
assert len(physical)==582 and len(cached)==407 and len(rows)==175
identity={'base_scoring_identity':old_identity['evaluation_id'],'protocol':PROTOCOL,'cohort':scope,'source_sha256':sha(SOURCE/'source_manifest.json'),'source_completions':expected,'model_identity_sha256':sha(ORIGINAL/'model_identity.json')}
identity['evaluation_id']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
write(ROOT/'identity.json',identity)
write(ROOT/'manifest.json',dict(identity=identity['evaluation_id'],indices=indices,rows=sorted(rows,key=lambda x:(x['index'],x['arm']))))
write(ROOT/'cache_receipts.json',cached)
write(ROOT/'physical_metrics.json',physical)
write(ROOT/'prepared.json',dict(state='SUCCEEDED',planned=582,cached_valid=407,pending=175,time=time.time()))
