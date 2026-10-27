import numpy as np
from shared import *
def aggregate():
    indices=sealed()['indices'];records={}
    for sub in ['cached_judgments','judgments']:
        for p in (ROOT/sub).glob('*.json'):
            r=json.loads(p.read_text());assert r['uid'] not in records
            records[r['uid']]=r
    assert len(records)==582
    assert set(records)=={f'rt1_{i:03d}_{arm}' for i in indices for arm in ARMS}
    def valid(r,primary=False):
        if not r['valid']:return False
        good=next(a for a in r['attempts'] if a.get('error') is None)
        assert parse(good['raw_response'])['scores']==r['judgment']['scores']
        return not primary or good['attempt']<2
    result={'evaluator':PROTOCOL['model_id'],'target_indices':indices,'planned':582,
            'valid_counts':{a:sum(valid(records[f'rt1_{i:03d}_{a}']) for i in indices) for a in ARMS},
            'invalid_uids':[uid for uid,r in records.items() if not valid(r)],
            'format_repaired_uids':[uid for uid,r in records.items() if valid(r) and not valid(r,True)],
            'identity':json.loads((ROOT/'identity.json').read_text()),'records':list(records.values())}
    for primary,key in [(True,'original_protocol'),(False,'with_bounded_format_repair')]:
        common=[i for i in indices if all(valid(records[f'rt1_{i:03d}_{a}'],primary) for a in ARMS)]
        assert common
        arr={a:np.array([[records[f'rt1_{i:03d}_{a}']['judgment']['scores'][k] for k in NAMES] for i in common]) for a in ARMS}
        rng=np.random.default_rng(20261026);sample=rng.integers(0,len(common),(10000,len(common)))
        rows=[]
        for a in ARMS:
            scores=arr[a];total=scores.sum(1);ci=np.quantile(total[sample].mean(1),[.025,.975])
            rows.append({'arm':a,'n':len(common),'mean_scores':dict(zip(NAMES,scores.mean(0).tolist())),'total':float(total.mean()),'ci95':ci.tolist()})
        pair={}
        for a in ARMS:
            delta=arr[a].sum(1)-arr['sft'].sum(1)
            pair[a]={'mean_delta_vs_sft':float(delta.mean()),'paired_ci95':np.quantile(delta[sample].mean(1),[.025,.975]).tolist()}
        result[key]={'common_valid_indices':common,'excluded_indices':[i for i in indices if i not in common],'rows':rows,'paired_vs_sft':pair}
    result['complete']=True;result['state']='SUCCEEDED' if not result['invalid_uids'] else 'COMPLETED_WITH_INVALID'
    write(ROOT/'summary.json',result)
    return result

if __name__=='__main__':aggregate()
