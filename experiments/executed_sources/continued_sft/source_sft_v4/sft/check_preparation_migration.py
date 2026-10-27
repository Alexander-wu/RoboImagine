"""CPU admission checks: only hardware metadata may differ from immutable v3."""
import ast,hashlib,json,tempfile
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

tree=ast.parse(Path(__file__).with_name('sft_common.py').read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['protocol','verified_preparation','own_seal']]
assert len(nodes)==3
with tempfile.TemporaryDirectory() as temp:
    r=Path(temp);s=r/'source_sft_v4/sft';s.mkdir(parents=True)
    legacy=r/'source_sft_v3/sft';legacy.mkdir(parents=True)
    e=r/'sft/rt1_sft500_v1';e.mkdir(parents=True)
    (s/'train.py').write_text('CURRENT4GPU\n');(legacy/'train.py').write_text('ORIGINAL8GPU\n')
    ns=dict(Path=Path,json=json,sha=sha,S=s,E=e,OLD=Path('/verified/old'),BASELINE=Path('/verified/base'),
            EXPECTED_INITIAL='verified_initializer',SEED=20261011,WORLD_SIZE=4,GLOBAL_BATCH=8,ACCUMULATION=2)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'sft_common.py','exec'),ns)
    current=ns['protocol']();old=dict(current)
    old.update(version='rt1_sft_continuation500_v1',world_size=8,scope='one synchronized 8-GPU training run; no independent training-seed claim')
    del old['gradient_accumulation']
    (e/'protocol.json').write_text(json.dumps(old));(e/'schedule.json').write_text('[0,1,2,3,4,5,6,7]')
    witnesses={'source':'frozen'}
    prepared=dict(state='READY',initial_digest='verified_initializer',witnesses=witnesses,
                  source={'train.py':sha(legacy/'train.py')},protocol_sha256=sha(e/'protocol.json'),
                  schedule_sha256=sha(e/'schedule.json'))
    (e/'prepared.json').write_text(json.dumps(prepared))
    saved={p:p.read_bytes() for p in e.iterdir()}
    _,admission=ns['verified_preparation'](witnesses)
    assert admission['hardware_only_migration'] and admission['current_protocol']['world_size']==4
    assert all(p.read_bytes()==b for p,b in saved.items())
    rejected=[]
    for change in ['optimizer','global_batch','source']:
        for p,b in saved.items():p.write_bytes(b)
        if change=='source':(legacy/'train.py').write_text('TAMPERED\n')
        else:
            altered=json.loads(json.dumps(old))
            if change=='optimizer':altered['optimizer']['lr']=1e-4
            else:altered['global_batch']=4
            (e/'protocol.json').write_text(json.dumps(altered))
            updated=dict(prepared,protocol_sha256=sha(e/'protocol.json'))
            (e/'prepared.json').write_text(json.dumps(updated))
        try:ns['verified_preparation'](witnesses)
        except AssertionError:rejected.append(change)
        else:raise AssertionError('Unauthorized migration passed: '+change)
    assert rejected==['optimizer','global_batch','source']
result=dict(state='PASS',hardware_only_admission=True,original_preparation_files_unchanged=True,
            rejected_nonhardware_changes=rejected)
out=Path(__file__).with_name('four_gpu_migration_checks.json');out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
