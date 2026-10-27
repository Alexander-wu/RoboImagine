"""Full-condition migration gate: proposals and feedback match the completed run."""
import json,math,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
D=R.parents[1]
old=D/'experiments/r2_separate_controls_20261012/rt1_hybrid500/smoke8'
new=R/'parity_diverse/smoke8'
checks=[]
for rank in range(8):
 a=json.loads((old/f'rank{rank}/train.jsonl').read_text().splitlines()[0])
 b=json.loads((new/f'rank{rank}/train.jsonl').read_text().splitlines()[0])
 assert a['episode']==b['episode'] and a['proposal_tokens_sha256']==b['proposal_tokens_sha256'],('Proposal drift',rank)
 for key in ['semantic','rewards']:
  assert len(a[key])==len(b[key])==4
  assert all(math.isclose(x,y,rel_tol=2e-5,abs_tol=2e-5) for x,y in zip(a[key],b[key])),('Feedback drift',rank,key)
 checks.append(dict(rank=rank,episode=b['episode'],proposal_sha256=b['proposal_tokens_sha256'],feedback_match=True))
out=dict(state='PASS',scope='Eight identical full-text diverse-search smoke trajectories and feedback; independent of dev outcomes',checks=checks)
(R/'parity_check.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
