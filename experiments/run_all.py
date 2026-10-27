#!/usr/bin/env python3
"""Offline redraw only: never trains a model, submits Ray jobs or calls an API."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys,time,os
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--figures',nargs='*',type=int,default=list(range(6,15)));ap.add_argument('--output',type=Path,default=ROOT/'reproduced');args=ap.parse_args()
 dest=args.output.resolve();dest.mkdir(parents=True,exist_ok=True)
 entries=json.loads((ROOT/'figure_index.json').read_text());checks=json.loads((ROOT/'input_checksums.json').read_text())
 for p,h in checks.items():assert sha(ROOT/p)==h,f'Input changed: {p}'
 shutil.copytree(ROOT/'common',dest/'common',dirs_exist_ok=True)
 report=[]
 for e in entries:
  n=e['figure']
  if n not in args.figures:continue
  src=ROOT/e['directory'];out=dest/e['directory'];shutil.copytree(src,out,dirs_exist_ok=True)
  start=time.monotonic();env=os.environ.copy();env['MPLBACKEND']='Agg';env['MPLCONFIGDIR']=str(dest/'.mplconfig')
  with (out/'run.log').open('w') as log:
   result=subprocess.run([sys.executable,str(out/e['entry'])],cwd=out,stdout=log,stderr=subprocess.STDOUT,env=env)
  if result.returncode:print((out/'run.log').read_text()[-3500:]);raise RuntimeError(f'Figure{n}: see {out}/run.log')
  assert (out/e['pdf']).is_file(),e
  report.append({'figure':n,'state':'PASS','seconds':round(time.monotonic()-start,2),'pdf':str((out/e['pdf']).relative_to(dest)),'sha256':sha(out/e['pdf'])})
  print(f'Fig {n:02d} PASS ({report[-1]["seconds"]} s)',flush=True)
 for p,h in checks.items():assert sha(ROOT/p)==h,f'Source modified: {p}'
 (dest/'execution.json').write_text(json.dumps(report,indent=2))
 print('Offline redraw complete. Source inputs unchanged; no training or inference performed.')
if __name__=='__main__':main()
