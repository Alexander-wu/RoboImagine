import subprocess,time,json,os
from common import D,E,ROOT,LEGACY,sha
class Judge:
 def __init__(self,out):
  self.out=out;self.log=(out/'judge_service.log').open('w');env=dict(os.environ);env['PYTHONPATH']=str(LEGACY/'judge_mjev_v1/pydeps_transformers_5_14_1')
  self.proc=subprocess.Popen([str(D/'env/bin/python'),'-u',str(ROOT/'judge_service.py'),'mjev4b',str(out/'judgments')],env=env,stdin=subprocess.PIPE,stdout=self.log,stderr=subprocess.STDOUT,text=True)
  self.wait(out/'judgments/READY.json')
 def wait(self,path):
  deadline=time.monotonic()+300
  while not path.exists():
   if self.proc.poll() is not None:raise RuntimeError('Judge exited: '+str(self.out/'judge_service.log'))
   if time.monotonic()>deadline:raise TimeoutError(str(path))
   time.sleep(.1)
  return json.loads(path.read_text())
 def score(self,step,request):
  response=request.with_suffix('.reward.json');assert not response.exists()
  self.proc.stdin.write(json.dumps(dict(step=step,path=str(request),result=str(response)))+'\n');self.proc.stdin.flush();ans=self.wait(response)
  assert ans['source_sha256']==sha(request);return ans
 def close(self):
  self.proc.stdin.close();self.proc.wait(timeout=60);assert self.proc.returncode==0
  final=json.loads((self.out/'judgments/complete.json').read_text());assert final['frozen_witness_unchanged'];self.log.close();return final
 def abort(self):
  if self.proc.poll() is None:
   self.proc.terminate()
   try:self.proc.wait(timeout=15)
   except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait()
  self.log.close()
