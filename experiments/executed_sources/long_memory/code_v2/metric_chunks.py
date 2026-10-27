"""Keep native metric reduction shapes at 32 frames for all horizons."""
import numpy as np
def score_in_fixed_chunks(evaluator,prediction,truth,horizons,block=32):
 if len(prediction)!=len(truth) or len(prediction)%block or block!=32:
  raise ValueError('Expected aligned videos with length a multiple of32')
 vectors={}
 for start in range(0,len(prediction),block):
  scored=evaluator(prediction[start:start+block],truth[start:start+block],[block])
  if vectors and set(vectors)!=set(scored['per_frame']):raise ValueError('Metric schema changed')
  for key,values in scored['per_frame'].items():
   if len(values)!=block or not np.isfinite(values).all():raise ValueError('Nonfinite or incomplete block')
   vectors.setdefault(key,[]).extend(values)
 if any(h<1 or h>len(prediction) for h in horizons):raise ValueError('Invalid prefix horizon')
 return dict(per_frame=vectors,metrics={str(h):{k:float(np.mean(v[:h])) for k,v in vectors.items()} for h in horizons})
