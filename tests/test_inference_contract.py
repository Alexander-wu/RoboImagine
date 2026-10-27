"""Small deterministic checks for public input/path and rolling-history contracts."""
import tempfile,unittest
from pathlib import Path
import torch
from roboimagine.checkpoints import safe_path
from roboimagine.window import retained_indices,make_window_prefix,prune_history

class ContractTests(unittest.TestCase):
 def test_paths(self):
  with tempfile.TemporaryDirectory() as d:
   for path in ('../weights','/tmp/model','a/../../model'):
    with self.assertRaises(ValueError):safe_path(d,path)
   self.assertEqual(safe_path(d,'rt1/model'),Path(d).resolve()/'rt1/model')
 def test_fixed_anchor_and_recent_history(self):
  self.assertEqual(retained_indices(10,3,True),[1,7,8,9])
  self.assertEqual(retained_indices(10,6,True),[1,4,5,6,7,8,9])
  self.assertEqual(retained_indices(10,1,False),[9])
 def test_state_action_pairing(self):
  history={k:torch.tensor([[k]]) for k in range(1,10)}
  actions=torch.arange(11).reshape(11,1)+100
  prefix,kept=make_window_prefix(torch.tensor([[999]]),history,actions,10,3,True)
  self.assertEqual(prefix.tolist(),[[999,1,101,7,107,8,108,9,109]])
  history[10]=torch.tensor([[10]])
  self.assertEqual(list(prune_history(history,10,3,True)),[1,8,9,10])
 def test_invalid_window(self):
  for w in (0,-1,True):
   with self.assertRaises(ValueError):retained_indices(2,w,True)
if __name__=='__main__':unittest.main()
