from original_common import D, MODEL, NAMES, MAXIMA, SYSTEM, PROTOCOL, sha, write, parse
import hashlib, json, os
from pathlib import Path
ROOT=D/'experiments/roboimagine_search_qwen97_20261026/v1'
SOURCE=ROOT/'source'
OLD=D/'experiments/r2_qwen8b_sixdim_rt1_100traj_20261014'
PILOT=D/'experiments/r2_rt1_search_qwen_pilot10_20261016/v1'
ORIGINAL=D/'experiments/r2_qwen8b_sixdim_20261014'
ARMS=['sft','mjev','metric','hybrid','sample','ordinary']
def sealed():
    for name,digest in json.loads((SOURCE/'source_manifest.json').read_text()).items():
        assert sha(SOURCE/name)==digest, name
    cohort=json.loads((SOURCE/'cohort.json').read_text())
    assert cohort['indices']==[i for i in range(100) if i not in [30,44,79]]
    assert sha(ORIGINAL/'common.py')==sha(SOURCE/'original_common.py')
    return cohort
