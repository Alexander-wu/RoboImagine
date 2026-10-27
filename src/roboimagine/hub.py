"""Hash-checked public weights; no private filesystem or cluster dependency."""
import json,shutil
from pathlib import Path
from .checkpoints import safe_path,file_sha256

REPO_ID='easylearning/RoboImagine'
# Immutable submission release; every downloaded asset is SHA-256 checked.
REVISION='cb4f5e2bdfbbbff2ad6caffa058d50fa21ec3ec7'

def download_bundle(dataset,variant='main',destination='weights',revision=REVISION,example_index=0):
    from huggingface_hub import hf_hub_download
    if dataset not in ('rt1','bridge','robocasa') or variant not in ('main','sft'):
        raise ValueError('Select rt1/bridge/robocasa and main/sft')
    manifest_path=hf_hub_download(REPO_ID,'release_manifest.json',revision=revision)
    release=json.loads(Path(manifest_path).read_text())
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    manifest=root/'manifest.json'
    previous=json.loads(manifest.read_text())['files'] if manifest.exists() else []
    records={r['path']:r for r in previous}
    for row in release['files']:
        name=row['bundle_path']
        if not name.startswith(dataset+'/'):continue
        if '/models/' in name and not name.startswith(f'{dataset}/models/{variant}/'):continue
        cached=Path(hf_hub_download(REPO_ID,row['path'],revision=revision))
        if cached.stat().st_size!=row['bytes'] or file_sha256(cached)!=row['sha256']:
            raise ValueError('Release asset hash mismatch: '+row['path'])
        target=safe_path(root,name);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists() or file_sha256(target)!=row['sha256']:shutil.copyfile(cached,target)
        records[name]={'path':name,'bytes':row['bytes'],'sha256':row['sha256']}
    manifest.write_text(json.dumps({'schema_version':1,'files':list(records.values())},indent=2)+'\n')
    example=None;reference=None
    if example_index is not None:
        for row in release['examples']:
            if row['dataset']!=dataset or row['example_index']!=example_index:continue
            path=Path(hf_hub_download(REPO_ID,row['path'],revision=revision))
            if file_sha256(path)!=row['sha256']:raise ValueError('Example checksum mismatch')
            if row.get('kind')=='saved_prediction_reference':reference=path
            else:example=path
        if example is None:raise ValueError('No example with this index')
    return root,example,reference
