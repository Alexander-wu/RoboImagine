"""Verify published source files without importing or executing experiment code."""
from pathlib import Path
import ast, hashlib, json
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
for record in manifest['files']:
    path = root / record['file']
    assert path.is_relative_to(root) and path.is_file(), path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['published_sha256'], path
    if path.suffix == '.py':
        ast.parse(path.read_text(), filename=str(path))
print(f"PASS: {len(manifest['files'])} archived files; hashes and Python syntax only. No experiments executed.")
