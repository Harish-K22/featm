"""Check built artifacts without importing the development package."""
import email,hashlib,json,sys,tarfile,tomllib,zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[2]
dist=Path(sys.argv[1]).resolve()
p=tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))['project']
assert p['name']=='featm' and p['version']=='0.5.9'
expected={'featm-0.5.9-py3-none-any.whl','featm-0.5.9.tar.gz'}
assert {f.name for f in dist.iterdir()}==expected
with zipfile.ZipFile(dist/'featm-0.5.9-py3-none-any.whl') as w,tarfile.open(dist/'featm-0.5.9.tar.gz') as t:
 m=email.message_from_string(w.read('featm-0.5.9.dist-info/METADATA').decode('utf-8'))
 s=email.message_from_string(t.extractfile('featm-0.5.9/PKG-INFO').read().decode('utf-8'))
 assert m['Name']==s['Name']=='featm' and m['Version']==s['Version']=='0.5.9'
 assert m.get_all('Requires-Dist')==s.get_all('Requires-Dist')
 assert m['Requires-Python']==s['Requires-Python']==p['requires-python']
 assert m.get_payload().replace('\r\n','\n').strip()==(root/'README.md').read_text(encoding='utf-8').strip()
 licenses=[n for n in w.namelist() if n.endswith('/LICENSE')]
 assert len(licenses)==1 and w.read(licenses[0])==(root/'LICENSE').read_bytes()
 assert t.extractfile('featm-0.5.9/LICENSE').read()==(root/'LICENSE').read_bytes()
 names={x.relative_to(root/'src').as_posix() for x in (root/'src/featm').rglob('*.py')}
 assert len(names)==79 and names=={n for n in w.namelist() if n.startswith('featm/') and n.endswith('.py')}
 for n in names:
  assert w.read(n)==t.extractfile('featm-0.5.9/src/'+n).read()==(root/'src'/n).read_bytes()
print('PASS: version, metadata, README, license and 79 product files')
(root/'release-artifact-hashes.json').write_text(json.dumps({f.name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in sorted(dist.iterdir())},indent=2))
