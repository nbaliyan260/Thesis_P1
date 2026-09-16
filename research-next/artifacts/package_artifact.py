"""Package only research evidence/code, never environments, sockets or credentials."""
from pathlib import Path
import hashlib, json, zipfile
ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT/'research-next'
PDF_NAMES = ['RECUT_01_Idea_and_Novelty.pdf','RECUT_02_Implementation_Plan.pdf',
             'RECUT_03_Professor_Draft_and_Results.pdf']
files=[]
for folder in ('experiments','literature','notes','runs','logs','artifacts'):
    for p in (PROJECT/folder).rglob('*'):
        if not p.is_file(): continue
        relative = p.relative_to(PROJECT)
        if any(part.startswith('.') or part=='__pycache__' for part in relative.parts): continue
        if 'qa' in relative.parts: continue
        if p.name in ('artifact_manifest.json','RECUT_Research_Package.zip'): continue
        if p.suffix not in ('.py','.json','.jsonl','.md','.txt','.out','.err','.sbatch','.png','.pt','.xml'): continue
        files.append(p)
files.append(PROJECT/'README.md')
files.append(ROOT/'Research Papers PDFs/download_manifest.csv')
for name in PDF_NAMES:
    p=ROOT/'output/pdf'/name
    assert p.is_file(),p
    files.append(p)
assert all(p.is_file() for p in files)
records=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(files))]
manifest=PROJECT/'artifacts/artifact_manifest.json'
manifest.write_text(json.dumps(dict(scope='RECUT bounded research pilot; no model weights/environments/secrets',
    records=records),indent=2)+'\n')
output=ROOT/'output/RECUT_Research_Package.zip'
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for p in sorted(set(files+[manifest])):
        archive.write(p,str(p.relative_to(ROOT)))
print(json.dumps(dict(output=str(output),files=len(records)+1,bytes=output.stat().st_size,
    sha256=hashlib.sha256(output.read_bytes()).hexdigest()),indent=2))
