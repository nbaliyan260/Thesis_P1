"""Reproduce original public model revisions, never silently resolve a new HEAD."""
from pathlib import Path
import argparse, json
from huggingface_hub import snapshot_download
parser=argparse.ArgumentParser()
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists(): raise RuntimeError('Refuse to overwrite an existing manifest')
records=json.loads(args.manifest.read_text())
for record in records:
    revision=record['revision']
    if len(revision)!=40 or any(x not in '0123456789abcdef' for x in revision):
        raise ValueError('Require immutable commit revision')
    record['local_path']=snapshot_download(record['model'],revision=revision,
        allow_patterns=['*.json','*.safetensors','merges.txt','vocab.json','tokenizer.model'])
args.output.write_text(json.dumps(records,indent=2)+'\n')
print(args.output)
