"""Download only prespecified immutable public model revisions on a CPU job."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise RuntimeError("Refuse to replace an existing download manifest")
records = json.loads(args.manifest.read_text())
for record in records:
    revision = record["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Require immutable commit revision")
    actual = snapshot_download(record["model"], revision=revision,
        allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "tokenizer.model"],
        max_workers=4)
    if Path(actual).resolve() != Path(record["local_path"]).resolve():
        raise RuntimeError("Downloaded path differs from fixed study manifest")
    print(json.dumps({"model": record["model"], "revision": revision, "path": actual}), flush=True)
with args.output.open("x") as stream:
    json.dump(records, stream, indent=2)
    stream.write("\n")
