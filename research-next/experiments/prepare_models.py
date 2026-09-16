"""Download public safetensors only and pin immutable model revisions."""
from pathlib import Path
import json, platform, subprocess, importlib.metadata
from huggingface_hub import HfApi, snapshot_download
import torch
root = Path(__file__).resolve().parent
records = []
for model in ('HuggingFaceTB/SmolLM2-135M', 'Qwen/Qwen2.5-0.5B'):
    revision = HfApi().model_info(model).sha
    path = snapshot_download(model, revision=revision,
        allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json', 'tokenizer.model'])
    records.append(dict(model=model, revision=revision, local_path=path))
environment = dict(platform=platform.platform(), python=platform.python_version(),
    torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
    gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
    packages={n:importlib.metadata.version(n) for n in ('torch','transformers','huggingface_hub','numpy')},
    nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True))
(root/'model_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
(root/'environment_setup.json').write_text(json.dumps(environment,indent=2)+'\n')
print(json.dumps(dict(models=records,environment=environment),indent=2),flush=True)
