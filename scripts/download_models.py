"""Download flan-t5-base and nli-deberta-v3-small into the HF cache (resumable).

Only the files the run needs (PyTorch weights + tokenizer/config) are
fetched; the TF/Flax/ONNX/safetensors-duplicate weights are skipped via
allow_patterns so we do not pull gigabytes of junk on a slow link.
"""

import os
import re

for var in ("no_proxy", "NO_PROXY"):
    v = os.environ.get(var)
    if v:
        os.environ[var] = re.sub(r"\[([0-9a-fA-F:]+)\]", r"\1", v)

from huggingface_hub import snapshot_download

MODELS = [
    (
        "google/flan-t5-base",
        [
            "pytorch_model.bin",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "spiece.model",
        ],
    ),
    (
        "cross-encoder/nli-deberta-v3-small",
        [
            "pytorch_model.bin",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "spm.model",
        ],
    ),
]

for model, patterns in MODELS:
    print(f"downloading {model}", flush=True)
    path = snapshot_download(repo_id=model, allow_patterns=patterns)
    print(f"done: {path}", flush=True)
print("ALL DOWNLOADS DONE")
