"""Print an environment report (Python, torch, CUDA, GPU memory, bf16, HF libs)."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffjev.utils import environment_report, project_paths  # noqa: E402

if __name__ == "__main__":
    print(json.dumps(environment_report(), indent=1))
    print(json.dumps({k: str(v) for k, v in project_paths().items()}, indent=1))

