"""Check that PyTorch can use the GPU and the VLM libraries are installed, before downloading a model.

Prints the versions of torch / transformers / accelerate / bitsandbytes, the GPU name and
memory, then runs a small bf16 matrix multiply and a 4-bit layer on the GPU. RTX 50-series
cards need a PyTorch build for CUDA 12.8 or newer; with an older build the GPU test fails
with "no kernel image is available".

Usage (from the repo root):
    python scripts/check_gpu.py
"""

import importlib
import sys


def version_of(name: str) -> str:
    try:
        return getattr(importlib.import_module(name), "__version__", "installed")
    except Exception as error:  # ImportError, or a library failing while it loads
        return f"NOT AVAILABLE ({type(error).__name__}: {error})"


def main() -> None:
    print(f"python        {sys.version.split()[0]}")
    for name in ["torch", "torchvision", "transformers", "accelerate", "bitsandbytes", "PIL"]:
        print(f"{name:13s} {version_of(name)}")

    import torch

    print(f"\nPyTorch built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible to PyTorch. Install the CUDA build of torch (see requirements-vlm.txt).")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU           {props.name}, {props.total_memory / 2**30:.1f} GB, compute capability {props.major}.{props.minor}")

    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print(f"bf16 matmul   ok (result mean {float((a @ a).float().mean()):.3f})")

    import bitsandbytes as bnb

    layer = bnb.nn.Linear4bit(256, 256, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
    out = layer(torch.randn(4, 256, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    print(f"4-bit layer   ok (output {tuple(out.shape)})")
    print("\nAll checks passed. You can run scripts/run_vlm.py.")


if __name__ == "__main__":
    main()
