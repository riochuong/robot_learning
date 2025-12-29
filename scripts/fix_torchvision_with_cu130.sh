#!/bin/bash

# Stop on any error
set -e

echo "=== 1. Uninstalling mismatched Torchvision ==="
uv pip uninstall torchvision

echo "=== 2. NUKE THE CACHE (Crucial Step) ==="
# This stops uv from finding the old "bad" wheel
uv cache clean

echo "=== 3. Rebuilding Torchvision from Source ==="
echo "Note: This will take a few minutes. Please wait..."
# --no-build-isolation: Uses the PyTorch Nightly ALREADY in your venv
# --reinstall: Forces the install even if packages look similar
uv pip install --no-build-isolation --reinstall git+https://github.com/pytorch/vision.git

echo "=== 4. Verification ==="
python -c "
import torch
import torchvision
print(f'\nSUCCESS!')
print(f'Torch Version: {torch.__version__}')
print(f'Torch CUDA:    {torch.version.cuda}')
print(f'Vision Version:{torchvision.__version__}')
print(f'Vision CUDA:   {torchvision.version.cuda}')

if torch.version.cuda == torchvision.version.cuda:
    print('\n✅ VERSIONS MATCH! You are ready to train.')
else:
    print('\n❌ MISMATCH DETECTED. Something went wrong.')
    exit(1)
"
