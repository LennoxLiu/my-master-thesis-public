import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch CUDA Build: {torch.version.cuda}")
    
    # Simple tensor test to ensure memory allocation works
    x = torch.rand(5, 3).cuda()
    print("Tensor successfully moved to GPU.")
else:
    print("CRITICAL: CUDA is not available to PyTorch.")