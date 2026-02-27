import torch; print(f'PyTorch版本: {torch.__version__}'); 
print(f'CUDA可用: {torch.cuda.is_available()}'); 
print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else "N/A"}')
