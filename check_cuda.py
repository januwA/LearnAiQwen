import torch; print(f'PyTorch版本: {torch.__version__}'); 
print(f'CUDA可用: {torch.cuda.is_available()}'); 
print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else "N/A"}')
if torch.cuda.is_available():
    print(f"显卡型号：{torch.cuda.get_device_name(0)}")
    print(f"显存总量：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
