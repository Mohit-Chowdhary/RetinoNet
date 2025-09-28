import torch
print(torch.version.cuda)      # Shows CUDA version PyTorch was built with
print(torch.cuda.is_available())  # Checks if GPU is accessible
print(torch.cuda.get_device_name(0))  # Shows your GPU name
