import torch

def check_cuda_availability():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("CUDA is available. GPU: " + torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    check_cuda_availability()