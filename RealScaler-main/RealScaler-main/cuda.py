import torch
import sys
import platform
import os
import subprocess

def check_cuda():
    """
    Check if CUDA is available and print system information
    """
    print("=" * 50)
    print("CUDA Availability Check for RealScaler")
    print("=" * 50)
    
    # System information
    print(f"Python version: {platform.python_version()}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    
    # PyTorch version and CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print("\n✅ CUDA is available!")
        print(f"CUDA version: {torch.version.cuda}")
        
        # Get device count and information
        device_count = torch.cuda.device_count()
        print(f"Number of available GPU devices: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            print(f"\nGPU {i}: {device_name}")
            print(f"Compute capability: {device_capability[0]}.{device_capability[1]}")
            
            # Get device properties
            props = torch.cuda.get_device_properties(i)
            print(f"Total memory: {props.total_memory / 1024 / 1024 / 1024:.2f} GB")
            print(f"CUDA cores: {props.multi_processor_count}")
    else:
        print("\n❌ CUDA is not available.")
        print("Possible reasons:")
        print("1. NVIDIA GPU drivers are not installed or outdated")
        print("2. PyTorch was installed without CUDA support")
        print("3. Your GPU does not support CUDA")
        
        # Check if NVIDIA GPU is present
        try:
            if platform.system() == "Windows":
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                if result.returncode == 0:
                    print("\nNVIDIA GPU detected but CUDA is not available in PyTorch.")
                    print("Try reinstalling PyTorch with CUDA support:")
                    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                else:
                    print("\nNo NVIDIA GPU detected or drivers not installed.")
        except FileNotFoundError:
            print("\nNVIDIA System Management Interface (nvidia-smi) not found.")
            print("This suggests NVIDIA drivers are not installed.")
    
    print("\n" + "=" * 50)
    print("DirectML Availability Check")
    print("=" * 50)
    
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        
        if 'DmlExecutionProvider' in providers:
            print("✅ DirectML is available for ONNX Runtime")
            print(f"Available providers: {providers}")
        else:
            print("❌ DirectML provider not found in ONNX Runtime")
            print(f"Available providers: {providers}")
            print("\nTo install ONNX Runtime with DirectML support:")
            print("pip install onnxruntime-directml")
    except ImportError:
        print("❌ ONNX Runtime is not installed")
        print("\nTo install ONNX Runtime with DirectML support:")
        print("pip install onnxruntime-directml")

if __name__ == "__main__":
    check_cuda()
    
    # Keep console window open if run directly
    if len(sys.argv) == 1:
        input("\nPress Enter to exit...")