import os
import sys
import traceback

def check_environment():
    """Check that the required directories and model files are present"""
    print("Checking ChromaVision environment setup...")
    
    # Check working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Working directory: {current_dir}")
    
    # Check for models directory
    models_dir = os.path.join(current_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory: {models_dir}")
    else:
        print(f"Models directory exists: {models_dir}")
    
    # Check for model file
    model_path = os.path.join(models_dir, 'ColorizeArtistic_gen.pth')
    if not os.path.exists(model_path):
        print(f"WARNING: Model file not found at {model_path}")
        print("Please download ColorizeArtistic_gen.pth from DeOldify repository")
        print("Place it in the models directory.")
    else:
        print(f"Model file found: {model_path} ({os.path.getsize(model_path) / (1024 * 1024):.2f} MB)")
    
    # Check for DeOldify directory
    deoldify_dir = os.path.join(current_dir, 'DeOldify')
    if not os.path.exists(deoldify_dir):
        print(f"WARNING: DeOldify directory not found at {deoldify_dir}")
        print("Please clone the DeOldify repository into your project directory")
    else:
        print(f"DeOldify directory found: {deoldify_dir}")
    
    # Check for required directories
    for directory in ['static/uploads', 'static/results']:
        dir_path = os.path.join(current_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        else:
            print(f"Directory exists: {dir_path}")
    
    # Check for Python packages
    print("\nChecking required Python packages:")
    packages = ['numpy', 'torch', 'torchvision', 'Pillow', 'opencv-python', 'fastai', 'flask']
    
    for package in packages:
        try:
            if package == 'Pillow':
                import PIL
                print(f"✓ PIL (Pillow) {PIL.__version__}")
            elif package == 'opencv-python':
                import cv2
                print(f"✓ OpenCV {cv2.__version__}")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                print(f"✓ {package} {version}")
        except ImportError:
            print(f"✗ {package} not found. Please install it using pip install {package}")
    
    # Try to import DeOldify modules
    print("\nChecking DeOldify modules:")
    try:
        sys.path.append(deoldify_dir)
        from deoldify import device
        from deoldify.device_id import DeviceId
        from deoldify.visualize import get_image_colorizer
        
        print("✓ DeOldify modules imported successfully")
        
        # Check CUDA
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                print(f"✓ CUDA is available! Found {device_count} device(s). Using: {device_name}")
            else:
                print("✗ CUDA is not available. Using CPU instead (colorization will be slower)")
        except Exception as e:
            print(f"✗ Error checking CUDA: {e}")
            
    except Exception as e:
        print(f"✗ Error importing DeOldify modules: {e}")
        traceback.print_exc()
    
    print("\nEnvironment check complete.")

if __name__ == "__main__":
    check_environment() 