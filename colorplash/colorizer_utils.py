import sys
import os
import shutil
import numpy as np
from PIL import Image, ImageOps

def check_cuda_available():
    """Check if CUDA is available for PyTorch"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"CUDA is available! Found {device_count} device(s). Using: {device_name}")
            return True
        else:
            print("CUDA is not available. Using CPU instead.")
            return False
    except Exception as e:
        print(f"Error checking CUDA: {e}")
        return False

# Function adapted from DeOldify colorizer
def colorize_image(source_path, results_path, render_factor=35):
    """
    Colorizes a grayscale image using the DeOldify model.
    
    Parameters:
    - source_path: Path to the source image to be colorized
    - results_path: Directory to save the results
    - render_factor: Quality factor for the colorization (higher is better but slower)
    
    Returns:
    - grayscale_path: Path to the grayscale version of the image
    - colorized_path: Path to the colorized output image
    """
    try:
        # Check if Model file exists
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'ColorizeArtistic_gen.pth')
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found at {model_path}. Please download the ColorizeArtistic_gen.pth file.")
            
        # Check for CUDA availability
        is_cuda_available = check_cuda_available()
        
        # Add DeOldify to path
        deoldify_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'DeOldify'))
        if not os.path.exists(deoldify_path):
            raise Exception(f"DeOldify directory not found at {deoldify_path}. Please check your installation.")
            
        sys.path.append(deoldify_path)
        
        print("Importing DeOldify modules...")
        from deoldify import device
        from deoldify.device_id import DeviceId
        from deoldify.visualize import get_image_colorizer
        print("DeOldify modules imported successfully")
        
        # Set GPU/CPU based on availability
        if is_cuda_available:
            print("Setting DeOldify to use GPU for image colorization...")
            device.set(device=DeviceId.GPU0)
            
            # Configure PyTorch for CUDA
            try:
                import torch
                torch.cuda.set_device(0)
                print(f"CUDA Memory: Total {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                free_memory, total_memory = torch.cuda.mem_get_info()
                print(f"CUDA Memory Available: {free_memory / 1e9:.2f} GB / {total_memory / 1e9:.2f} GB")
            except Exception as e:
                print(f"Error configuring CUDA details: {e}")
        else:
            print("Setting DeOldify to use CPU (colorization will be slower)...")
            device.set(device=DeviceId.CPU)
        
        # Create RGB grayscale image that DeOldify expects
        try:
            print(f"Opening image: {source_path}")
            # Load image and convert to RGB
            with Image.open(source_path) as img:
                # Make sure it's in RGB mode
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create grayscale version
                grayscale_img = ImageOps.grayscale(img)
                
                # Convert back to RGB (DeOldify expects RGB input)
                grayscale_rgb = grayscale_img.convert('RGB')
                
                # Save the grayscale RGB image
                grayscale_path = os.path.join(results_path, 'grayscale_' + os.path.basename(source_path))
                grayscale_rgb.save(grayscale_path)
                print(f"Saved grayscale image to: {grayscale_path}")
        except Exception as e:
            print(f"Error preparing image: {e}")
            raise
        
        # Use the grayscale image for colorization
        print("Initializing DeOldify colorizer model...")
        colorizer = get_image_colorizer(artistic=True)
        print("Model initialized. Colorizing image...")
        
        # DIRECT METHOD - Bypass the path error completely
        try:
            # Load the image and colorize directly using the model
            gray_img = np.array(Image.open(grayscale_path))
            
            # Apply colorization directly using the model
            colorizer.model.eval()
            result = colorizer._transform_img(gray_img, render_factor)
            
            # Create the output path
            result_filename = 'colorized_' + os.path.basename(source_path)
            new_filename = os.path.join(results_path, result_filename)
            
            # Save the result
            result_img = Image.fromarray(result)
            result_img.save(new_filename)
            
            print(f"Image colorized successfully using direct model access")
            
        except Exception as e:
            print(f"Error in direct colorization: {e}, trying with workaround...")
            
            try:
                # Try using the plot_transformed_image but catch the TypeError
                try:
                    # Get absolute paths
                    grayscale_abs_path = os.path.abspath(grayscale_path)
                    results_abs_path = os.path.abspath(results_path)
                    
                    result_path = colorizer.plot_transformed_image(
                        grayscale_abs_path,
                        render_factor=render_factor,
                        display_render_factor=False,
                        results_dir=results_abs_path
                    )
                    
                except TypeError as te:
                    if "unsupported operand type(s) for /" in str(te):
                        print("Working around path division error...")
                        
                        # Manual workaround for the division error
                        # Create a path that matches what DeOldify would have created
                        result_filename = os.path.basename(grayscale_path)
                        result_path = os.path.join(results_path, result_filename)
                        
                        # Load the grayscale image
                        img = np.array(Image.open(grayscale_path))
                        
                        # Apply the colorization model directly
                        colorizer.model.eval()
                        result_img = colorizer._transform_img(img, render_factor)
                        
                        # Save the result
                        Image.fromarray(result_img).save(result_path)
                    else:
                        raise
                
                # Rename the output file to match our expected naming convention
                filename = os.path.basename(source_path)
                new_filename = os.path.join(results_path, 'colorized_' + filename)
                
                # Delete the file if it already exists
                if os.path.exists(new_filename):
                    os.remove(new_filename)
                
                # Copy the file to the new location with the expected name
                if os.path.exists(result_path):
                    shutil.copy2(result_path, new_filename)
                else:
                    raise Exception(f"Result path not found: {result_path}")
                
            except Exception as e:
                print(f"Error in workaround colorization: {e}")
                raise
        
        # Clear CUDA cache after colorization
        if is_cuda_available:
            try:
                import torch
                torch.cuda.empty_cache()
                print("CUDA cache cleared")
            except Exception as e:
                print(f"Error clearing CUDA cache: {e}")
        
        # Return both the grayscale and colorized paths
        return grayscale_path, new_filename
    except Exception as e:
        print(f"Error in colorize_image: {e}")
        import traceback
        traceback.print_exc()
        raise 