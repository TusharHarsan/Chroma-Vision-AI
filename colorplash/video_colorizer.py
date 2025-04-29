import sys
import os
import cv2
import shutil
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import threading

# Verify numpy is available
try:
    import numpy as np
    print(f"NumPy is available. Version: {np.__version__}")
    # Check version compatibility
    version = [int(x) for x in np.__version__.split('.')]
    if version[0] > 1:
        print("Warning: NumPy version 2+ detected. Downgrading to version 1.26.4 for compatibility...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4"])
        # Reload numpy after downgrade
        import importlib
        importlib.reload(np)
        print(f"NumPy downgraded. Version: {np.__version__}")
except ImportError:
    print("Error: NumPy is not available. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4"])
    import numpy as np
    print(f"NumPy installed. Version: {np.__version__}")

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

def colorize_video(source_path, output_path, render_factor=35, fps=None, update_progress_callback=None):
    """
    Colorizes a video file using the DeOldify model.
    
    Parameters:
    - source_path: Path to the source video to be colorized
    - output_path: Path to save the colorized video
    - render_factor: Quality factor for the colorization (higher is better but slower)
    - fps: Frames per second for the output video (None for original)
    - update_progress_callback: Optional callback function to update progress (task_id, progress_percentage)
    
    Returns:
    - output_path: Path to the colorized video
    """
    try:
        # Check if Model file exists
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'ColorizeArtistic_gen.pth')
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found at {model_path}. Please download the ColorizeArtistic_gen.pth file.")
        
        # Make sure numpy is properly imported
        try:
            import numpy as np
            print(f"NumPy version: {np.__version__}")
        except ImportError:
            raise Exception("NumPy is required but not available.")
        
        # Check for CUDA availability
        is_cuda_available = check_cuda_available()
        
        # Add DeOldify to path
        deoldify_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'DeOldify'))
        if not os.path.exists(deoldify_path):
            raise Exception(f"DeOldify directory not found at {deoldify_path}. Please check your installation.")
        
        sys.path.append(deoldify_path)
        
        # Get task_id from the current thread name if it contains 'task_'
        task_id = None
        thread_name = threading.current_thread().name
        if 'task_' in thread_name:
            task_id = thread_name.split('task_')[1]
        
        # Report progress
        def report_progress(percentage):
            if update_progress_callback and task_id:
                update_progress_callback(task_id, percentage)
                
        report_progress(5)  # Initial progress
        
        # Try to use the safe colorizer wrapper first
        try:
            print("Using safe colorizer wrapper...")
            from colorizer_wrapper import get_safe_colorizer
            colorizer = get_safe_colorizer(render_factor=render_factor, artistic=True)
            print("Safe colorizer created successfully")
        except Exception as e:
            print(f"Error using safe colorizer: {e}, falling back to standard import...")
            # Import DeOldify modules directly if wrapper fails
            try:
                print("Importing DeOldify modules...")
                from deoldify import device
                from deoldify.device_id import DeviceId
                from deoldify.visualize import get_image_colorizer
                print("DeOldify modules imported successfully")
                
                # Set to GPU if available, otherwise CPU
                if is_cuda_available:
                    print("Setting DeOldify to use GPU...")
                    device.set(device=DeviceId.GPU0)
                else:
                    print("Setting DeOldify to use CPU (colorization will be slower)...")
                    device.set(device=DeviceId.CPU)
                    
                # Get colorizer
                colorizer = get_image_colorizer(artistic=True)
            except Exception as e:
                raise Exception(f"Failed to import DeOldify modules and create colorizer: {e}")
        
        # Create temporary directories for frame processing
        temp_dir = os.path.join(os.path.dirname(output_path), 'temp_frames')
        os.makedirs(temp_dir, exist_ok=True)
        
        colorized_frames_dir = os.path.join(os.path.dirname(output_path), 'colorized_frames')
        os.makedirs(colorized_frames_dir, exist_ok=True)
        
        # Open the video file
        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {source_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = float(cap.get(cv2.CAP_PROP_FPS))
        
        # Convert fps to float or use original
        if fps is None:
            fps = original_fps
        else:
            try:
                fps = float(fps)
            except (ValueError, TypeError):
                fps = original_fps
                print(f"Invalid fps value, using original: {original_fps}")
        
        # Make sure fps is valid
        if fps <= 0:
            fps = 30.0
            print(f"Invalid fps value, using default: {fps}")
        
        # Configure PyTorch to use CUDA if available
        if is_cuda_available:
            try:
                import torch
                # Set PyTorch device
                torch.cuda.set_device(0)
                # Print CUDA memory info
                print(f"CUDA Memory: Total {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                free_memory, total_memory = torch.cuda.mem_get_info()
                print(f"CUDA Memory Available: {free_memory / 1e9:.2f} GB / {total_memory / 1e9:.2f} GB")
            except Exception as e:
                print(f"Error configuring CUDA details: {e}")
        
        print("Model initialized successfully")
        
        try:
            # Extract frames
            frames = []
            frame_count = 0
            print(f"Extracting frames from video ({total_frames} total)...")
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frames.append(frame)
                    frame_count += 1
                    
                    if frame_count % 10 == 0:
                        print(f"Extracted {frame_count} frames...")
                        # Clear CUDA cache periodically during extraction
                        if is_cuda_available:
                            try:
                                import torch
                                torch.cuda.empty_cache()
                            except Exception as e:
                                print(f"Warning: Could not clear CUDA cache during extraction: {e}")
                    
                    # Optional limit: process only first N frames for large videos
                    # Remove this limit if you want to process entire videos
                    if total_frames > 1000 and frame_count >= 300:
                        print(f"WARNING: Video has {total_frames} frames, limiting to first 300 for memory safety")
                        break
            except Exception as e:
                print(f"Error during frame extraction: {e}")
                import traceback
                traceback.print_exc()
                if frame_count > 0:
                    print(f"Will continue with {frame_count} extracted frames")
                else:
                    raise Exception(f"Failed to extract any frames: {e}")
            
            print(f"Extracted {frame_count} frames in total")
            
            # Release the video capture
            cap.release()
            
            # Process frames in batches
            colorized_frames = []
            total_frames = len(frames)
            for i, frame in enumerate(tqdm(frames, desc="Colorizing frames")):
                # Calculate and report progress (video extraction = 20%, colorization = 60%, encoding = 20%)
                # So here we go from 20% to 80% of the overall process
                progress_percentage = 20 + int((i / total_frames) * 60)
                report_progress(progress_percentage)
                
                # Create grayscale image for DeOldify
                # DeOldify expects RGB image input, but needs grayscale content
                try:
                    # Convert the original frame to RGB from BGR (OpenCV uses BGR)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Save as RGB image first (DeOldify expects RGB input)
                    pil_img = Image.fromarray(rgb_frame)
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
                    pil_img.save(frame_path)
                    
                    # Now create a grayscale version using PIL
                    with Image.open(frame_path) as img:
                        # Convert to grayscale and then to RGB mode
                        # (DeOldify expects RGB input even for grayscale content)
                        gray_img = ImageOps.grayscale(img).convert('RGB')
                        gray_path = os.path.join(temp_dir, f"gray_{i:04d}.jpg")
                        gray_img.save(gray_path)
                        print(f"Created grayscale image for frame {i+1}")
                        
                except Exception as e:
                    print(f"Error in PIL conversion: {e}, trying OpenCV fallback")
                    # Fallback to OpenCV
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_3channel = cv2.merge([gray, gray, gray])
                    gray_path = os.path.join(temp_dir, f"gray_{i:04d}.jpg")
                    cv2.imwrite(gray_path, gray_3channel)
                
                # Verify grayscale image exists and has content
                if not os.path.exists(gray_path) or os.path.getsize(gray_path) < 1000:  # 1KB minimum
                    print(f"Warning: Grayscale image invalid for frame {i}, using direct save of original")
                    # Just save the original frame as grayscale
                    cv2.imwrite(gray_path, frame)
                
                # SAFE COLORIZATION USING DIRECT MODEL ACCESS
                # Skip the path manipulation issues by directly using numpy arrays
                try:
                    print(f"Colorizing frame {i+1}/{len(frames)} using direct method")
                    
                    # Read the grayscale image into a numpy array
                    gray_img = np.array(Image.open(gray_path))
                    
                    # Direct colorization without saving to file first
                    colorizer.model.eval()
                    result = colorizer._transform_img(gray_img, render_factor)
                    
                    # Convert result to BGR for OpenCV
                    if result.shape[2] == 3:  # Make sure it's RGB
                        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                        colorized_frames.append(result_bgr)
                        
                        # Also save to a file for debugging
                        result_path = os.path.join(colorized_frames_dir, f"result_{i:04d}.jpg")
                        cv2.imwrite(result_path, result_bgr)
                        print(f"Successfully colorized frame {i+1} using direct method")
                    else:
                        print(f"Error: Unexpected colorized result shape: {result.shape}")
                        colorized_frames.append(frame)  # Use original as fallback
                    
                except Exception as e:
                    print(f"Error in direct colorization: {e}, trying standard method")
                    
                    try:
                        # Use standard DeOldify method with error handling
                        result_path = colorizer.plot_transformed_image(
                            gray_path,
                            render_factor=render_factor,
                            display_render_factor=False,
                            results_dir=colorized_frames_dir
                        )
                        
                        # Verify the result exists
                        if os.path.exists(result_path):
                            colorized_frame = cv2.imread(result_path)
                            if colorized_frame is not None:
                                colorized_frames.append(colorized_frame)
                                print(f"Successfully colorized frame {i+1} using standard method")
                            else:
                                print(f"Error reading colorized result, using original frame")
                                colorized_frames.append(frame)
                        else:
                            print(f"No result path from colorizer, using original frame")
                            colorized_frames.append(frame)
                            
                    except Exception as e:
                        print(f"Error in standard colorization: {e}, using original frame")
                        colorized_frames.append(frame)
                
                # Optionally clear CUDA cache every few frames to prevent memory buildup
                if is_cuda_available and i % 5 == 0:
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Error clearing CUDA cache: {e}")
                
                # Clean up temporary files
                try:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                    if os.path.exists(gray_path):
                        os.remove(gray_path)
                except Exception as e:
                    print(f"Error cleaning temporary files: {e}")
                    pass
            
            # Create an output video file
            print(f"Creating video with {len(colorized_frames)} frames at {fps} FPS")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Ensure height and width are valid
            if height <= 0 or width <= 0:
                # Use the size of the first frame
                if len(colorized_frames) > 0:
                    height, width, _ = colorized_frames[0].shape
                else:
                    height, width = 480, 640
                print(f"Invalid dimensions, using {width}x{height}")
            
            # Use a different codec that's more reliable
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            
            # Create temporary path for intermediate file
            temp_output = os.path.join(os.path.dirname(output_path), 'temp_output.mp4')
            
            try:
                out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    # Fallback to XVID codec if H.264 failed
                    print("Failed to open VideoWriter with H.264 codec, trying XVID...")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    # Last resort - use MJPG
                    print("Failed to open VideoWriter with XVID codec, trying MJPG...")
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
                    
                if not out.isOpened():
                    raise Exception("Could not open VideoWriter with any codec")
                
                # Write frames to video
                for j, frame in enumerate(tqdm(colorized_frames, desc="Writing video")):
                    # Calculate and report final progress (80-100%)
                    progress_percentage = 80 + int((j / len(colorized_frames)) * 20)
                    report_progress(progress_percentage)
                    
                    # Ensure frame is the correct size
                    if frame.shape[0] != height or frame.shape[1] != width:
                        frame = cv2.resize(frame, (width, height))
                    out.write(frame)
                
                # Release resources
                out.release()
                
                # Copy to final destination
                shutil.copy2(temp_output, output_path)
                print(f"Video saved to {output_path}")
                
            except Exception as e:
                print(f"Error writing video: {e}")
                raise
            finally:
                # Remove temp file
                if os.path.exists(temp_output):
                    os.remove(temp_output)
            
        finally:
            # Release resources and clean up
            if 'cap' in locals() and cap is not None:
                cap.release()
            
            # Clean up temp directories
            for directory in [temp_dir, colorized_frames_dir]:
                if os.path.exists(directory):
                    try:
                        shutil.rmtree(directory)
                    except Exception as e:
                        print(f"Warning: Could not clean up directory {directory}: {e}")
        
        return output_path
    
    except Exception as e:
        print(f"Error in colorize_video: {e}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Failed to process video: {e}") 