
import os
import sys
from pathlib import Path
import shutil

# Add DeOldify to path
deoldify_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'DeOldify'))
sys.path.append(deoldify_path)

# Create a wrapper for DeOldify's image colorizer with error handling
def get_safe_colorizer(render_factor=35, artistic=True):
    """
    Get a wrapper around DeOldify's colorizer that handles path errors
    """
    try:
        from deoldify.visualize import get_image_colorizer
        colorizer = get_image_colorizer(artistic=artistic)
        
        # Create the original transform_image method
        original_transform = colorizer.plot_transformed_image
        
        # Create a safe wrapper
        def safe_transform(source_path, render_factor=render_factor, display_render_factor=False, results_dir=None):
            try:
                # Use the original method
                return original_transform(
                    source_path, 
                    render_factor=render_factor, 
                    display_render_factor=display_render_factor,
                    results_dir=results_dir
                )
            except TypeError as e:
                if "unsupported operand type(s) for /" in str(e):
                    print("Handling Path division error...")
                    
                    # Convert paths to strings and join manually
                    source_path_str = str(source_path)
                    if results_dir is None:
                        results_dir_str = str(Path("./result_images"))
                    else:
                        results_dir_str = str(results_dir)
                    
                    # Create the result path manually
                    base_name = os.path.basename(source_path_str)
                    result_path = os.path.join(results_dir_str, f"result_{base_name}")
                    
                    # Call the model directly
                    from PIL import Image
                    import numpy as np
                    
                    # Read the image
                    img = np.array(Image.open(source_path_str))
                    
                    # Apply transformation directly
                    colorizer.model.eval()
                    result = colorizer._transform_img(img, render_factor)
                    
                    # Save the result
                    result_img = Image.fromarray(result)
                    result_img.save(result_path)
                    
                    return result_path
                else:
                    raise
            
        # Replace the original method with the safe wrapper
        colorizer.plot_transformed_image = safe_transform
        
        return colorizer
    except Exception as e:
        print(f"Error creating safe colorizer: {e}")
        raise
