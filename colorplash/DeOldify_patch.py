import os
import sys
import shutil
import re
from pathlib import Path

def patch_deoldify():
    """Patch the DeOldify library to fix all instances of path division errors"""
    
    # Path to the DeOldify directory
    deoldify_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'DeOldify'))
    
    # Check if DeOldify exists
    if not os.path.exists(deoldify_dir):
        print(f"Error: DeOldify directory not found at {deoldify_dir}")
        return False
    
    # Path to visualize.py file
    visualize_path = os.path.join(deoldify_dir, 'deoldify', 'visualize.py')
    
    # Check if visualize.py exists
    if not os.path.exists(visualize_path):
        print(f"Error: visualize.py not found at {visualize_path}")
        return False
    
    # Back up the original file
    backup_path = visualize_path + '.bak'
    if not os.path.exists(backup_path):
        print(f"Creating backup of visualize.py at {backup_path}")
        shutil.copy2(visualize_path, backup_path)
    
    print(f"Patching DeOldify visualize.py to fix ALL Path division errors...")
    
    # Read the file
    with open(visualize_path, 'r') as f:
        content = f.read()
    
    # Add import os if not already present
    if "import os" not in content:
        content = content.replace(
            "from pathlib import Path",
            "from pathlib import Path\nimport os"
        )
    
    # Replace all instances of Path division with os.path.join
    
    # Pattern 1: result_path = results_dir / source_path.name
    content = content.replace(
        "result_path = results_dir / source_path.name",
        "result_path = Path(os.path.join(str(results_dir), str(source_path.name)))"
    )
    
    # Pattern 2: path_name.parent / name
    content = re.sub(
        r'([a-zA-Z_][a-zA-Z0-9_]*\.parent)\s*/\s*([a-zA-Z_][a-zA-Z0-9_]*)',
        r'Path(os.path.join(str(\1), str(\2)))',
        content
    )
    
    # Pattern 3: workfolder / "source"
    content = re.sub(
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*/\s*("[^"]+")',
        r'Path(os.path.join(str(\1), \2))',
        content
    )
    
    # Pattern 4: results_dir / source_path
    content = re.sub(
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*/\s*([a-zA-Z_][a-zA-Z0-9_]*)',
        r'Path(os.path.join(str(\1), str(\2)))',
        content
    )
    
    # Write the modified content back
    with open(visualize_path, 'w') as f:
        f.write(content)
    
    print("DeOldify patched successfully with all path division fixes!")
    
    # Also create a wrapper for get_image_colorizer
    create_colorizer_wrapper()
    
    return True

def create_colorizer_wrapper():
    """Create a wrapper for get_image_colorizer to handle Path division errors"""
    wrapper_path = os.path.join(os.path.dirname(__file__), 'colorizer_wrapper.py')
    
    wrapper_content = '''
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
'''
    
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    
    print(f"Created colorizer_wrapper.py with safe DeOldify wrapper")

if __name__ == "__main__":
    patch_deoldify() 