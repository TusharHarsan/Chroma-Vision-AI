import os
import sys
import shutil
import re
import site
from pathlib import Path

def find_visualize_path():
    """Find the visualize.py file in site-packages"""
    # Get site-packages directories
    site_packages = site.getsitepackages()
    
    for sp in site_packages:
        visualize_path = os.path.join(sp, 'deoldify', 'visualize.py')
        if os.path.exists(visualize_path):
            return visualize_path
    
    # Try user site-packages
    user_site = site.getusersitepackages()
    visualize_path = os.path.join(user_site, 'deoldify', 'visualize.py')
    if os.path.exists(visualize_path):
        return visualize_path
    
    # Try anaconda environment
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_site_path = os.path.join(conda_prefix, 'Lib', 'site-packages', 'deoldify', 'visualize.py')
        if os.path.exists(conda_site_path):
            return conda_site_path
    
    # Manual check of standard anaconda locations
    anaconda_path = os.path.expanduser('~/anaconda3/Lib/site-packages/deoldify/visualize.py')
    if os.path.exists(anaconda_path):
        return anaconda_path
    
    # Check explicitly for the path shown in the error
    explicit_path = r"C:\Users\savit\anaconda3\Lib\site-packages\deoldify\visualize.py"
    if os.path.exists(explicit_path):
        return explicit_path
    
    return None

def patch_site_packages():
    """Patch the visualize.py file in site-packages"""
    visualize_path = find_visualize_path()
    
    if not visualize_path:
        print("Could not find visualize.py in site-packages.")
        return False
    
    print(f"Found visualize.py at: {visualize_path}")
    
    # Back up the original file
    backup_path = visualize_path + '.bak'
    if not os.path.exists(backup_path):
        print(f"Creating backup of visualize.py at {backup_path}")
        shutil.copy2(visualize_path, backup_path)
    
    print(f"Patching site-packages visualize.py to fix Path division error...")
    
    # Read the file
    with open(visualize_path, 'r') as f:
        content = f.read()
    
    # Fix the _save_result_image method
    if "result_path = results_dir / source_path.name" in content:
        # This is the problematic code using Path division
        modified_content = content.replace(
            "result_path = results_dir / source_path.name",
            "result_path = Path(os.path.join(str(results_dir), str(source_path.name)))"
        )
        
        # Make sure os is imported
        if "import os" not in modified_content:
            modified_content = modified_content.replace(
                "from pathlib import Path",
                "from pathlib import Path\nimport os"
            )
        
        # Write the modified content back
        with open(visualize_path, 'w') as f:
            f.write(modified_content)
        
        print("Patched successfully!")
        return True
    else:
        print("Could not find the problematic code pattern in visualize.py.")
        
        # Try to insert the modified _save_result_image method
        new_method = '''
    def _save_result_image(self, source_path: Path, image: Image, results_dir = None) -> Path:
        if results_dir is None:
            results_dir = Path(self.results_dir)
        else:
            results_dir = Path(results_dir)
            
        source_path = Path(source_path)
        # Fix for division error - use os.path.join instead of / operator
        result_path = Path(os.path.join(str(results_dir), str(source_path.name)))
        image.save(result_path)
        return result_path
'''
        
        # Look for the method definition
        pattern = r'def _save_result_image.*?return result_path'
        if re.search(pattern, content, re.DOTALL):
            modified_content = re.sub(pattern, new_method.strip(), content, flags=re.DOTALL)
            
            # Make sure os is imported
            if "import os" not in modified_content:
                modified_content = modified_content.replace(
                    "from pathlib import Path",
                    "from pathlib import Path\nimport os"
                )
            
            # Write the modified content back
            with open(visualize_path, 'w') as f:
                f.write(modified_content)
            
            print("Patched _save_result_image method with replacement.")
            return True
        
        return False

if __name__ == "__main__":
    success = patch_site_packages()
    if success:
        print("DeOldify site-packages patch applied successfully!")
    else:
        print("Failed to apply patch. Please check the script and try again.") 