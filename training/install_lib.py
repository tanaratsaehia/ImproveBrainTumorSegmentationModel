import subprocess
import sys
import importlib.util

def install_package(package_names):
    """Checks if a package is installed and installs it if not."""
    for package_name in package_names:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            print(f"{package_name} not found. Installing now...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                print(f"{package_name} successfully installed.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package_name}. Error: {e}")
                # sys.exit(1)
        else:
            print(f"{package_name} is already installed.")

# install_package(["nibabel"])
# import nibabel
# print(f"NiBabel version: {nibabel.__version__}")