import sys
import subprocess

def install_packages():
    print("==========================================")
    print("Installing required libraries...")
    print("==========================================\n")

    # sys.executable stellt sicher, dass das exakt richtige Python genutzt wird
    python_exe = sys.executable

    print("Upgrading pip...")
    subprocess.check_call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
    print("\n")

    packages = [
        "anytree",
        "hydra-core",
        "shapely",
        "matplotlib",
        "casadi",
        "snakeviz",
        "bokeh"
    ]

    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([python_exe, "-m", "pip", "install", package])
        print("\n")

    print("==========================================")
    print("Installation successfully completed!")
    print("==========================================")

if __name__ == "__main__":
    install_packages()
