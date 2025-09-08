#!/usr/bin/env python3
"""
Setup script for the Delight.AI SEDS system.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print installation header."""
    print("\n" + "=" * 50)
    print("  Delight.AI SEDS System Setup")
    print("=" * 50)

def check_python_version():
    """Check if Python version is compatible."""
    required_version = (3, 9)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"❌ Python {required_version[0]}.{required_version[1]}+ is required. "
              f"Current version: {sys.version.split()[0]}")
        sys.exit(1)
    
    print(f"✓ Python {current_version[0]}.{current_version[1]} detected")

def create_virtualenv():
    """Create a virtual environment."""
    venv_dir = ".venv"
    if not os.path.exists(venv_dir):
        print("\nCreating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
            print("✓ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            sys.exit(1)
    else:
        print("✓ Virtual environment already exists")
    
    return venv_dir

def install_dependencies(venv_dir):
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    
    # Determine the correct pip executable
    pip_exec = os.path.join(venv_dir, "bin", "pip")
    if platform.system() == "Windows":
        pip_exec = os.path.join(venv_dir, "Scripts", "pip.exe")
    
    try:
        # Install requirements
        subprocess.run([pip_exec, "install", "-r", "requirements.txt"], check=True)
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def check_cuda():
    """Check CUDA availability and version."""
    print("\nChecking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"✓ CUDA {cuda_version} is available")
            print(f"   - Current CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"   - Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("ℹ️  CUDA is not available. Using CPU (slower performance).")
            print("   For better performance, install CUDA 11.7+ and compatible PyTorch.")
    except ImportError:
        print("ℹ️  PyTorch not installed. CUDA check skipped.")

def create_directories():
    """Create necessary directories."""
    print("\nSetting up directories...")
    dirs = [
        "data/audio",
        "data/images",
        "results/benchmark",
        "results/models",
        "results/logs"
    ]
    
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def main():
    """Main setup function."""
    print_header()
    check_python_version()
    
    # Create virtual environment
    venv_dir = create_virtualenv()
    
    # Install dependencies
    install_dependencies(venv_dir)
    
    # Check CUDA
    check_cuda()
    
    # Create directories
    create_directories()
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print(f"   .\\{venv_dir}\\Scripts\\activate")
    else:
        print(f"   source {venv_dir}/bin/activate")
    print("\n2. Run the example usage:")
    print("   python example_usage.py")
    print("\n3. Run tests:")
    print("   python -m pytest tests/")

if __name__ == "__main__":
    main()
