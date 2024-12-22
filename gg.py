import sys
import subprocess
import os
import platform

import os
import requests
from tqdm import tqdm
from huggingface_hub import snapshot_download
import subprocess

# Set environment variable for faster HF downloads
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

def install_huggingface_hub():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub>=0.25.2"])

def install_hf_transfer():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hf_transfer>=0.1.8"])

# Check for huggingface_hub
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("huggingface_hub is not installed. Installing now...")
    install_huggingface_hub()
    from huggingface_hub import snapshot_download

# Check for hf_transfer
try:
    import hf_transfer
except ImportError:
    print("hf_transfer is not installed. Installing now...")
    install_hf_transfer()
    import hf_transfer


base_path = os.path.join("InstantID")


def ensure_directories_exist():
    directories = [
        os.path.join(base_path, "checkpoints"),
        os.path.join(base_path, "checkpoints", "ControlNetModel"),
        os.path.join(base_path, "models", "antelopev2")
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Created or verified directory: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {str(e)}")

def download_models():
    ensure_directories_exist()

    # InstantID files
    snapshot_download(
        repo_id="InstantX/InstantID",
        allow_patterns=["ip-adapter.bin", "ControlNetModel/diffusion_pytorch_model.safetensors", "ControlNetModel/config.json"],
        local_dir=os.path.join(base_path, "checkpoints")
    )

    # Antelopev2 models
    snapshot_download(
        repo_id="MonsterMMORPG/tools",
        allow_patterns=["1k3d68.onnx", "2d106det.onnx", "genderage.onnx", "glintr100.onnx", "scrfd_10g_bnkps.onnx"],
        local_dir=os.path.join(base_path, "models", "antelopev2")
    )
    
    # Best realism SG161222/RealVisXL_V4.0
    snapshot_download(
        repo_id="SG161222/RealVisXL_V4.0",
        allow_patterns=["RealVisXL_V4.0.safetensors"],
        local_dir=os.path.join(base_path, "models")
    )
    
    # Best realism eldritchPhotography_v1
    snapshot_download(
        repo_id="OwlMaster/Some_best_SDXL",
        allow_patterns=["eldritchPhotography_v1.safetensors"],
        local_dir=os.path.join(base_path, "models")
    )  

    # Best realism eldritchPhotography_v1
    snapshot_download(
        repo_id="OwlMaster/xinsir-controlnet-canny-sdxl-1.0",
        allow_patterns=["diffusion_pytorch_model.safetensors","config.json"],
        local_dir=os.path.join(base_path, "xinsir_controlnet", "controlnet-canny-sdxl-1.0")
    )  
    
    # Best realism eldritchPhotography_v1
    snapshot_download(
        repo_id="OwlMaster/xinsir-controlnet-depth-sdxl-1.0",
        allow_patterns=["diffusion_pytorch_model.safetensors","config.json"],
        local_dir=os.path.join(base_path, "xinsir_controlnet", "controlnet-depth-sdxl-1.0")
    )   

    # Best realism eldritchPhotography_v1
    snapshot_download(
        repo_id="OwlMaster/xinsir-controlnet-openpose-sdxl-1.0",
        allow_patterns=["diffusion_pytorch_model.safetensors","config.json"],
        local_dir=os.path.join(base_path, "xinsir_controlnet", "controlnet-openpose-sdxl-1.0")
    )      

if __name__ == "__main__":
    download_models()