import os
import requests
from tqdm import tqdm

def create_directory(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")

def download_file(url, folder_path):
    print(f"{url}")
    """Download a file from a given URL to a specified folder."""
    local_filename = url.split('/')[-1]
    local_filepath = os.path.join(folder_path, local_filename)

    # Stream download to handle large files
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(local_filepath, 'wb') as f:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    else:
        print(f"Downloaded {local_filename} to {folder_path}")

# Define the folders and their corresponding file URLs
folders_and_files = {
    os.path.join("checkpoints"): [
        "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin"
    ],
	os.path.join("checkpoints","ControlNetModel"): [
        "https://huggingface.co/LiheYoung/depth_anything_vitl14/resolve/main/pytorch_model.bin",
		"https://huggingface.co/LiheYoung/depth_anything_vitl14/resolve/main/config.json",
    ],
	os.path.join("models","antelopev2"): [
        "https://huggingface.co/MonsterMMORPG/tools/resolve/main/1k3d68.onnx",
		"https://huggingface.co/MonsterMMORPG/tools/resolve/main/2d106det.onnx",
		"https://huggingface.co/MonsterMMORPG/tools/resolve/main/genderage.onnx",
		"https://huggingface.co/MonsterMMORPG/tools/resolve/main/glintr100.onnx",
		"https://huggingface.co/MonsterMMORPG/tools/resolve/main/scrfd_10g_bnkps.onnx"
    ]
}

# Perform the download process
for folder, files in folders_and_files.items():
    create_directory(folder)
    for file_url in files:
        download_file(file_url, folder)
