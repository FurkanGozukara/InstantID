import os
import sys
import time
import zipfile
import subprocess

# Ensure required libraries are installed
def ensure_libraries():
    try:
        import requests
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests
    try:
        from tqdm import tqdm
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        from tqdm import tqdm

def download_file(url, dest, max_retries=5):
    import requests
    from tqdm import tqdm

    headers = {}
    initial_pos = 0
    if os.path.exists(dest):
        initial_pos = os.path.getsize(dest)
        headers['Range'] = f'bytes={initial_pos}-'

    attempt = 0
    while attempt < max_retries:
        try:
            with requests.get(url, headers=headers, stream=True) as r:
                if r.status_code == 416:  # Requested Range Not Satisfiable
                    print(f"File {dest} is already fully downloaded.")
                    return
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0)) + initial_pos
                mode = 'ab' if initial_pos > 0 else 'wb'
                with open(dest, mode) as f, tqdm(
                        total=total_size,
                        initial=initial_pos,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=dest,
                        ascii=True,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
                ) as progress:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            progress.update(len(chunk))
            break
        except (requests.HTTPError, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            print(f"Error downloading file: {e}")
            attempt += 1
            if attempt < max_retries:
                print(f"Retrying... ({attempt}/{max_retries})")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

    # Verify file size
    if os.path.exists(dest):
        actual_size = os.path.getsize(dest)
        if actual_size == total_size:
            print(f"File {dest} successfully downloaded and verified.")
        else:
            print(f"Warning: File size mismatch. Expected {total_size}, got {actual_size}.")

# Extract the zip file and overwrite existing files
def extract_zip(file_path, extract_to='.'):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)