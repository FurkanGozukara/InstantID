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
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0)) + initial_pos
                with open(dest, 'ab') as f, tqdm(
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

# Extract the zip file and overwrite existing files
def extract_zip(file_path, extract_to='.'):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
