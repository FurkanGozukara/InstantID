from huggingface_hub import snapshot_download

snapshot_download(repo_id="lllyasviel/Annotators", local_dir="C:\models\lllyasviel\Annotators", local_dir_use_symlinks=False)