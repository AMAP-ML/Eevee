from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="JianhaoZeng/Eevee",
    local_dir="./data",
    repo_type="model"
)