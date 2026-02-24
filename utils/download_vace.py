from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Wan-AI/Wan2.1-VACE-14B",
    local_dir="./checkpoints/Wan2.1-VACE-14B",
    repo_type="model"
)