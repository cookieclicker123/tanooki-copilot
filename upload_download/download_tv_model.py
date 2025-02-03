import os
from huggingface_hub import hf_hub_download, login
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def download_model_from_hf(
    repo_id: str = "SebLogsdon/tv-model",
):
    """
    Download model files from Hugging Face Hub
    """
    # Get token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Please set HF_TOKEN environment variable")
    
    # Create tmp/tv_model directory structure
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp', 'tv_model')
    os.makedirs(output_dir, exist_ok=True)
    
    # Login to Hugging Face
    login(token=hf_token)
    
    # List of files to download
    model_files = [
        "unsloth.F16.gguf",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json"
    ]

    # Download each file
    for filename in model_files:
        try:
            print(f"Downloading {filename}...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=hf_token
            )
            # Copy to output directory
            output_path = os.path.join(output_dir, filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            os.replace(downloaded_path, output_path)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    download_model_from_hf()