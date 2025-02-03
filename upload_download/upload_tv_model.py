import os
from huggingface_hub import HfApi, login, create_repo
from pathlib import Path
import shutil
from dotenv import load_dotenv

load_dotenv()

def verify_token_and_namespace(token: str, namespace: str) -> bool:
    """Verify if the token is valid and has access to the namespace"""
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"Authenticated as: {user_info}")  # This will show us who we're authenticated as
        
        if user_info['name'] != namespace:
            print(f"Warning: Token namespace ({user_info['name']}) doesn't match target namespace ({namespace})")
            return False
        return True
    except Exception as e:
        print(f"Token verification failed: {e}")
        return False

def upload_model_to_hf(
    model_path: str,
    repo_name: str,
    hf_token: str,
    repo_type: str = "model",
):
    """Upload model files to Hugging Face Hub"""
    
    # Extract namespace from repo_name
    namespace = repo_name.split('/')[0]
    
    # Verify token and namespace match
    if not verify_token_and_namespace(hf_token, namespace):
        raise ValueError("Token doesn't have correct namespace permissions")

    # Initialize API once
    api = HfApi(token=hf_token)
    
    # Try to create/access repo but don't exit on failure
    try:
        api.create_repo(
            repo_id=repo_name,
            private=True,
            repo_type=repo_type,
            exist_ok=True
        )
        print(f"Repository {repo_name} ready")
    except Exception as e:
        print(f"Note: Repository operation returned: {e}")
        # Continue anyway - repo might already exist

    # File upload attempts
    model_files = [
        "unsloth.F16.gguf",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json"
    ]

    for filename in model_files:
        file_path = os.path.join(model_path, filename)
        if os.path.exists(file_path):
            try:
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=filename,
                    repo_id=repo_name,
                    repo_type=repo_type
                    # Removed duplicate token here as it's in the api instance
                )
                print(f"Uploaded {filename}")
            except Exception as e:
                print(f"Error uploading {filename}: {e}")
        else:
            print(f"Warning: {filename} not found")

if __name__ == "__main__":
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Please set HF_TOKEN environment variable")

    model_path = "tmp/tv_model"
    repo_name = "SebLogsdon/tv-model"

    upload_model_to_hf(
        model_path=model_path,
        repo_name=repo_name,
        hf_token=hf_token
    )