import os
import sys

import clip
import open_clip
import torch

from dotenv import load_dotenv





def load_clip_model(model_name: str, device: torch.device,clip_backend:str):
    """Load CLIP model and preprocessing transform."""
    if clip_backend == "openai":
        model, preprocess = clip.load(model_name, device=device)
        print(f"Loaded openai/CLIP  model='{model_name}'  device={device}")
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
        model = model.to(device)
        print(f"Loaded open_clip  model='{model_name}'  device={device}")
    model.eval()
    return model, preprocess


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

COLLECTION_NAME = "cifar_clip"


def get_chroma_db_client():
    
    try:
        import chromadb
    except ImportError:
        sys.exit("ChromaDB client not found.\nInstall it with:  pip install chromadb-client")
        
    load_dotenv()
   
    # Load ChromaDB connection settings from environment variables
    CHROMA_DB_HOST = os.getenv("CHROMA_DB_HOST", "localhost")
    CHROMA_DB_PORT = os.getenv("CHROMA_DB_PORT", "8000")

    # Create ChromaDB client with connection settings
    try:
        client = chromadb.HttpClient(
            host=CHROMA_DB_HOST,
            port=CHROMA_DB_PORT
        )
        print(f"Connected to ChromaDB at {CHROMA_DB_HOST}:{CHROMA_DB_PORT}")
    except Exception as e:
        sys.exit(f"Failed to connect to ChromaDB at {CHROMA_DB_HOST}:{CHROMA_DB_PORT}\nError: {e}")
    return client