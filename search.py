import sys


import torch
import torch.nn.functional as F

from utils import get_chroma_db_client, load_clip_model,COLLECTION_NAME


try:
    import clip
    CLIP_BACKEND = "openai"
except ImportError:
    try:
        import open_clip
        CLIP_BACKEND = "open_clip"
    except ImportError:
        sys.exit(
            "No CLIP library found.\n"
            "Install one of:\n"
            "  pip install git+https://github.com/openai/CLIP.git\n"
            "  pip install open-clip-torch"
        )




# ---------------------------------------------------------------------------
# DB + CLIP
# ---------------------------------------------------------------------------

def connect_collection(db_dir: str):
    client = get_chroma_db_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        sys.exit(
            f"Collection '{COLLECTION_NAME}' not found in '{db_dir}'.\n"
            f"Run build_index.py first:\n"
            f"  python build_index.py --dataset_dir ./cifar_images"
        )
    meta  = collection.metadata or {}
    model = meta.get("model", "ViT-B/32")
    print(f"Connected to ChromaDB  —  {collection.count():,} vectors  |  model='{model}'")
    return collection, model




def encode_text(prompt: str, model, device: torch.device) -> list[float]:
    if CLIP_BACKEND == "openai":
        tokens = clip.tokenize([prompt]).to(device)
    else:
        tokens = open_clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        feat = model.encode_text(tokens)
    return F.normalize(feat, dim=-1).cpu().squeeze(0).tolist()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def get_images_from_db(
    collection,
    query_embedding: list[float],
    top_k: int,
) -> tuple[list[dict], list[float]]:
    """
    Query ChromaDB.  Supports optional metadata filters
    """
    
    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"],
    )


    results = collection.query(**kwargs)

    metadatas = results["metadatas"][0]   # list of dicts
    distances = results["distances"][0]   # cosine distance: 0 = identical, 2 = opposite

    # Convert cosine distance -> similarity score in [0, 1]
    # similarity = 1 - (distance / 2)
    scores = [1.0 - (d / 2.0) for d in distances]

    return metadatas, scores



def search_images(prompt: str = "a photo of a cat", top_k: int = 5,db_dir: str = "./chroma_db") -> list[dict]:
    # Connect to DB and CLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n" + "=" * 55)
    print("  CLIP Image Search  (ChromaDB)")
    print("=" * 55)
    print(f"  Prompt       : {prompt}")
    print(f"  DB dir       : {db_dir}")
    print(f"  Top-k        : {top_k}")
    print(f"  Device       : {device}")
    print("=" * 55 + "\n")
    
    collection, model_name = connect_collection(db_dir=db_dir)
 
    model, preprocess = load_clip_model(model_name, device,CLIP_BACKEND)

    # Encode query text
    print(f'\nEncoding prompt: "{prompt}"')
    query_embedding = encode_text(prompt, model, device)

    # Search DB
    print("Searching ChromaDB ...")
    results, scores = get_images_from_db(collection, query_embedding, top_k)

    #Print results
    print(f"\nTop-{len(results)} results for: \"{prompt}\"\n")
    print(f"  {'Rank':<5}  {'Score':>6}  {'File'}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*5}  {'-'*14}  {'-'*30}")
    for rank, (rec, score) in enumerate(zip(results, scores), 1):
        print(
            f"  {rank:<5}  {score:>6.4f}  "
            f"  {rec['filename']}"
        )
        
    # Format results for output
    image_list = []
    for meta, score in zip(results, scores):
        image_list.append({
            "filename": meta.get("filename"),
            "path": meta.get("path"),
            "score": score,
        })

    return image_list,results,scores