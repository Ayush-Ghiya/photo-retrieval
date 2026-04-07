import clip
import open_clip
import torch


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