# change to model you want to download
from colpali_engine.models import ColQwen2, ColQwen2Processor
from pathlib import Path
from model2vec import StaticModel
import torch
import huggingface_hub

model_name = "vidore/colqwen2-v1.0"  # or whatever model you want to download
static_model_name = "minishlab/potion-base-2M"
if torch.cuda.is_available():
    device_map = "cuda"
elif torch.backends.mps.is_available():
    device_map = "mps"
else:
    device_map = None


def download_static_model(first_time=False):
    folder_or_repo_path = static_model_name
    folder_or_repo_path = Path(folder_or_repo_path)
    if first_time:
        huggingface_hub.hf_hub_download(
            folder_or_repo_path.as_posix(), "model.safetensors", cache_dir="models_hub/"
        )
        huggingface_hub.hf_hub_download(
            folder_or_repo_path.as_posix(), "README.md", cache_dir="models_hub/"
        )
        huggingface_hub.hf_hub_download(
            folder_or_repo_path.as_posix(), "config.json", cache_dir="models_hub/"
        )
        huggingface_hub.hf_hub_download(
            folder_or_repo_path.as_posix(), "tokenizer.json", cache_dir="models_hub/"
        )

    model = StaticModel.load_local("models_hub/models--minishlab--potion-base-2M/snapshots/ed90dd52cd420507eef6f5f0c638e935b4e992c3")
    return model


def first_time():
    model = ColQwen2.from_pretrained(
        model_name,
        cache_dir="models_hub/",  # where to save the model
        device_map=device_map,
    )

    processor = ColQwen2Processor.from_pretrained(model_name, cache_dir="models_hub/")
    return model, processor


def test_after_1st_time():
    model = ColQwen2.from_pretrained(
        model_name,
        local_files_only=True,
        cache_dir="models_hub/",
        device_map=device_map,
    )
    processor = ColQwen2Processor.from_pretrained(
        model_name, local_files_only=True, cache_dir="models_hub/"
    )
    # it shoudln't download anything from the internet again
    return model, processor
