# change to model you want to download
from colpali_engine.models import ColQwen2, ColQwen2Processor
import torch

model_name = "manu/colqwen2-v0.2"  # or whatever model you want to download
if torch.cuda.is_available():
    device_map = "cuda"
elif torch.backends.mps.is_available():
    device_map = "mps"
else:
    device_map = None

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
