# change to model you want to download
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
import torch

model_name = "vidore/colSmol-256M"  # or whatever model you want to download
if torch.backends.mps.is_available():
    device_map = "mps"
elif torch.cuda.is_available():
    device_map = "auto"
else:
    device_map = None


def first_time():
    model = ColIdefics3.from_pretrained(
        model_name,
        cache_dir="models_hub/",  # where to save the model
        device_map=device_map,
    )

    processor = ColIdefics3Processor.from_pretrained(
        model_name, cache_dir="models_hub/"
    )
    return model, processor


# call the function
first_time()


def test_after_1st_time():
    model = ColIdefics3.from_pretrained(
        model_name,
        local_files_only=True,
        cache_dir="models_hub/",
        device_map=device_map,
    )
    processor = ColIdefics3Processor.from_pretrained(
        model_name, local_files_only=True, cache_dir="models_hub/"
    )
    # it shoudln't download anything from the internet again
    return model, processor
