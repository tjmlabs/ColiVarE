import base64
from io import BytesIO

import runpod
import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from PIL import Image

device_map = "cuda" if torch.cuda.is_available() else None

model = ColPali.from_pretrained(
    "models_hub/models--vidore--colpali/snapshots/55e76ff047b92147638dbdd7aa541b721f794be1",
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    local_files_only=True,
)


processor = ColPaliProcessor.from_pretrained(
    "models_hub/models--google--paligemma-3b-mix-448/snapshots/ead2d9a35598cb89119af004f5d023b311d1c4a1",
    local_files_only=True,
)


def encode_image(input_data):
    """
    Compute embeddings for one or more images
    Args:
        input_data is a list of base64 encoded images

    Returns:
        an array of floats representing the embeddings of the input images

    Example in repo: images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (16, 16), color="black"),
        ]
    """
    # goal is to convert input_data to a list of PIL images
    images = []
    for image in input_data:
        img_data = base64.b64decode(image)
        img = Image.open(BytesIO(img_data))
        img = img.convert("RGB")
        images.append(img)

    batch_images = processor.process_images(images).to(model.device)

    with torch.no_grad():
        image_embeddings = model(**batch_images)

    # Compute total tokens
    seq_length = image_embeddings.shape[1]  # Sequence length dimension
    total_tokens = seq_length * len(input_data)

    results = []
    for idx, embedding in enumerate(image_embeddings):
        embedding = embedding.to(torch.float32).detach().cpu().numpy().tolist()
        result = {"object": "embedding", "embedding": embedding, "index": idx}
        results.append(result)
    return results, total_tokens


def encode_query(queries):
    """
        Compute embeddings for one or more text queries.
        Args:
            queries
                A list of text queries.
        Returns:
            an array of floats representing the embeddings of the input queries
        Example in repo: queries = [
        "Is attention really all you need?",
        "Are Benjamin, Antoine, Merve, and Jo best friends?",
    ]
    """
    batch_queries = processor.process_queries(queries)
    input_ids = batch_queries["input_ids"]
    total_tokens = sum(len(ids) for ids in input_ids)
    batch_queries = batch_queries.to(model.device)
    for input_ids in batch_queries["input_ids"]:
        total_tokens += len(input_ids)

    with torch.no_grad():
        query_embeddings = model(**batch_queries)

    # Build the list of embedding objects
    results = []
    for idx, embedding in enumerate(query_embeddings):
        embedding = embedding.to(torch.float32).detach().cpu().numpy().tolist()
        result = {"object": "embedding", "embedding": embedding, "index": idx}
        results.append(result)

    return results, total_tokens


def handler(job):
    job_input = job["input"]
    # job_input is a dictionary with the following keys:
    # - input_data: a list of base64 encoded images or text queries
    # - task: a string indicating the task to perform (either 'image' or 'query')
    if job_input["task"] == "image":
        embeddings, total_tokens = encode_image(job_input["input_data"])
    elif job_input["task"] == "query":
        embeddings, total_tokens = encode_query(job_input["input_data"])
    else:
        raise ValueError(f"Invalid task: {job_input['task']}")
    return {
        "object": "list",
        "data": embeddings,  # has to be a list of lists for scoring
        "model": "vidore/colpal-v1.2",
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


# runpod.serverless.start({"handler": handler})
