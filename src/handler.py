import base64
from io import BytesIO

import runpod
import torch
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.utils.colpali_processing_utils import (
    process_images,
    process_queries,
)
from PIL import Image
from transformers import AutoProcessor


def handler(job):
    job_input = job["input"]
    # job_input is a dictionary with the following keys:
    # - input_data: a list of base64 encoded images or text queries
    # - task: a string indicating the task to perform (either 'image' or 'query')
    if job_input["task"] == "image":
        embeddings = encode_image(job_input["input_data"])
    elif job_input["task"] == "query":
        embeddings = encode_query(job_input["input_data"])
    else:
        raise ValueError(f"Invalid task: {job_input['task']}")
    return {"output": embeddings}


MOCK_IMAGE = Image.new("RGB", (448, 448), (255, 255, 255))
device_map = "cuda" if torch.cuda.is_available() else None
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
if device_map:
    print(f"Using device: {device_map}")

model = ColPali.from_pretrained(
    "vidore/colpali",
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    token="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", 
)

model = model.eval()
processor = AutoProcessor.from_pretrained(
    "vidore/colpali", token="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"
)


def encode_image(input_data):
    """
    Compute embeddings for one or more images
    Args:
        input_data is a list of base64 encoded images

    Returns:
        an array of floats representing the embeddings of the input images
    """
    # goal is to convert input_data to a list of PIL images
    images = []
    for image in input_data:
        img_data = base64.b64decode(image)
        img = Image.open(BytesIO(img_data))
        images.append(img)

    with torch.no_grad():
        batch = process_images(processor, images)
        batch = {k: v.to(device) for k, v in batch.items()}
        embeddings = model(**batch)

    return embeddings.detach().to(torch.float32).cpu().numpy().tolist()


def encode_query(query):
    """
    Compute embeddings for one or more text queries.
    Args:
        query
            A list of text queries.
    Returns:
        an array of floats representing the embeddings of the input queries
    """
    with torch.no_grad():
        batch = process_queries(processor, query, MOCK_IMAGE)
        batch = {k: v.to(device) for k, v in batch.items()}
        embeddings = model(**batch)
 
    
    return embeddings.detach().to(torch.float32).cpu().numpy().tolist()
    
    


runpod.serverless.start({"handler": handler})


# code to pass to Qdrant
# # Parse the JSON response into a Python dictionary
#response = json.loads(response_json)

# Extract the vector
# Navigate through 'output' -> first list -> first sublist to get the vector
#vector = response["output"]["output"][0][0]
