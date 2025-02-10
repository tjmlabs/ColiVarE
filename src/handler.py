import base64
from io import BytesIO
from typing import Any, Dict, List, Tuple

import runpod
import torch
from colpali_engine.models import ColQwen2, ColQwen2Processor
from PIL import Image

from scipy.cluster.hierarchy import linkage, fcluster

if torch.cuda.is_available():
    device_map = "cuda"
elif torch.backends.mps.is_available():
    device_map = "mps"
else:
    device_map = None

model_name = "vidore/colqwen2-v1.0"
model = ColQwen2.from_pretrained(
    model_name,
    local_files_only=True,
    cache_dir="models_hub/",
    torch_dtype=torch.bfloat16,
    device_map=device_map,
)


processor = ColQwen2Processor.from_pretrained(
    model_name, local_files_only=True, cache_dir="models_hub/"
)


def pool_embeddings(embeddings: torch.Tensor, pool_factor: int = 3) -> List[List[float]]:
    """
    Reduces number of embeddings by clustering similar ones together.
    
    Args:
        embeddings: Single image embeddings of shape (1038, 128)
                   Example with 4 vectors, 3 dimensions for simplicity:
                   [[1,0,1],
                    [1,0,1],
                    [0,1,0],
                    [0,1,0]]
    """
    similarities = torch.mm(embeddings, embeddings.t())
    
    distances = 1 - similarities.cpu().numpy()
    del similarities   
    torch.cuda.empty_cache() 
    
    target_clusters = max(embeddings.shape[0] // pool_factor, 1)

    clusters = linkage(distances, method="ward")
    cluster_labels = fcluster(clusters, t=target_clusters, criterion="maxclust")

    pooled = []
    for cluster_id in range(1, target_clusters + 1):
        mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[mask]
        cluster_mean = cluster_embeddings.mean(dim=0)

        pooled.append(cluster_mean.cpu().tolist())  
        del cluster_embeddings, cluster_mean
        torch.cuda.empty_cache() 
    
    return pooled

def encode_image(input_data: List[str]) -> Tuple[List[Dict[str, Any]], int]:
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

    with torch.no_grad():
        batch_images = processor.process_images(images).to(model.device)
        image_embeddings = model(**batch_images)

    results = []
    for idx, embedding in enumerate(image_embeddings):
        embedding = embedding.to(torch.float32)
        pooled = pool_embeddings(embedding)
        del embedding
        torch.cuda.empty_cache()
        result = {"object": "embedding", "embedding": pooled, "index": idx}
        results.append(result)
    
    del batch_images, image_embeddings
    torch.cuda.empty_cache()
    # Compute total tokens
    total_tokens = len(results) * len(pooled)
    return results, total_tokens


def encode_query(queries: List[str]) -> Tuple[List[Dict[str, Any]], int]:
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
    # Count tokens
    total_tokens = sum(len(ids) for ids in batch_queries["input_ids"])

    batch_queries = batch_queries.to(model.device)

    with torch.no_grad():
        query_embeddings = model(**batch_queries)

    results = []
    for idx, embedding in enumerate(query_embeddings):
        embedding = embedding.to(torch.float32).detach().cpu().numpy().tolist()
        result = {"object": "embedding", "embedding": embedding, "index": idx}
        results.append(result)
        del embedding
        torch.cuda.empty_cache()
    del batch_queries, query_embeddings
    torch.cuda.empty_cache()
    return results, total_tokens


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
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
        "data": embeddings,
        "model": model_name,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


runpod.serverless.start({"handler": handler})
