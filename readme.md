# Setup 

1. Git clone the repository.
```bash
git clone repo_url
```
2. Optional: download uv and install it in your environment. You can also use pip to install the requirements.
```bash
pip install uv
```
3. Create a virtual environment and activate it.
```bash
uv venv # or python -m venv .venv
source .venv/bin/activate
```

4. Compile requirements based on your environment.
```bash
uv pip compile builder/requirements.in -o builder/requirements.txt # uv is optional, but recommended
``` 

5. Install the requirements.
```bash
uv pip sync builder/requirements.txt
```

6. Download the models from huggingface and save them in the `models_hub` directory before building. See src/download_models.py for more details.

7. Run the service locally using the following command.
```bash
python3 src/handler.py --rp_serve_api
```

8. The Embedding service is now running on `http://localhost:8000/`. You can test it using the following command.
```bash
curl --request POST \
  --url http://localhost:8000/runsync \
  --header 'Content-Type: application/json' \
  --data '{"input": {"task": "query","input_data": ["hello"]}}'  
```

# Production

The service is deplyed via Docker. The Dockerfile is located in the root directory. Due to GPU requirements, the service works well either on a mac chips with MPS backend (local development) or on a linux machine with CUDA installed (production).

# Commands

1. Test locally without docker (recommended on MacOS)

```bash
python src/handler.py --rp_serve_api
```

2. Build and publish image
   version is usually date  e.g. 20240930 or 20240930

```bash
./build_publish.sh {org} {repo} 
```

4. Push image to docker hub manually (instead of using buld_publish.sh - aka, you want a specific version)

```bash
docker push {org}/{repo}:version
```

## Hosting

We use runpod serverless endpoint. The Docker image is hosted on Docker Hub. 