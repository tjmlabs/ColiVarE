# Commands

1. Run local image bash
```bash
docker run -it --rm jonathanadly/colipali-embeddings:version-local
```

2. Build cloud image
version is usually date + version, e.g. 20240930-cloud or 20240930-local
```bash
docker build --platform linux/amd64 --tag jonathanadly/colipali-embeddings:20240930-cloud .
```

3. Build local image
```bash
docker build -f Dockerfile.local --platform linux/amd64 --tag jonathanadly/colipali-embeddings:20240930-local .
```

3. Test locally without docker 
```bash
python src/handler.py --rp_serve_api
```

4. Push image to docker hub
```bash
docker push jonathanadly/colipali-embeddings:version
```