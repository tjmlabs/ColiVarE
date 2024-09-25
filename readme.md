# Commands

1. Run local image bash
```bash
docker run -it --rm jonathanadly/colipali-embeddings:version /bin/bash
```

2. Build image
version is usually date + iteration, e.g. 9.24.2024.0 
```bash
docker build --platform linux/amd64 --tag jonathanadly/colipali-embeddings:9.24.2024.0 .
```

3. Test locally without docker 
```bash
python src/handler.py --rp_serve_api
```

4. Push image to docker hub
```bash
docker push jonathanadly/colipali-embeddings:version
```