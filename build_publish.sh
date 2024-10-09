#!/bin/bash

# Usage:
# ./build_publish.sh <docker_username> <image_name>

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <docker_username> <image_name>"
    exit 1
fi

DOCKER_USERNAME="$1"
IMAGE_NAME="$2"
VERSION=$(date +'%Y%m%d') # version is today's date, example: 20210101

DOCKERFILE="Dockerfile"

# Build the Docker image
docker build --platform linux/amd64 -f $DOCKERFILE --tag $DOCKER_USERNAME/$IMAGE_NAME:$VERSION .

# Push the image to Docker Hub
docker push $DOCKER_USERNAME/$IMAGE_NAME:$VERSION
