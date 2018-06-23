#! /usr/bin/env bash

REPO="tanmaniac"
IMAGE="cs231n"
TAG="gpu"
CONTAINER_NAME="tensorflow-${TAG}"

if [ ! "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=${CONTAINER_NAME})" ]; then
        # cleanup
        docker rm ${CONTAINER_NAME}
    fi
    docker run -it \
    --runtime=nvidia \
    --privileged \
    --name ${CONTAINER_NAME} \
    -p 8888:8888 -p 6006:6006 \
    -v $(pwd)/../../tensorflow:/home/$(id -un)/tensorflow \
    ${REPO}/${IMAGE}:${TAG} \
    /bin/bash
fi
