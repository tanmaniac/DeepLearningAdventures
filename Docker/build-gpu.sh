#! /usr/bin/env bash

REPO="tanmaniac"
IMAGE="cs231n"
TAG="gpu"

docker build --build-arg UID=$(id -u) \
--build-arg GID=$(id -g) \
--build-arg UNAME=$(id -un) \
--tag ${REPO}/${IMAGE}:${TAG} -f Dockerfile.${TAG} .