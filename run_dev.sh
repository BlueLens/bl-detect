#!/usr/bin/env bash

RELEASE_MODE='dev'
GRPC_PORT=50052

sudo nvidia-docker run -dit --restart unless-stopped \
    -e RELEASE_MODE=$RELEASE_MODE \
    -e GRPC_PORT=$GRPC_PORT \
    -p $GRPC_PORT:$GRPC_PORT bluelens/bl-search-vector:$RELEASE_MODE
