#!/usr/bin/env bash

RELEASE_MODE='prod'
GRPC_PORT=30052

sudo nvidia-docker run -dit --restart unless-stopped \
    -e RELEASE_MODE=$RELEASE_MODE \
    -e GRPC_PORT=$GRPC_PORT \
    -p $GRPC_PORT:$GRPC_PORT bluelens/bl-search-vector:$RELEASE_MODE
