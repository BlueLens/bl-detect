#!/usr/bin/env bash

sudo nvidia-docker run -dit --restart unless-stopped -p 50051:50051 bluelens/bl-detect:latest