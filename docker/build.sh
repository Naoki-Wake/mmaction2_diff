#!/bin/bash

cd ../..
docker build --network=host -t naoki:mmaction2_2206 -f mmaction2/docker/Dockerfile2022 .
