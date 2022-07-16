#!/bin/bash

name="action_recognition_mmactionv1"
docker stop $name;
docker rm $name;

docker run --rm \
       --network=host \
       --privileged \
       --gpus all \
       --name="$name" \
       --volume="/dev:/dev" \
       --volume=$(pwd)/../webapp:/mmaction2/webapp \
       --volume=$(pwd)/../data:/mmaction2/data \
       --volume=$(pwd)/../demo:/mmaction2/demo \
       --volume=$(pwd)/../mmaction_diff:/mmaction2/mmaction \
       --volume=$(pwd)/../configs:/mmaction2/configs \
       --volume=$(pwd)/../pretrained_models:/mmaction2/pretrained_models \
       -it naoki:mmaction2_2206 \
       /bin/sh -c 'cd /mmaction2/webapp && uvicorn webapp:app --reload --host 0.0.0.0 --port 8083'
       # --volume="/home/ubuntu18/Codes/actionrecognition/mmaction2/imagemagick_config:/etc/ImageMagick-6" \
