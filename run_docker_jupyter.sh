#!/usr/bin/env bash

docker run -it --rm \
       -v $PWD:/home/manifold_learning \
       -w /home/manifold_learning \
       -p 8888:8888 \
       nvcr.io/nvidia/pytorch:19.06-py3 \
       jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
