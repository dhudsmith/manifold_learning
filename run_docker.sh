#!/usr/bin/env bash

docker run --rm -it -v {$PWD}:/home/maifold_learning -w /home/manifold_learning -p 8888:8888 nvcr.io/nvidia/pytorch:19.06-py3