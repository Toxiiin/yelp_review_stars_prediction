# Rating prediction via review scan for yelp database
This repository contains an implementation for a review (text) to stars mapping prediction in keras.

# Installation

1) Install docker from https://docs.docker.com/
2) Install container for cpu build via:```docker build -t keras_cpu --build-arg python_version=3.6 --build-arg cuda_version=9.0 --build-arg cudnn_version=7 -f Dockerfile_cpu .``` or for gpu build via ```docker build -t keras_gpu --build-arg python_version=3.6 --build-arg cuda_version=9.0 --build-arg cudnn_version=7 -f Dockerfile_gpu .```
3) 
