# Rating prediction via review scan for yelp database
This repository contains an implementation for a review (text) to stars mapping prediction in keras.

# Installation

1) Install docker from https://docs.docker.com/.
2) Open command line window and navigate to the project folder with all files from this repository.
3) Install a container for cpu build via: ```docker build -t keras_cpu --build-arg python_version=3.6 --build-arg cuda_version=9.0 --build-arg cudnn_version=7 -f Dockerfile_cpu .``` or for gpu build via ```docker build -t keras_gpu --build-arg python_version=3.6 --build-arg cuda_version=9.0 --build-arg cudnn_version=7 -f Dockerfile_gpu .```.

# Start Training

Start a docker container. For cpu version type: ```docker run -it -v <path_to_project_folder>:/home -v <path_to_yelp_database>:/data --env KERAS_BACKEND=tensorflow keras_cpu bash``` and for gpu version: ```nvidia-docker run -it -v <path_to_project_folder>:/home -v <path_to_yelp_database>:/data --env KERAS_BACKEND=tensorflow keras_gpu bash```. Attention: For GPU Version, please install NVIDIA drivers (ideally latest) and nvidia-docker from https://github.com/NVIDIA/nvidia-docker.

To start the training of network, log into your preferred docker container and run ```train.py```. The script takes several arguments to control the training process and the data extraction. To get an overview, type ```train.py -h``` for a help message.

After training there are three files generated as output:
1) The network model as \*.hdf5-file or \*.h5-file
2) A training log
3) A raw data file from pickle, which contains the word mappings from text tokenizer
