# Rating prediction via review text scan for yelp database
This repository contains an implementation for a review (text) to stars mapping prediction in keras.

# 1) Installation

1) Install docker from https://docs.docker.com/.
2) Open command line window and navigate to the project folder with all files from this repository.
3) Install a container for cpu build via: ```docker build -t keras_cpu --build-arg python_version=3.6 -f Dockerfile_cpu .``` or for gpu build via ```docker build -t keras_gpu --build-arg python_version=3.6 --build-arg cuda_version=9.0 --build-arg cudnn_version=7 -f Dockerfile_gpu .```.

# 2) Start Training and Evaluation

Start a docker container. For cpu version type: ```docker run -it -v <path_to_project_folder>:/home -v <path_to_yelp_database>:/data --env KERAS_BACKEND=tensorflow keras_cpu bash``` and for gpu version: ```nvidia-docker run -it -v <path_to_project_folder>:/home -v <path_to_yelp_database>:/data --env KERAS_BACKEND=tensorflow keras_gpu bash```. Attention: For GPU Version, please install NVIDIA drivers (ideally latest) and nvidia-docker from https://github.com/NVIDIA/nvidia-docker.

To start the training of network, run ```train.py```. The script takes several arguments to control the training process and the data extraction. To get an overview, type ```train.py -h``` for a help message.

After training there are three files generated as output:
1) The network model as \*.hdf5-file or \*.h5-file
2) A training log
3) A raw data file from pickle, which contains the word mappings from text tokenizer

# 3) Predict rating on new custom texts

Start a docker container (like you did to start a training). Start the prediction with the execution of ```predict.py```. The scripts takes three parameters, type ```predict.py -h``` to get more information.

To enter new texts you can define strings direcly in the script with assigning them to variable ```input_texts```.

# Backup: Anaconda

If the docker container wont work you could take a look at the conda_environments folder. The folder contains \*.yml files to generate conda environments. To generate an environment, install a valid anaconda version (see https://www.anaconda.com/) and type ```conda env create -f environment.yml``` to set the environment up (more information @ https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).
