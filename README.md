DEEP Open Catalogue: Image classification
=========================================

[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code%2FDEEP-OC-org%2FUC-lifewatch-phyto-plankton-classification%2Fmaster)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/UC-lifewatch-phyto-plankton-classification/job/master/)


**Author:** [Ignacio Heredia & Wout Decrop](https://github.com/IgnacioHeredia) (CSIC & VLIZ)

**Project:** This work is part of the [iMagine](https://www.imagine-ai.eu/) project that receives funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 101058625.

**Project:** This work is part of the [DEEP Hybrid-DataCloud](https://deep-hybrid-datacloud.eu/) project that has
received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 777435.

This is a plug-and-play tool to train and evaluate an phytoplankton classifier on a custom dataset using deep neural networks.

You can find more information about it in the [iMagine Marketplace](https://dashboard.cloud.imagine-ai.eu/marketplace/modules/uc-lifewatch-deep-oc-phyto-plankton-classification).

**Table of contents**
1. [Installing this module](#installing-this-module)
    1. [Local installation](#local-installation)
    2. [Docker installation](#install-through-docker-recommended)
        1. [Install docker](#11-install-docker)
        2. [Run docker](#12-run-docker)
        3. [Clone the directory](#13-clone-the-directory)
        4. [Run the Docker container inside the local folder](#14-run-the-docker-container-inside-the-local-folder)
2. [Activating the module](#activating-the-module)
    1. [Activation of the API](#activation-of-the-api)
    2. [Activation of Jupyter notebook](#activation-of-jupyter-notebook)
3. [Train the phyto-plankton-classifier](#train-the-phyto-plankton-classifier)
    1. [Data preprocessing](#1-data-preprocessing)
        1. [Prepare the images](#11-prepare-the-images)
        2. [Prepare the data splits](#12-prepare-the-data-splits)
    2. [Training methods](#2-training-methods)
        1. [Train with cmd](#21-train-with-cmd)
            1. [Adapting the yaml file](#211-adapting-the-yaml-file)
            2. [Running the training](#212-running-the-training)
        2. [Train with Jupyter Notebooks (Recommended)](#22-train-with-jupyter-notebooks-recommended)
            1. [Adapting the yaml file](#221-adapting-the-yaml-file)
            2. [Go to Notebooks](#222-go-to-notebooks)
        3. [Train with Deepaas](#23-train-with-deepaas)
4. [Test an image classifier](#test-an-image-classifier)
    1. [Testing methods](#3-testing-methods)
        1. [Test with Jupyter Notebooks (Recommended)](#31-test-with-jupyter-notebooks-recommended)
            1. [Adapting the yaml file](#311-adapting-the-yaml-file)
            2. [Go to Notebooks](#312-go-to-notebooks)
        2. [Test with Deepaas](#32-test-with-deepaas)
5. [More info](#more-info)
6. [Acknowledgements](#acknowledgements)

# Installing this module

## Local installation (not recommended)
Although a local installation is possible, we recommend an installation through docker. This is less likely to breake support and has been tested with latest updates. We are working with python 3.6.9 which can be difficult to install. 
> **Requirements**
>
> This project has been tested in Ubuntu 18.04 with Python 3.6.9. Further package requirements are described in the
> `requirements.txt` file.
> - It is a requirement to have [Tensorflow>=1.14 installed](https://www.tensorflow.org/install/pip) (either in gpu 
> or cpu mode). This is not listed in the `requirements.txt` as it [breaks GPU support](https://github.com/tensorflow/tensorflow/issues/7166). 
> - Run `python -c 'import cv2'` to check that you installed correctly the `opencv-python` package (sometimes
> [dependencies are missed](https://stackoverflow.com/questions/47113029/importerror-libsm-so-6-cannot-open-shared-object-file-no-such-file-or-directo) in `pip` installations).

To start using this framework clone the repo and download the [default weights](https://share.services.ai4os.eu/index.php/s/rJQPQtBReqHAPf3/download):

```bash
# First line installs OpenCV requirement
apt-get update && apt-get install -y libgl1
git clone https://github.com/lifewatch/phyto-plankton-classification
cd phyto-plankton-classification
pip install -e .
curl -o ./models/phytoplankton_vliz.tar.xz https://share.services.ai4os.eu/index.php/s/rJQPQtBReqHAPf3/download #create share link from nextcloud
cd models && tar -xf phytoplankton_vliz.tar.xz && rm phytoplankton_vliz.tar.xz
```

## Install through Docker (recommended)

### 1.1 Install docker
Install [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/). 

### 1.2 Run docker
Ensure Docker is installed and running on your system before executing the DEEP-OC Phytoplankton Classification module using Docker containers.
So open docker, if correct, you should see a small ship (docker desktop) symbol running on the bottom right of your windows screen

### 1.3 Clone the directory
The directory is cloned so that the remote and the local directory are the same. This makes it easier to copy files inside the remote directory
```bash
git clone https://github.com/lifewatch/phyto-plankton-classification
cd phyto-plankton-classification
```

### 1.4 Run the Docker Container Inside the Local Folder

After Docker is installed and running, you can run the ready-to-use [Docker container](https://hub.docker.com/r/deephdc/uc-lifewatch-deep-oc-phyto-plankton-classification) to run this module. There are two options for handling images based on their storage location:

Run container and only have local access
```bash
docker run -ti -p 8888:8888 -p 5000:5000 -v "$(pwd):/srv/phyto-plankton-classification" deephdc/uc-lifewatch-deep-oc-phyto-plankton-classification:latest /bin/bash
```

> **Tip**: Rclone can also be configured to acces nextcloud server, follow [Tutorial](https://docs.ai4eosc.eu/en/latest/user/howto/rclone.html#configuring-rclone).


Now the environment has the right requiremens to be excecuted. 


# 1. Train the phyto-plankton-classifier

You can train your own audio classifier with your custom dataset. For that you have to:

## 1. Data preprocessing

The first step to train you image classifier if to have the data correctly set up. 

### 1.1 Prepare the images

The model needs to be able to access the images. So you have to place your images in the [./data/images](/data/images) folder. If you have your data somewhere else you can use that location by setting the `image_dir` parameter in the training args. 
Please use a standard image format (like `.png` or `.jpg`). 

You can copy the images to [phyto-plankton-classification/data/images](/data/images) folder on your pc. 
If the images are on nextcloud, you can one of the next steps depending if you have rclone or not. 


### 1.2 Prepare the data splits (optional)

Next, you need add to the [./data/dataset_files](/data/dataset_files) directory the following files:

| *Mandatory files* | *Optional files*  | 
|:-----------------------:|:---------------------:|
|  `classes.txt`, `train.txt` |  `val.txt`, `test.txt`, `info.txt`,`aphia_ids.txt`|

The `train.txt`, `val.txt` and `test.txt` files associate an image name (or relative path) to a label number (that has
to *start at zero*).
The `classes.txt` file translates those label numbers to label names.
The `aphia_ids.txt` file translates those the classes to their corresponding aphia_ids.
Finally the `info.txt` allows you to provide information (like number of images in the database) about each class. 

You can find examples of these files at [./data/demo-dataset_files](/data/demo-dataset_files).

If you don't want to create your own datasplit, this will be done automatically for you with a 80% train, 10% validation, and 10% test split.


## 2. Training methods

### 2.1: Train with cmd

#### 2.1.1: Adapting the yaml file
Clarify the location of the images inside the [yaml file](/etc/config.yaml) file. If not, [./data/images](/data/images) will be taken. 
Any additional parameter can also be changed here such as the type of split for training/validation/testing, batch size, etc

You can change the config file directly as shown below, or you can change it when running the api.

```bash
  images_directory:
    value: "/srv/phyto-plankton-classification/data/images"
    type: "str"
    help: >
          Base directory for images. If the path is relative, it will be appended to the package path.
```
#### 2.1.2: Running the training
After this, you can go to `/srv/phyto-plankton-classification/planktonclas#` and run `train_runfile.py`.

```bash
cd /srv/phyto-plankton-classification/planktonclas` 
python train_runfile.py
```
The new model will be saved under [phyto-plankton-classification/models](/models)

### 2.2: Train with Jupyter Notebooks (Recommended)
#### 2.2.1: Adapting the yaml file
Similar to [2.1.2: Running the training](#2.1.2:_Running_the_training),clarify the location of the images inside the [yaml file](/etc/config.yaml) file. If not, [./data/images](/data/images) will be taken. 
Any additional parameter can also be changed here such as the type of split for training/validation/testing, batch size, etc

You can change the config file directly as shown below.

```bash
  images_directory:
    value: "/srv/phyto-plankton-classification/data/images"
    type: "str"
    help: >
          Base directory for images. If the path is relative, it will be appended to the package path.
```
#### 2.2.2: Go to Notebooks

You can have more info on how to interact directly with the module (not through the DEEPaaS API) by examining the 
``./notebooks`` folder:

* [dataset exploration notebook](./notebooks/1.0-Dataset_exploration.ipynb):
  Visualize relevant statistics that will help you to modify the training args.
* [Image transformation notebook](./notebooks/1.1-Image_transformation.ipynb):
  To conform a new dataset with the training set that was used
* [Image transformation notebook](./notebooks/1.2-Image_augmentation):
  Notebook to perform image augmentation and expand the dataset.
* [Model training notebook](./notebooks/2.0-Model_training):
  Notebook to perform image augmentation and expand the dataset.
* [computing predictions notebook](./notebooks/3.0-Computing_predictions.ipynb):
  Test the classifier on a number of tasks: predict a single local image (or url), predict multiple images (or urls),
  merge the predictions of a multi-image single observation, etc.
* [predictions statistics notebook](./notebooks/3.1-Prediction_statistics.ipynb):
  Make and store the predictions of the `test.txt` file (if you provided one). Once you have done that you can visualize
  the statistics of the predictions like popular metrics (accuracy, recall, precision, f1-score), the confusion matrix, etc.

## 2.2: Train with Deepaas
### activation of the API
now run DEEPaaS:
```
deepaas-run --listen-ip 0.0.0.0
```
and open http://0.0.0.0:5000/ui (or http://127.0.0.1:5000/api#/) and look for the methods belonging to the `planktonclas` module.
Look for the ``TRAIN`` POST method. Click on 'Try it out', change whatever training args
you want and click 'Execute'. The training will be launched and you will be able to follow its status by executing the 
``TRAIN`` GET method which will also give a history of all trainings previously executed.

You can follow the training monitoring (Tensorboard) on http://0.0.0.0:6006.



# TEST the phyto-plankton-classifier
## 3. Testing methods
### 3.1: Train with Jupyter Notebooks (Recommended)
#### 3.1.1: Adapting the yaml file
Similar to [2.1.2: Running the test](#2.1.2:_Running_the_test),clarify the location of the images that need to be predicted inside the [yaml file](/etc/config.yaml) file.  
You can change the config file directly as shown below.

```bash
testing:
  file_location:
    value: "/srv/phyto-plankton-classification/data/demo-images/Actinoptychus"
    type: "str"
    help: >
      Select the folder of the images you want to classify. For example: /storage/.../images_to_be_predicted
```   

#### 3.1.2: Go to Notebooks

You can have more info on how to interact directly with the module (not through the DEEPaaS API) by examining the 
``./notebooks`` folder:

* [computing predictions notebook](./notebooks/3.0-Computing_predictions.ipynb):
  Test the classifier on a number of tasks: predict a single local image (or url), predict multiple images (or urls),
  merge the predictions of a multi-image single observation, etc.
* [predictions statistics notebook](./notebooks/3.1-Prediction_statistics.ipynb):
  Make and store the predictions of the `test.txt` file (if you provided one). Once you have done that you can visualize
  the statistics of the predictions like popular metrics (accuracy, recall, precision, f1-score), the confusion matrix, etc.


## 3.2: Test with Deepaas
### activation of the API
now run DEEPaaS:
```
deepaas-run --listen-ip 0.0.0.0
```
Go to http://0.0.0.0:5000/ui (or http://127.0.0.1:5000/api#/) and look for the `PREDICT` POST method. Click on 'Try it out', change whatever test args
you want and click 'Execute'. You can **either** supply a:

* a `image` argument with a path pointing to an image.

* a `zip` argument with an URL pointing to zipped folder with images.

* a `file_location` argument with the local location of the folder with images you want predicted

#### option 2: Follow the notebooks 
Follow the notebook for [computing the predictions](./notebooks/3.0-Computing_predictions.ipynb)
Make sure to select DEMO or not if you want to predict your own data of the demo data as an example.

## Extra information
### Activation of jupyter notebook
You can also activate the jupyter notebooks inside the docker container and work from there. 
```
deep-start -j
```
This will automatically start the notebook. You get the following output

you get the following output:
```bash
[I 12:34:56.789 NotebookApp]  To access the notebook, open this file in a browser:
     file:///root/.local/share/jupyter/runtime/nbserver-1234-open.html
[I 12:34:56.789 NotebookApp]  Or copy and paste one of these URLs:
     http://127.0.0.1:8888/?token=your_token_here
```
You can this go to think link in your brower or copy this final link and use it as a kernel on your local vsc

## Acknowledgements

If you consider this project to be useful, please consider citing the DEEP Hybrid DataCloud project:

> García, Álvaro López, et al. [A Cloud-Based Framework for Machine Learning Workloads and Applications.](https://ieeexplore.ieee.org/abstract/document/8950411/authors) IEEE Access 8 (2020): 18681-18692. 
