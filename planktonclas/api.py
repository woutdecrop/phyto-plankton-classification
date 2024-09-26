"""
API for the image classification package

Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia

Notes: Based on https://github.com/indigo-dc/plant-classification-theano/blob/package/plant_classification/api.py

Descriptions:
The API will use the model files inside ../models/api. If not found it will use the model files of the last trained model.
If several checkpoints are found inside ../models/api/ckpts we will use the last checkpoint.

Warnings:
There is an issue of using Flask with Keras: https://github.com/jrosebr1/simple-keras-rest-api/issues/1
The fix done (using tf.get_default_graph()) will probably not be valid for standalone wsgi container e.g. gunicorn,
gevent, uwsgi.
"""

import builtins
import glob
import json
import os
import re
import tempfile
import warnings
import zipfile
from collections import OrderedDict
from datetime import datetime
from functools import wraps

import numpy as np
import pkg_resources
import requests
import tensorflow as tf
from deepaas.model.v2.wrapper import UploadedFile
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from webargs import fields

from planktonclas import config, paths, test_utils, utils
from planktonclas.data_utils import (load_aphia_ids, load_class_info,
                                     load_class_names)
from planktonclas.train_runfile import train_fn

NOW = str("{:%Y_%m_%d_%H_%M_%S}".format(datetime.now()))
# print(NOW, ": Starting the process")
# Mount NextCloud folders (if NextCloud is available)
import os
from marshmallow import fields, ValidationError
from pathlib import Path
from marshmallow import ValidationError
# Empty model variables for inference (will be loaded the first time we
# perform inference)
loaded_ts, loaded_ckpt = None, None
graph, model, conf, class_names, class_info, aphia_ids = (
    None,
    None,
    None,
    None,
    None,
    None,
)

# Additional parameters
allowed_extensions = set(
    ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]
)  # allow only certain file extensions
top_K = 5  # number of top classes predictions to return


def load_inference_model(timestamp=None, ckpt_name=None):
    """
    Load a model for prediction.

    Parameters
    ----------
    * timestamp: str
        Name of the timestamp to use. The default is the last timestamp in `./models`.
    * ckpt_name: str
        Name of the checkpoint to use. The default is the last checkpoint in `./models/[timestamp]/ckpts`.
    """
    global loaded_ts, loaded_ckpt
    global graph, model, conf, class_names, class_info, aphia_ids

    # Set the timestamp
    timestamp_list = next(os.walk(paths.get_models_dir()))[1]
    timestamp_list = sorted(timestamp_list)
    if not timestamp_list:
        raise Exception(
            "You have no models in your `./models` folder to be used for inference. "
            "Therefore the API can only be used for training."
        )
    elif timestamp is None:
        timestamp = timestamp_list[-1]
    elif timestamp not in timestamp_list:
        raise ValueError(
            "Invalid timestamp name: {}. Available timestamp names are: {}".format(
                timestamp, timestamp_list
            )
        )
    paths.timestamp = timestamp
    print("Using TIMESTAMP={}".format(timestamp))

    # Set the checkpoint model to use to make the prediction
    ckpt_list = os.listdir(paths.get_checkpoints_dir())
    ckpt_list = sorted([name for name in ckpt_list if name.endswith(".h5")])
    if not ckpt_list:
        raise Exception(
            "You have no checkpoints in your `./models/{}/ckpts` folder to be used for inference. ".format(
                timestamp
            )
            + "Therefore the API can only be used for training."
        )
    elif ckpt_name is None:
        ckpt_name = ckpt_list[-1]
    elif ckpt_name not in ckpt_list:
        raise ValueError(
            "Invalid checkpoint name: {}. Available checkpoint names are: {}".format(
                ckpt_name, ckpt_list
            )
        )
    print("Using CKPT_NAME={}".format(ckpt_name))

    # Clear the previous loaded model
    K.clear_session()

    # Load the class names and info
    splits_dir = paths.get_ts_splits_dir()
    class_names = load_class_names(splits_dir=splits_dir)
    aphia_ids = load_aphia_ids(splits_dir)
    class_info = None
    if "info.txt" in os.listdir(splits_dir):
        class_info = load_class_info(splits_dir=splits_dir)
        if len(class_info) != len(class_names):
            warnings.warn(
                """The 'classes.txt' file has a different length than the 'info.txt' file.
            If a class has no information whatsoever you should leave that classes row empty or put a '-' symbol.
            The API will run with no info until this is solved."""
            )
            class_info = None
    if class_info is None:
        class_info = ["" for _ in range(len(class_names))]

    # Load training configuration
    conf_path = os.path.join(paths.get_conf_dir(), "conf.json")
    with open(conf_path) as f:
        conf = json.load(f)
        update_with_saved_conf(conf)

    # Load the model
    model = load_model(
        os.path.join(paths.get_checkpoints_dir(), ckpt_name),
        custom_objects=utils.get_custom_objects(),
    )
    graph = tf.get_default_graph()

    # model = load_model(
    #     os.path.join(paths.get_checkpoints_dir(), ckpt_name),
    #     custom_objects=utils.get_custom_objects(),
    # )
    # graph = tf.get_default_graph()
    # graph = tf.compat.v1.get_default_graph()
    # Set the model as loaded
    loaded_ts = timestamp
    loaded_ckpt = ckpt_name


def update_with_saved_conf(saved_conf):
    """
    Update the default YAML configuration with the configuration saved from training
    """
    # Update the default conf with the user input
    CONF = config.CONF
    for group, val in sorted(CONF.items()):
        if group in saved_conf.keys():
            for g_key, g_val in sorted(val.items()):
                if g_key in saved_conf[group].keys():
                    g_val["value"] = saved_conf[group][g_key]

    # Check and save the configuration
    config.check_conf(conf=CONF)
    config.conf_dict = config.get_conf_dict(conf=CONF)


def update_with_query_conf(user_args):
    """
    Update the default YAML configuration with the user's input args from the API query
    """
    # Update the default conf with the user input
    CONF = config.CONF
    for group, val in sorted(CONF.items()):
        for g_key, g_val in sorted(val.items()):
            if g_key in user_args:
                g_val["value"] = json.loads(user_args[g_key])

    # Check and save the configuration
    config.check_conf(conf=CONF)
    config.conf_dict = config.get_conf_dict(conf=CONF)


def catch_error(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            pred = f(*args, **kwargs)
            return {"status": "OK", "predictions": pred}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return wrap


def catch_url_error(url_list):
    # Error catch: Empty query
    if not url_list:
        raise ValueError("Empty query")

    for i in url_list:
        if not i.startswith(
            "data:image"
        ):  # don't do the checks for base64 encoded images
            # Error catch: Inexistent url
            try:
                url_type = requests.head(i).headers.get("content-type")
            except Exception:
                raise ValueError(
                    "Failed url connection: "
                    "Check you wrote the url address correctly."
                )

            # Error catch: Wrong formatted urls
            if url_type.split("/")[0] != "image":
                raise ValueError(
                    "Url image format error: Some urls were not in image format. "
                    "Check you didn't uploaded a preview of the image rather than the image itself."
                )


def catch_localfile_error(file_list):
    # Error catch: Empty query
    if not file_list:
        raise ValueError("Empty query")

    # Error catch: Image format error
    for f in file_list:
        extension = os.path.basename(f.content_type).split("/")[-1]
        # extension = mimetypes.guess_extension(f.content_type)
        if extension not in allowed_extensions:
            raise ValueError(
                "Local image format error: "
                "At least one file is not in a standard image format ({}).".format(
                    allowed_extensions
                )
            )


def warm():
    try:
        load_inference_model()
    except Exception as e:
        print(e)


def prepare_files(directory):
    """
    Prepare a list of dictionaries with attributes mimicking UploadedFile from image files in the directory.

    :param directory: The directory to search for image files.
    :return: A list of dictionaries with attributes similar to UploadedFile.
    """
    # Get all image files from the directory with given extensions
    extensions = ["*.jpg", "*.png", "*.jpeg"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))

    # Create a list of dictionaries with attributes similar to UploadedFile
    uploaded_files = []
    for file_path in files:
        # Extract the filename from the path
        file_name = os.path.basename(file_path)
        uploaded_files.append(
            UploadedFile(
                name="data",
                filename=file_path,
                content_type="image/jpeg",  # Adjust if necessary based on file type
                original_filename=file_name,
            )
        )
    return uploaded_files

def validate_directory(path):
    # Convert the input to a Path object if it's a string
    if isinstance(path, str):
        path = Path(path.strip('\'"'))  # Remove any leading/trailing quotes

    # Check if the path is a valid directory
    if not path.is_dir():
        raise ValidationError(f"{path} is not a valid directory")

    return path
from pathlib import Path


@catch_error
def predict(**args):
    if not any([args["image"], args["zip"]]):
        raise Exception(
            "You must provide either 'urls', 'image','file_location' or 'zip' in the payload"
        )

    if args["zip"]:
        # Check if zip file is provided
        zip_file = args["zip"]

        # Create a temporary directory to extract the files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the zip file
            with zipfile.ZipFile(zip_file.filename, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            # Get the list of files extracted from the zip
            folder_files = os.listdir(temp_dir)

            # Assign the list of files to args["files"]

            uploaded_files = []
            for file in folder_files:
                file_path = os.path.join(temp_dir, file)
                uploaded_files.append(
                    UploadedFile(
                        name="data",
                        filename=file_path,
                        content_type="image/jpeg",
                        original_filename=file,
                    )
                )

            # Assign the list of files to args["files"]
            args["files"] = uploaded_files

            # raise RuntimeError("zipped ", uploaded_files)

            # Call predict_data function (assuming it handles a list of files)
            return predict_data(args)
    elif args["image"]:
        args["files"] = [args["image"]]  # patch until list is available
        # raise RuntimeError("args files ", args["files"])
        print(args["files"])
        return predict_data(args)
    elif args['urls']:
        args['urls'] = [args['urls']]  # patch until list is available
        return predict_url(args)

    else:
        conf = config.conf_dict
        if conf["testing"]["predict_this"]=="yes":
            path='/srv/phyto-plankton-classification/data/predict_these'
        # path=validate_directory(args["file_location"])
        path = Path(path)
        # raise RuntimeError("popu ", args["file_location"])
        try:
            # List everything under the directory
            contents = list(path.glob("**/*"))  # Recursively get all files and directories
            if not contents:
                raise RuntimeError(f"Nothing found in {path}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while accessing {path}: {str(e)}")
        # raise RuntimeError(f"An error occurred while accessing {contents}") 
        args["files"] = prepare_files(path)
        # raise RuntimeError("args files ", args["files"])
        return predict_data(args)


def predict_url(args):
    """
    Function to predict an url
    """
    # Check user configuration
    update_with_query_conf(args)
    conf = config.conf_dict

    merge = True
    catch_url_error(args['urls'])

    # Load model if needed
    if loaded_ts != conf['testing']['timestamp'] or loaded_ckpt != conf['testing']['ckpt_name']:
        load_inference_model(timestamp=conf['testing']['timestamp'],
                             ckpt_name=conf['testing']['ckpt_name'])
        conf = config.conf_dict

    # Make the predictions
    with graph.as_default():
        pred_lab, pred_prob = test_utils.predict(model=model,
                                                 X=args['urls'],
                                                 conf=conf,
                                                 top_K=top_K,
                                                 filemode='url',
                                                 merge=merge,
                                                 use_multiprocessing=False)  # safer to avoid memory fragmentation in failed queries

    if merge:
        pred_lab, pred_prob = np.squeeze(pred_lab), np.squeeze(pred_prob)

    return format_prediction(pred_lab, pred_prob)

def predict_data(args):
    """
    Function to predict an image in binary format
    """
    # Check user configuration
    update_with_query_conf(args)
    conf = config.conf_dict

    merge = False
    catch_localfile_error(args["files"])

    if (
        loaded_ts != conf["testing"]["timestamp"]
        or loaded_ckpt != conf["testing"]["ckpt_name"]
    ):
        load_inference_model(
            timestamp=conf["testing"]["timestamp"],
            ckpt_name=conf["testing"]["ckpt_name"],
        )
        conf = config.conf_dict
    # Create a list with the path to the images
    filenames = [f.filename for f in args["files"]]
    print(filenames)
    original_filenames = [f.original_filename for f in args["files"]]
    # Make the predictions
    # try:
    with graph.as_default():
        pred_lab, pred_prob = test_utils.predict(
            model=model,
            X=filenames,
            conf=conf,
            top_K=top_K,
            filemode="local",
            merge=merge,
            use_multiprocessing=False,
        )  # safer to avoid memory fragmentation in failed queries
    # finally:
    #     for f in filenames:
    #         os.remove(f)

    if merge:
        pred_lab, pred_prob = np.squeeze(pred_lab), np.squeeze(pred_prob)

    return format_prediction(pred_lab, pred_prob, original_filenames)

def get_predictions_dir(CONF):
    file_location = CONF.get("testing", {}).get("file_location", None)
    output_directory = CONF["testing"]["output_directory"]
    # if file_location is None:
    #     if output_directory is None:
    #         # Define your get_timestamped_dir() function accordingly
    #         return os.path.join(get_timestamped_dir(), "predictions")
    #     else:
    #         return os.path.join(output_directory)
    if file_location is not None: 
        if os.path.exists(file_location):
            os.makedirs(os.path.join(os.path.dirname(file_location), "predictions"), exist_ok=True)
            return os.path.join(os.path.dirname(file_location), "predictions")
    else:
        if output_directory is None:
            # Define your get_timestamped_dir() function accordingly
            return os.path.join(paths.get_timestamped_dir(), "predictions")
        else:
            return os.path.join(output_directory)
        
def format_prediction(labels, probabilities, original_filenames):
    if aphia_ids is not None:
        pred_aphia_ids = [aphia_ids[i] for i in labels]
        pred_aphia_ids = [aphia_id.tolist() for aphia_id in pred_aphia_ids]
    else:
        pred_aphia_ids = aphia_ids
    class_index_map = {
        index: class_name for index, class_name in enumerate(class_names)
    }
    pred_lab_names = [[class_index_map[label] for label in labels] for labels in labels]
    # pred_labels=[class_names[i] for i in labels]
    pred_prob = probabilities

    pred_dict = {
        "filenames": list(original_filenames),
        "pred_lab": pred_lab_names,  # Use converted list
        "pred_prob": pred_prob.tolist(),
        "aphia_ids": pred_aphia_ids,
    }
    conf = config.conf_dict
    ckpt_name = conf["testing"]["ckpt_name"]
    split_name = "test"
    pred_path = os.path.join(
        get_predictions_dir(conf),
        "{}+{}+top{}.json".format(ckpt_name, split_name, top_K),
    )
    with open(pred_path, "w") as outfile:
        json.dump(pred_dict, outfile, sort_keys=True)

    return pred_dict


def train(**args):
    """
    Train an image classifier
    """
    update_with_query_conf(user_args=args)
    CONF = config.conf_dict
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    config.print_conf_table(CONF)
    K.clear_session()  # remove the model loaded for prediction
    train_fn(TIMESTAMP=timestamp, CONF=CONF)

    return {"modelname": timestamp}


def populate_parser(parser, default_conf):
    """
    Returns a arg-parse like parser.
    """
    print("popu parser")
    for group, val in default_conf.items():
        for g_key, g_val in val.items():
            gg_keys = g_val.keys()

            # Load optional keys
            help = g_val["help"] if ("help" in gg_keys) else ""
            type = getattr(builtins, g_val["type"]) if ("type" in gg_keys) else None
            choices = g_val["choices"] if ("choices" in gg_keys) else None

            # Additional info in help string
            help += "\n" + "<font color='#C5576B'> Group name: **{}**".format(
                str(group)
            )
            if choices:
                help += "\n" + "Choices: {}".format(str(choices))
            if type:
                help += "\n" + "Type: {}".format(g_val["type"])
            help += "</font>"

            # Create arg dict
            opt_args = {
                "missing": json.dumps(g_val["value"]),
                "description": help,
                "required": False,
            }
            if choices:
                opt_args["enum"] = [json.dumps(i) for i in choices]

            parser[g_key] = fields.Str(**opt_args)

    return parser


def get_train_args():
    parser = OrderedDict()
    default_conf = config.CONF
    default_conf = OrderedDict(
        [
            ("general", default_conf["general"]),
            ("model", default_conf["model"]),
            ("training", default_conf["training"]),
            ("monitor", default_conf["monitor"]),
            ("dataset", default_conf["dataset"]),
            ("augmentation", default_conf["augmentation"]),
        ]
    )

    return populate_parser(parser, default_conf)


def get_directory_choices(base_path="/srv/data/"):
    # Get a list of all directories in the base_path
    try:
        directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        return directories
    except Exception as e:
        print(f"Error accessing directories: {e}")
        return []

import os
from webargs import fields

def get_predict_args():
    parser = OrderedDict()
    default_conf = config.CONF
    default_conf = OrderedDict([("testing", default_conf["testing"])])

    # Add options for modelname
    timestamp = default_conf["testing"]["timestamp"]
    timestamp_list = next(os.walk(paths.get_models_dir()))[1]
    timestamp_list = sorted(timestamp_list)
    if not timestamp_list:
        timestamp["value"] = ""
    else:
        timestamp["value"] = timestamp_list[-1]
        timestamp["choices"] = timestamp_list

#     # Directory choices for file_location
#     data_dir = "/srv/phyto-plankton-classification/data"
#     file_location = default_conf["testing"].get("file_location", {})
    
#     # Walk through the /srv/data directory and collect subdirectories
#     subdirectories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
#     subdirectories = sorted(subdirectories)  # Optional: Sort the directory names

#     if not subdirectories:
#         file_location["value"] = None
#         file_location["choices"] = []
#     else:
#         file_location["value"] = subdirectories[0]  # Set default to the first directory
#         file_location["choices"] = subdirectories
    # parser["file_location"]=file_location["value"]
    parser["file_location"] = fields.Field(
        required=False,
        missing=None,
        data_key="file_location",
        description="Select the folder of the images you want to classify. For example: '/srv/phyto-plankton-classification/data/demo-images/Actinoptychus'",
        type="string",  # Use string type to indicate a path
    )

    # parser["file_location"] = fields.Field(
    #     required=False,
    #     missing='no',  # Default value
    #     enum=['yes', 'no'],  # Options for yes or no
    #     description="Indicate whether to use a file location (yes/no).",
    # )


        
    parser["image"] = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="image",
        location="form",
        description="Select the image you want to classify.",
    )

    parser["zip"] = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="zip_data",  # Unique data key for zip
        location="form",
        description="Select the ZIP file containing images you want to classify.",
    )

    # # Use field.String instead of field.Url because I also want to allow uploading of base 64 encoded data strings
    # parser['urls'] = fields.String(required=False,
    #                                missing=None,
    #                                description="Select an URL of the image you want to classify.")

    
    return populate_parser(parser, default_conf)


# def get_predict_args():
#     parser = OrderedDict()
#     default_conf = config.CONF
#     default_conf = OrderedDict([("testing", default_conf["testing"])])

#     # Add options for modelname
#     timestamp = default_conf["testing"]["timestamp"]
#     timestamp_list = next(os.walk(paths.get_models_dir()))[1]
#     timestamp_list = sorted(timestamp_list)
#     if not timestamp_list:
#         timestamp["value"] = ""
#     else:
#         timestamp["value"] = timestamp_list[-1]
#         timestamp["choices"] = timestamp_list



#     # parser["zip"] = fields.Field(
#     #     required=False,
#     #     missing=None,
#     #     type="file",
#     #     data_key="zip_data",  # Unique data key for zip
#     #     location="form",
#     #     description="Select the ZIP file containing images you want to classify.",
#     # )

#     directory_choices = get_directory_choices()
#     parser["file_location"] = fields.Field(
#         required=False,
#         missing=None,
#         data_key="file_location",
#         description="Select the folder of the images you want to classify. For example: '/srv/phyto-plankton-classification/data/demo-images/Actinoptychus'",
#         validate=validate_directory,
#         choices=directory_choices  # Provide the list of directory choices
#     )

# #     parser["image"] = fields.Field(
# #         required=False,
# #         missing=None,
# #         type="file",
# #         data_key="image",
# #         location="form",
# #         description="Select the image you want to classify.",
# #     )
        
#     # parser["file_location"] = fields.Field(
#     #     required=False,
#     #     missing=None,
#     #     data_key="file_location",
#     #     description="Select the folder of the images you want to classify. For example: /srv/phyto-plankton-classification/data/demo-images/Actinoptychus",
#     # )
#     # raise RuntimeError("popu ", parser["file_location"])
#     return populate_parser(parser, default_conf)


def get_metadata(distribution_name="planktonclas"):
    """
    Function to read metadata
    """

    pkg = pkg_resources.get_distribution(distribution_name)
    meta = {
        "name": None,
        "version": None,
        "summary": None,
        "home-page": None,
        "author": None,
        "author-email": None,
        "license": None,
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        line_low = line.lower()  # to avoid inconsistency due to letter cases
        for par in meta:
            if line_low.startswith(par.lower() + ":"):
                _, value = line.split(": ", 1)
                meta[par] = value

    # Update information with Docker info (provided as 'CONTAINER_*' env
    # variables)
    r = re.compile("^CONTAINER_(.*?)$")
    container_vars = list(filter(r.match, list(os.environ)))
    for var in container_vars:
        meta[var.capitalize()] = os.getenv(var)

    return meta


schema = {
    "status": fields.Str(),
    "message": fields.Str(),
    "predictions": fields.Field(),
}
