"""
Mask R-CNN
Configurations and data loading to use MS COCO to spot people

Copyright (c) 2019 C. H. Pratten
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import fnmatch
import sys
import time
import random
import numpy as np
import logging
import imgaug  # https://github.com/aleju/imgaug (pip3 install imgaug)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import json
import skimage.color
import skimage.io
import skimage.transform
# to import matlab file
import scipy.io
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import keras.activations as KA
import keras.utils as KU


# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils as utils
from tf_smpl.batch_lbs import batch_rodrigues

from abstract_data import *
from coco_data import *
from mpii_data import *
from lsp_data import *
from occlude_data import *

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

############################################################
#  Configurations
############################################################


class HumanConfig(Config):
    """Configuration for training on MS COCO to match humans.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "human"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 2 # // 1 + 80  # COCO has 80 classes

############################################################
#  Data
############################################################

class H36Data:
    
    
    def __init__(self,dataset_dir, subset):
        pruint("TODO")

############################################################
#  Dataset
############################################################

class DensePoseDataSet(utils.Dataset):
    
    def __init__(self, image_data):
        utils.Dataset.__init__(self)
        self.image_data = image_data

    def load_prepare(self):
        self.add_class("data", self.image_data.person_cat['id'], self.image_data.person_cat["name"])
        self.image_set = self.image_data.load_images()
        image_count = 0
        for image in self.image_set.images:
            if image.has_dp_data:
                self.add_image('data', image_count, image)
                image_count = image_count+1
        self.prepare()

    def load_image(self, image_id):
        image = self.image_set.images[image_id]
        return image.image_data.read_image()

    def merge_occlusion(mask, occlusion_mask):
        mask[occlusion_mask] = 2
        return mask
    
    def load_mask(self, image_id):
        image = self.image_set.images[image_id]
        occlusion_mask = image.create_occlusion_mask()
        person_class = self.map_source_class_id("data.{}".format(self.image_data.person_cat['id']))
        mask = np.stack([DensePoseDataSet.merge_occlusion(person.regions_to_mask(image), occlusion_mask) for person in image.people if person.regions is not None], axis=2).astype(np.bool)
        class_ids = np.array([person_class for person in image.people if person.regions is not None], dtype=np.int32)
        return mask, class_ids
        
        
class HumanDataset(utils.Dataset):

    def __init__(self):
        utils.Dataset.__init__(self)
        
    def load_prepare(self):
        raise Exception("Expecting sub class")

    def evaluate(self):
        raise Exception("Expecting sub class")

    def show_example(self, model):
        raise Exception("Expecting sub class") 
    
class CocoHumanDataset(HumanDataset):
    
    def __init__(self, year=DEFAULT_DATASET_YEAR):
        HumanDataset.__init__(self)
        self.year = year
        
    def evaluate(self, model, limit=500) :
        evaluate_coco(model, self, self.coco, "bbox", limit=int(limit))

        
    def load_prepare(self,
                     dataset_dir, 
                     is_train, 
                     return_lib_object=False, 
                     auto_download=False):
        if is_train:
            subset = "train"
        else:
            subset = "val"
        self.load_coco(dataset_dir,
                       subset,
                       self.year,
                       auto_download = auto_download)
        self.prepare()
        
    def load_coco(self, 
                  dataset_dir, 
                  subset, 
                  year=DEFAULT_DATASET_YEAR, 
                  auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        self.dataset_dir = dataset_dir
        self.coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        self.person_keypoints = COCO("{}/annotations/person_keypoints_{}{}.json".format(dataset_dir, subset, year))
        self.panoptic = COCO("{}/annotations/panoptic_{}{}.json".format(dataset_dir, subset, year))

        image_dir = "{}/images/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        class_ids = sorted(self.coco.getCatIds(catNms=["person"]))

        # All images or a subset?
        image_ids = []
        for id in class_ids:
            image_ids.extend(list(self.coco.getImgIds(catIds=[id])))
            # Remove duplicates
        image_ids = list(set(image_ids))

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, self.coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco",
                image_id=i,
                path=os.path.join(image_dir, self.coco.imgs[i]['file_name']),
                width=self.coco.imgs[i]["width"],
                height=self.coco.imgs[i]["height"],
                annotations=self.coco.loadAnns(self.coco.getAnnIds(imgIds=[i],
                                                                   catIds=class_ids,
                                                                   iscrowd=None)),
                keypoints=self.person_keypoints.loadAnns(self.person_keypoints.getAnnIds(imgIds=[i],
                                                                                         catIds=class_ids,
                                                                                         iscrowd=None)),
                panoptic=self.panoptic.loadAnns([i])
            )

    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        dataDir: The root directory of the COCO dataset.
        dataType: What to load (train, val, minival, valminusminival)
        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """

        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(HumanDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(HumanDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(HumanDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  H36 evaluation
############################################################

############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results

def show_coco_examples(model, dataset, coco, image_ids=None):
    print("Show coco examples")

def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

############################################################
#  Model
############################################################

# From HMR paper
def Discriminator_separable_rotations(
        poses,
        shapes,
        weight_decay,
):
    """
    23 Discriminators on each joint + 1 for all joints + 1 for shape.
    To share the params on rotations, this treats the 23 rotation matrices
    as a "vertical image":
    Do 1x1 conv, then send off to 23 independent classifiers.

    Input:
    - poses: N x 23 x 1 x 9, NHWC ALWAYS!!
    - shapes: N x 10
    - weight_decay: float

    Outputs:
    - prediction: N x (1+23) or N x (1+23+1) if do_joint is on.
    - variables: tf variables
    """
    data_format = "NHWC"
    with tf.name_scope("Discriminator_sep_rotations", [poses, shapes]):
        with tf.variable_scope("D") as scope:
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                    weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([slim.conv2d], data_format=data_format):
                    poses = slim.conv2d(poses, 32, [1, 1], scope='D_conv1')
                    poses = slim.conv2d(poses, 32, [1, 1], scope='D_conv2')
                    theta_out = []
                    for i in range(0, 23):
                        theta_out.append(
                            slim.fully_connected(
                                poses[:, i, :, :],
                                1,
                                activation_fn=None,
                                scope="pose_out_j%d" % i))
                    theta_out_all = tf.squeeze(tf.stack(theta_out, axis=1))

                    # Do shape on it's own:
                    shapes = slim.stack(
                        shapes,
                        slim.fully_connected, [10, 5],
                        scope="shape_fc1")
                    shape_out = slim.fully_connected(
                        shapes, 1, activation_fn=None, scope="shape_final")
                    """ Compute joint correlation prior!"""
                    nz_feat = 1024
                    poses_all = slim.flatten(poses, scope='vectorize')
                    poses_all = slim.fully_connected(
                        poses_all, nz_feat, scope="D_alljoints_fc1")
                    poses_all = slim.fully_connected(
                        poses_all, nz_feat, scope="D_alljoints_fc2")
                    poses_all_out = slim.fully_connected(
                        poses_all,
                        1,
                        activation_fn=None,
                        scope="D_alljoints_out")
                    out = tf.concat([theta_out_all,
                                     poses_all_out, shape_out], 1)

            variables = tf.contrib.framework.get_variables(scope)
            return out, variables


def humans_build_fpn_mask_graph(rois, feature_maps, image_meta,
                                pool_size, num_classes, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = modellib.PyramidROIAlign([pool_size, pool_size],
                                 name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Conv layers
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)

    x =  KL.TimeDistributed(KL.Conv2D(num_classes*3, (1, 1), strides=1), name="mrcnn_mask")(x)
    
    def mcrnn_mask_softmax(t):
        t_shape = tf.shape(t)
        t = tf.reshape(t,[-1,t_shape[1], t_shape[2], num_classes,3])
        t = KA.softmax(t, axis=3)
        return t
    x = KL.TimeDistributed(KL.Lambda(mcrnn_mask_softmax))(x)

    # Classes are
    #      0 background
    #      1 foreground (occluding figure)
    #      2 human clothing
    #      3 underlying human shape
    # x = KL.TimeDistributed(KL.Conv2D(4, (1, 1), strides=1),
    #                        name="mrcnn_mask")(x)
    # x = KL.Activation('softmax')(x)

    """
    real_rotations = batch_rodrigues(tf.reshape(self.pose_loader, [-1, 3]))
    real_rotations = tf.reshape(real_rotations, [-1, 24, 9])
    # Ignoring global rotation. N x 23*9
    # The # of real rotation is B*num_stage so it's balanced.
    real_rotations = real_rotations[:, 1:, :]
    all_fake_rotations = tf.reshape(
        tf.concat(fake_rotations, 0),
        [self.batch_size * self.num_stage, -1, 9])
    comb_rotations = tf.concat(
        [real_rotations, all_fake_rotations], 0, name="combined_pose")
    
    comb_rotations = tf.expand_dims(comb_rotations, 2)
    all_fake_shapes = tf.concat(fake_shapes, 0)
    comb_shapes = tf.concat(
        [self.shape_loader, all_fake_shapes], 0, name="combined_shape")

    disc_input = {
        'weight_decay': self.d_wd,
        'shapes': comb_shapes,
        'poses': comb_rotations
    }

    d_out, d_var = Discriminator_separable_rotations(
            **disc_input)
    """
    return x

def custom_mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0, 1 or 2. Uses zero padding to fill array. (2 refers to an occlusion)
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    roi_occlusion_masks [batch, num_rois, height, width]
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = K.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = K.reshape(pred_masks,
                           (-1, pred_shape[2], pred_shape[3], pred_shape[4], pred_shape[5]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2, 4])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
         tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_mask = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)
    y_true = tf.one_hot(tf.cast(y_mask, 'uint8'), depth=3)

    logging.debug("Shape y_true %s, y_pred %s", tf.shape(y_true), tf.shape(y_pred))
    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    # loss = K.switch(tf.size(y_true) > 0,
    #                 K.binary_crossentropy(target=y_true, output=y_pred),
    #                 tf.constant(0.0))
    loss = K.switch(tf.size(y_true) > 0,
                    K.categorical_crossentropy(target=y_true, output=y_pred),
                    tf.constant(0.0))
    loss = K.mean(loss)
    return loss

    

############################################################
#  Training
############################################################

def add_common_args(parser):
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)                        

def add_data_common_args(parser):
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the dataset')
    parser.add_argument('--data-type', required=False,
                        choices=["COCO", "H36", "MPII", "LSP"],
                        default="COCO",
                        metavar="<COCO|H36|MPII|LSP>",
                        help='Type of the input data')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--occlude-data-path', required=False,
                        metavar="<path to VOC data>",
                        help='Path to pascal VOC data for occlusion')
    
def load_data(data_type, dataset, subset, occlude_data_path, **kwargs):
    logging.debug("Loading data of type %sm=, subset %s from location %s", data_type, subset, dataset)
    if data_type.upper() == "COCO":
        data = CocoData(dataset,subset,kwargs['year'])
    elif False:
        if args.data_type.upper() == "MPII":
            data = MpiiData(args.dataset,subset)
        elif args.data_type.upper() == "LSP":
            data = LspData(args.dataset,subset)
    else:
        raise Exception("Unkown data {}".format(args.data_type))
    if occlude_data_path != None:
        logging.debug("Adding occlusions to the data")
        occluder = DataOccluder(occlude_data_path)
        data = OccludedData(occluder, data)
    return data

def get_data(args):
    if data_type.upper() == "COCO":
        CocoData.get_data(args.dataset, year=args.year, train_data=args.train_data)
    else:
        logging.error("Dont know how to get data for {}".format(data_type))

    if data_type.upper() == "VOD":
        print("http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar")

def show_examples(args):
    logging.debug("Performing action show-examples")
    data=load_data(**dict(args.__dict__))
    images = data.load_images()
    while True:
        image = random.choice(images.images)
        person_index = random.randint(0,len(image.people)-1)
        fig = plt.figure(figsize=(15,10))
        gs = gridspec.GridSpec(2, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[0, 1])
        ax4 = plt.subplot(gs[1, 1])
        logging.info("Showing image {}".format(image))
        plt.sca(ax1)
        image.show_image(person_index=person_index)
        image.show_regions(person_index=person_index)
        image.show_joints(person_index=person_index)
        plt.sca(ax2)
        image.show_image(person_index=person_index)
        image.show_dp_data(person_index=person_index)
        plt.sca(ax3)
        image.show_image(person_index=person_index)
        plt.sca(ax4)
        image.show_image(person_index=person_index)
        image.show_occlusions(person_index=person_index)
        plt.show()
        plt.close()

def train(args):
    logging.info("training the model")
    train_args   = dict(args.__dict__, subset=args.train_subset)
    val_args     = dict(args.__dict__, subset=args.val_subset)
    data_train   = load_data(**train_args)
    data_val     = load_data(**val_args) 
    config       = HumanConfig()
    config.display()
    # build the model
    model = modellib.MaskRCNN(mode="training",
                              config=config,
                              model_dir=args.logs,
                              custom_build_fpn_mask_graph=humans_build_fpn_mask_graph,
                              custom_mrcnn_mask_loss_graph=custom_mrcnn_mask_loss_graph)
    if args.model.lower() == "last":
            # Find last trained weights
        model_path = model.find_last()
    else:
        model_path = args.model

    dataset_train = DensePoseDataSet(data_train)
    dataset_train.load_prepare()
    logging.info("Training data set has {} images".format(len(dataset_train.image_info)))
    
    dataset_val = DensePoseDataSet(data_val)
    dataset_val.load_prepare()
    logging.info("Validation data set has {} images".format(len(dataset_val.image_info)))
    
    print("Loading weights ", model_path)
    model.load_weights(model_path,
                       by_name=True,
                       exclude=args.model_exclude)

    
    
    # Training - Stage 1
    logging.info("Training network heads")
    model.train(dataset_train,
                dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=13,
                layers='heads',
                augmentation=None,
                use_multiprocessing=args.use_multiprocessing)
    
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    logging.info("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=15,
                layers='4+',
                augmentation=None)
    
    # Training - Stage 3
    # Fine tune all layers
    logging.info("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=18,
                layers='all',
                augmentation=None)

def eval_image(args):
    # build the model
    config       = HumanConfig()
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=args.logs,
                              custom_build_fpn_mask_graph=humans_build_fpn_mask_graph)
    if args.model.lower() == "last":
            # Find last trained weights
        model_path = model.find_last()
    else:
        model_path = args.model

    logging.info("Loading weights %s", model_path)
    model.load_weights(model_path,
                       by_name=True)

    image = ImageFile(args.image_path).read_image()
    results = model.detect([image], verbose=1)[0]
    final_rois = results["rois"]
    final_class_ids = results["class_ids"]
    final_scores = results["scores"]
    final_masks = results["masks"]
    
    logging.debug("Number of ROI found %s", len(final_rois))

    fig = plt.figure(figsize=(15,10))
    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0, 0])
    plt.sca(ax1)
    plt.imshow(image)

    for i in range(len(final_rois)):
        roi = final_rois[i]
        class_id = final_class_ids[i]
        score  = final_scores[i]
        mask   = final_masks[:,:,i]
        mask_bool = mask!=0
        mask_colour = np.zeros((mask.shape[0], mask.shape[1],4))
        mask_colour[:,:,0][mask_bool] = 1.0
        mask_colour[:,:,1][mask_bool] = 0.0
        mask_colour[:,:,2][mask_bool] = 1.0
        mask_colour[:,:,3][mask_bool] = 0.8
        logging.debug("Mask %s Class id %s with score %s", i, class_id, score) 
        plt.imshow(mask_colour)
    plt.show()
    plt.close()

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')

    subparsers = parser.add_subparsers()

    get_data_parser = subparsers.add_parser("get-data")
    get_data_parser.add_argument('--train-data', dest="train_data", action="store_true")
    get_data_parser.set_defaults(train_data=False)
    get_data_parser.set_defaults(func=get_data)
    add_common_args(get_data_parser)
    add_data_common_args(get_data_parser)
    
    show_examples_parser = subparsers.add_parser("show-examples")
    show_examples_parser.add_argument('--subset', required=False,
                                      choices=["train", "val", "minival", "valminusminival"],
                                      default=None,
                                      metavar="<train|val|minival|valminusminival>",
                                      help='whether to use training or vaildation set')
    show_examples_parser.set_defaults(func=show_examples)
    add_common_args(show_examples_parser)
    add_data_common_args(show_examples_parser)
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument('--train-subset', required=False,
                              choices=["train", "val", "minival", "valminusminival"],
                              default=None,
                              metavar="<train|val|minival|valminusminival>",
                              help='whether to use training or vaildation set')
    train_parser.add_argument('--val-subset', required=False,
                              choices=["train", "val", "minival", "valminusminival"],
                              default=None,
                              metavar="<train|val|minival|valminusminival>",
                              help='whether to use training or vaildation set')
    train_parser.add_argument('--model', required=True,
                              metavar="/path/to/weights.h5",
                              help="Path to weights .h5 file or 'coco'")
    train_parser.add_argument('--model-exclude',
                              nargs='*',
                              metavar="exclude model weights when loading model",
                              help="Exclude model weights when loading. Only use in train mode")
    train_parser.add_argument('--logs', required=False,
                              default=DEFAULT_LOGS_DIR,
                              metavar="/path/to/logs/",
                              help='Logs and checkpoints directory (default=logs/)')
    train_parser.add_argument('--limit', required=False,
                              default=500,
                              metavar="<image count>",
                              help='Images to use for evaluation (default=500)')
    train_parser.add_argument('--use-multiprocessing', dest='use_multiprocessing', action='store_true')
    train_parser.add_argument('--no-multiprocessing', dest='use_multiprocessing', action='store_false')
    train_parser.set_defaults(use_multiprocessing=True)
    train_parser.set_defaults(func=train)
    add_common_args(train_parser)
    add_data_common_args(train_parser)

    eval_image_parser = subparsers.add_parser("eval-image")
    add_common_args(eval_image_parser)    
    eval_image_parser.add_argument('--model', required=True,
                                   metavar="/path/to/weights.h5",
                                   help="Path to weights .h5 file or 'coco'")
    eval_image_parser.add_argument('--logs', required=False,
                                   default=DEFAULT_LOGS_DIR,
                                   metavar="/path/to/logs/",
                                   help='Logs and checkpoints directory (default=logs/)')
    eval_image_parser.add_argument('--image-path', required=True,
                                   metavar="/path/to/image",
                                   help="Path to image to evaluate")
    eval_image_parser.set_defaults(func=eval_image)
    args = parser.parse_args()
    FORMAT = '%(asctime)-15s %(levelname)s %(name)s : %(message)s'
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format=FORMAT)
        logging.debug("Running humans [DEBUG]")
    else:
        logging.basicConfig(level=logging.INFO, format=FORMAT)
        logging.info("Running humans")
        
    logging.debug("Parsed args %s", args)
    args.func(args)
    """
        # Configurations
        if args.command == "train":
            config = HumanConfig()
        else:
            class InferenceConfig(HumanConfig):
                # Set batch size to 1 since we'll be running inference on
                # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
                GPU_COUNT = 1
                IMAGES_PER_GPU = 1
                DETECTION_MIN_CONFIDENCE = 0
                
            config = InferenceConfig()
        config.display()
                
        # Create model
        if args.command == "train":
            model = modellib.MaskRCNN(mode="training",
                                      config=config,
                                      model_dir=args.logs,
                                      custom_build_fpn_mask_graph=humans_build_fpn_mask_graph)
        else:
            model = modellib.MaskRCNN(mode="inference",
                                      config=config,
                                      model_dir=args.logs,
                                      custom_build_fpn_mask_graph=humans_build_fpn_mask_graph)

        # Select weights file to load
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = model.get_imagenet_weights()
        else:
            model_path = args.model

        # Load weights
        print("Loading weights ", model_path)
        model.load_weights(model_path,
                           by_name=True,
                           exclude=["mrcnn_bbox_fc",
                                    "mrcnn_bbox",
                                    "mrcnn_mask",
                                    "mrcnn_class_logits",
                                    "mrcnn_class"
                           ])

    
    if args.data_type == "COCO":
        dataset = CocoHumanDataset(args.year)
    elif  args.data_type == "H36":
        dataset = H36HumanDataset()
    else:
        raise Exception("Unknown data type {typ}".format({typ:args.data_type}))
    
    if args.command == "train":
        is_train = True
    else:
        is_train = False
        
    dataset.load_prepare(args.dataset,
                         is_train,
                         auto_download=args.download)
    
    # Train or evaluate
    if args.command == "train":

        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset.coco,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset.coco,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)
        if False:
           # Training - Stage 3
           # Fine tune all layers
           print("Fine tune all layers")
           model.train(dataset_train, dataset_val,
                       learning_rate=config.LEARNING_RATE / 10,
                       epochs=160,
                       layers='all',
                       augmentation=augmentation)

    elif args.command == "evaluate":
        # Validation dataset
        print("Running COCO evaluation on {} images.".format(args.limit))
        if args.data_type == "COCO":
            dataset.evaluate(model, dataset.coco, limit=int(args.limit))
        else:
            raise Exception("Evaluation not implemented for {}".format(args.data_type))
    elif args.command == "show-examples":
        print("Running COCO evaluation on {} images.".format(args.limit))
        if args.data_type == "COCO":
            dataset.show_examples(model)
        else:
            raise Exception("Evaluation not implemented for {}".format(args.data_type))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
    """

