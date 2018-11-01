
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
import os
import fnmatch
import sys
import time
import numpy as np
import json
import wget

from abstract_data import *

############################################################
#  Coco Data
############################################################

class CocoData(object):
    """
    Coco data set support - utilises cocoapi. Not a great dataset for humans. For example no way of telling what is background and what is occlusion
    """

    def __init__(self,dataset_dir, subset, year):
        self.dataset_dir = dataset_dir
        self.year = year
        self.subset = subset
        if self.subset == 'minival' or self.subset == 'valminusminival':
            self.image_subset = 'val'
        else:
            self.image_subset = self.subset
        instance_path   = os.path.join(dataset_dir, "annotations", "instances_{}{}.json".format(self.subset, self.year))
        keypoints_path  = os.path.join(dataset_dir, "annotations", "person_keypoints_{}{}.json".format(self.subset, self.year))
        denspose_path   = os.path.join(dataset_dir, "annotations", "densepose_coco_{}_{}.json".format(self.year, self.subset))
        panoptic_path   = os.path.join(dataset_dir, "annotations", "panoptic_{}{}.json".format(self.subset, self.year))
        if os.path.exists(denspose_path):
            self.densepose    = COCO(denspose_path)
            self.person_keypoints = self.densepose
            self.coco = self.densepose
        elif os.path.exists(keypoints_path):
            self.person_keypoints = COCO(keypoints_path)
            self.coco = self.person_keypoints
        elif os.path.exists(instance_path):
            self.coco = COCO(instance_path)
        else:
            print("No coco data exists at {}".format(instance_path)) 
        if os.path.exists(panoptic_path):
            self.panoptic = COCO("{}/annotations/panoptic_{}{}.json".format(dataset_dir, subset, year))
        else:
            print("No coco data exists at {}".format(panoptic_path))

        self.person_cat = self.coco.loadCats(ids = self.coco.getCatIds(catNms=['person']))[0]

    _JOINT_MAP = {
        'nose': JointPositions.NOSE,
        'left_eye': JointPositions.LEFT_EYE,
        'right_eye': JointPositions.RIGHT_EYE,
        'left_ear': JointPositions.LEFT_EAR,
        'right_ear': JointPositions.RIGHT_EAR,
        'left_shoulder': JointPositions.LEFT_SHOULDER,
        'right_shoulder': JointPositions.RIGHT_SHOULDER,
        'left_elbow': JointPositions.LEFT_ELBOW,
        'right_elbow': JointPositions.RIGHT_ELBOW,
        'left_wrist': JointPositions.LEFT_WRIST,
        'right_wrist': JointPositions.RIGHT_WRIST,
        'left_hip': JointPositions.LEFT_HIP,
        'right_hip': JointPositions.RIGHT_HIP,
        'left_knee': JointPositions.LEFT_KNEE,
        'right_knee': JointPositions.RIGHT_KNEE,
        'left_ankle': JointPositions.LEFT_ANKLE,
        'right_ankle': JointPositions.RIGHT_ANKLE
        }

    def load_images(self):
        image_objs = []
        image_ids = self.coco.getImgIds(catIds = [self.person_cat['id']])
        image_anns = {}
        # build keypoints map
        keypoint_names = self.person_cat['keypoints']
        joint_names = [CocoData._JOINT_MAP[key_name] for key_name in keypoint_names]
        for ann_id in self.coco.getAnnIds(catIds = self.person_keypoints.getCatIds(catNms=['person'])) :
            ann = self.coco.anns[ann_id]
            if ann['image_id'] not in image_anns:
                image_anns[ann['image_id']] = [ ann ]
            else:
                image_anns[ann['image_id']].append(ann)                
        for image_id in image_ids:
            image = self.coco.imgs[image_id]
            image_path = os.path.join(self.dataset_dir, 'images', "{}{}".format(self.image_subset, self.year), image['file_name'])
            people = []
            for ann in image_anns[image_id]:
                bbox =  ann['bbox']
                segmentation = ann['segmentation']
                if isinstance(segmentation, list) :
                    def map_segment(segment):
                        segment = np.array(segment)
                        return segment.reshape(-1,2)
                    regions = list(map(map_segment, segmentation))
                else:
                    # could be counts
                    regions = None
                if 'keypoints' in ann:
                    keypoints = np.array(ann['keypoints']).reshape((-1,3))
                    joints = []
                    for index in range(len(keypoint_names)):
                        keypoint = keypoints[index]
                        v = keypoint[2]
                        if v == 0:
                            continue
                        else:
                            x = keypoint[0]
                            y = keypoint[1]
                            position = joint_names[index]
                            joints.append(Joint(position, x, y, v==1))
                else:
                    joints = None
                if 'dp_masks' in ann:
                    mask_polys = ann['dp_masks']
                    dp_mask = np.zeros([256,256])
                    for i in range(1,15):
                        if(mask_polys[i-1]):
                            current_mask = mask_utils.decode(mask_polys[i-1])
                            dp_mask[current_mask>0] = i
                    dp_data = {
                        'dp_x':ann['dp_x'],
                        'dp_y':ann['dp_y'],
                        'dp_I':ann['dp_I'],
                        'dp_U':ann['dp_U'],
                        'dp_V':ann['dp_V']
                    }
                else:
                    dp_data = None
                    dp_mask = None
                people.append(Person(joints=joints,regions=regions, bbox=bbox, dp_data=dp_data, dp_mask = dp_mask))
            image_obj = Image(ImageFile(image_path),
                              height=image['height'],
                              width=image['width'],
                              people=people)
            image_objs.append(image_obj)
        return ImageSet(image_objs)

    def download_if_missing(dataset_dir, download):
        if not os.exists(os.join(dataset_dir, os.path.basename(download))):
            wget.download(download, os.path.join(dataset_dir,"/."))
        else
            print("No download required {} exists".format(os.path.join(dataset_dir, os.path.basename(download))))

    def get_data(dataset_dir, year, train_data=False):
        print("Downloading missing data for coco year {} [train_data={})".format(year, train_data))
        if year == 2014:
            for download in [ "https://s3.amazonaws.com/densepose/densepose_coco_2014_train.json",
                              "https://s3.amazonaws.com/densepose/densepose_coco_2014_valminusminival.json",
                              "wget https://s3.amazonaws.com/densepose/densepose_coco_2014_minival.json",
                              "wget https://s3.amazonaws.com/densepose/densepose_coco_2014_test.json"
            ] :
                download_if_missing(dataset_dir, download)
    
