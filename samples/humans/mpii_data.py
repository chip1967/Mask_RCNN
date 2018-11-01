
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import os
import fnmatch
import sys
import time
import numpy as np
import json

from abstract_data import *

############################################################
#  Mpii Data
############################################################

class MpiiData(object):

    _COMMON_JOINT_IDS = [
            0,  # R ankle
            1,  # R knee
            2,  # R hip
            3,  # L hip
            4,  # L knee
            5,  # L ankle
            10,  # R Wrist
            11,  # R Elbow
            12,  # R shoulder
            13,  # L shoulder
            14,  # L Elbow
            15,  # L Wrist
            8,  # Neck top
            9,  # Head top
        ]

    _JOINT_CONNECTIONS = [
        [0, 1],  # R ankle R knee
        [5, 4],  # L ankle L knee
        [1, 2],  # R knee R hip
        [4, 3],  # L knee L hip

        [10, 11], # L wrist L elbow
        [15, 14], # R wrist R elbow
        [11, 12], # L elbow L shoulder
        [14, 13], # R elbow R shoulder
        [12, 8], # R shoulder Neck top
        [13, 8], # L shoulder Neck top
        [2, 8], # R hip Neck top
        [3, 8], # L hip Neck top
        [8, 9] # Neck top Head top
        ]

    def __init__(self, dataset_dir, subset):
        self.dataset_dir = dataset_dir
        self.subset = subset
        # find file name
        ann_dir = os.path.join(dataset_dir, 'annotations')
        img_dir = os.path.join(dataset_dir, 'images')
        pos_ann_files = [file for file in os.listdir(ann_dir) if fnmatch.fnmatch(file, 'mpii_human_pose_v*.mat') ]
        pos_ann_files.sort()
        if len(pos_ann_files) == 0:
            raise Exception("No annotation files in {}".format(ann_dir))
        ann_file = os.path.join(ann_dir, pos_ann_files[0])
        file = scipy.io.loadmat(ann_file, struct_as_record=False, squeeze_me=True)
        anno = file['RELEASE']
        # process the annotations - inspired by https://github.com/akanazawa/hmr/blob/master/src/datasets/mpii_to_tfrecords.py
        all_ids = np.array(range(len(anno.annolist)))
        if subset == "train":
            img_inds = all_ids[anno.img_train.astype('bool')]
        else:
            raise Exception("Cant use val data as no body positions!")
            img_inds = all_ids[np.logical_not(anno.img_train)]
        print("We have {0} images".format(len(img_inds)))
        self.annotations = []
        for img_id in img_inds:
            anno_info = anno.annolist[img_id]
            single_persons = anno.single_person[img_id]
            if not isinstance(single_persons, np.ndarray):
                single_persons = np.array([single_persons])
            people = self.parse_people(img_dir, anno_info, single_persons)
            if len(people) != 0:
                self.annotations.append({
                    'id': img_id,
                    'image_path':os.path.join(img_dir, anno_info.image.name),
                    'people':people
                })
        print("{0} annotation processed".format(len(self.annotations)))

    def convert_is_visible(self,is_visible):
        """
        this field is u'1' or empty numpy array..
        """
        if isinstance(is_visible, np.ndarray):
            assert (is_visible.size == 0)
            return 0
        else:
            return int(is_visible)
        
    def read_joints(self,rect):
        """
        Reads joints in the common joint order.
        Assumes rect has annopoints as field.
        Returns:
        joints: 3 x |common joints|
        """
        # Mapping from MPII joints to LSP joints (0:13). In this roder:
        assert ('annopoints' in rect._fieldnames)
        points = rect.annopoints.point
        if not isinstance(points, np.ndarray):
            # There is only one! so ignore this image
            return None
        # Not all joints are there.. read points in a dict.
        read_points = {}
        joints = []

        for point in points:
            vis = self.convert_is_visible(point.is_visible)
            joints.append({
                'id': point.id,
                'x': point.x,
                'y': point.y,
                'visible':vis
            })
        return joints
                

    def parse_people(self, img_dir, anno_info, single_persons):
        '''
        inspired by https://github.com/akanazawa/hmr/blob/master/src/datasets/mpii_to_tfrecords.py
        Parses people from rect annotation.
        Assumes input is train data.
        Input:
        img_dir: str
        anno_info: annolist[img_id] obj
        single_persons: rect id idx for "single" people
        Returns:
        people - list of annotated single-people in this image.
        Its Entries are tuple (label, img_scale, obj_pos)
        '''
        # No single persons in this image.
        if single_persons.size == 0:
            return []

        rects = anno_info.annorect
        if not isinstance(rects, np.ndarray):
            rects = np.array([rects])

        # Read each human:
        people = []
        for ridx in single_persons:
            rect = rects[ridx - 1]
            pos = np.array([rect.objpos.x, rect.objpos.y])
            joints = self.read_joints(rect)
            if joints is None:
                continue
            # Compute the scale using the keypoints so the person is 150px.
            visible = np.zeros(1+np.max(MpiiData._COMMON_JOINT_IDS)).astype(bool)
            joints_array = [None] * len(visible)
            for joint in joints:
                visible[joint['id']] = joint['visible']
                joints_array[joint['id']] = joint
            # If ankles are visible
            if visible[0] or visible[5]:
                min_pt = np.min([joint['y'] for joint in joints])
                max_pt = np.max([joint['y'] for joint in joints])
                person_height = np.linalg.norm(max_pt - min_pt)
                scale = 150. / person_height
            else:
                # Torso points left should, right shold, right hip, left hip
                # torso_points = joints[:, [8, 9, 3, 2]]
                torso_heights = []
                if visible[13] and visible[2]:
                    torso_heights.append(
                        np.linalg.norm(joints_array[13]['y'] - joints_array[2]['y']))
                if visible[13] and visible[3]:
                    torso_heights.append(
                        np.linalg.norm(joints_array[13]['y'] - joints_array[3]['y']))
                    # Make torso 75px
                if len(torso_heights) > 0:
                    scale = 75. / np.mean(torso_heights)
                else:
                    if visible[8] and visible[2]:
                        torso_heights.append(
                        np.linalg.norm(joints_array[8]['y'] - joints_array[2]['y']))
                    if visible[9] and visible[3]:
                        torso_heights.append(
                            np.linalg.norm(joints_array[9]['y'] - joints_array[3]['y']))
                    if len(torso_heights) > 0:
                        scale = 56. / np.mean(torso_heights)
                    else:
                        # Skip, person is too close.
                        continue
            person = {
                'joints':joints,
                'scale':scale,
                'pos':pos
            }

            people.append(person)

        return people        
            
    def load_image(self, ann) :
        image         = skimage.io.imread(ann['image_path'])

        return image
        
    def show_example(self, pos = None) :
        fig = plt.figure(figsize=(15,10))
        gs = gridspec.GridSpec(2, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[0, 1])
        ax4 = plt.subplot(gs[1, 1])
        if pos is None:
            pos = np.random.randint(1,len(self.annotations))
        ann = self.annotations[pos]
        image = self.load_image(ann)
        ax1.imshow(image)
        ax2.imshow(image)
        for person in ann['people']:
           print(person)
           joint_array = [None] * (1+np.max(MpiiData._JOINT_CONNECTIONS))
           for joint in person["joints"]:
               joint_array[joint['id']] = joint
           for link in MpiiData._JOINT_CONNECTIONS:
               joint1 = joint_array[link[0]]
               joint2 = joint_array[link[1]]
               print("{} --> {}".format(joint1, joint2))
               if joint1 is not None and joint2 is not None:
                   ax1.plot([joint1['x'],joint2['x']], [joint1['y'],joint2['y']])
        plt.show()

        

