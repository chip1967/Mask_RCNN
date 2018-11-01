
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
import os
import fnmatch
import sys
import time
import numpy as np
import json
# to import matlab file
import scipy.io


from abstract_data import *

############################################################
#  LSP Data
############################################################

class LspData(object):
    """
    Lsp Coco data set support 
    """

    def __init__(self,dataset_dir, subset):
        self.dataset_dir = dataset_dir
        self.subset = subset

    _JOINTS = [
        JointPositions.RIGHT_ANKLE,
        JointPositions.RIGHT_KNEE,
        JointPositions.RIGHT_HIP,
        JointPositions.LEFT_HIP,
        JointPositions.LEFT_KNEE,
        JointPositions.LEFT_ANKLE,
        JointPositions.RIGHT_WRIST,
        JointPositions.RIGHT_ELBOW,
        JointPositions.RIGHT_SHOULDER,
        JointPositions.LEFT_SHOULDER,
        JointPositions.LEFT_ELBOW,
        JointPositions.LEFT_WRIST,
        JointPositions.NECK_TOP,
        JointPositions.HEAD_TOP
        ]

    def load_images(self):
        image_objs = []
        ann_file = os.path.join(self.dataset_dir, "joints.mat")
        file = scipy.io.loadmat(ann_file, struct_as_record=False, squeeze_me=True)
        joints = file['joints']
        for index in range(len(joints)):
            img_joints = joints[index]
            filename   = "im{:05}.jpg".format(index)
            image_path = os.path.join(self.dataset_dir, "images", filename)
            joint_objs = []
            if index == 0:
                print(img_joints)
            for joint_index in range(0,len(LspData._JOINTS)-1):
                if img_joints[2][joint_index] != 0:
                    joint_objs.append(Joint(LspData._JOINTS[joint_index],
                                            img_joints[1][joint_index],
                                            img_joints[0][joint_index],
                                            img_joints[2][joint_index] == 1)),
            person = Person(joints = joint_objs, regions=None)
            image_objs.append(Image(ImageFile(image_path), people=[person]))
        return ImageSet(image_objs)
        """
        image_ids = self.person_keypoints.getImgIds(catIds = self.person_keypoints.getCatIds(catNms=['person']))
        image_anns = {}
        # build keypoints map
        keypoint_names = person_cat['keypoints']
        joint_names = [CocoData._JOINT_MAP[key_name] for key_name in keypoint_names]
        for ann_id in self.person_keypoints.getAnnIds(catIds = self.person_keypoints.getCatIds(catNms=['person'])) :
            ann = self.person_keypoints.anns[ann_id]
            if ann['image_id'] not in image_anns:
                image_anns[ann['image_id']] = [ ann ]
            else:
                image_anns[ann['image_id']].append(ann)                
        for image_id in image_ids:
            image = self.person_keypoints.imgs[image_id]
            image_path = os.path.join(self.dataset_dir, 'images', "{}{}".format(self.subset, self.year), image['file_name'])
            people = []
            for ann in image_anns[image_id]:
                def map_segment(segment):
                    return np.array(segment).reshape(-1,2)
                regions = map(map_segment, ann['segmentation'])
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
                people.append(Person(joints=joints,regions=regions))
            image_obj = Image(ImageFile(image_path),
                              people=people)
            image_objs.append(image_obj)
        return ImageSet(image_objs)
        """
    
    """
    def image_path(self, image_id) :
        img           = self.coco.imgs[image_id]
        image_path    = os.path.join(self.dataset_dir, "images", "{}{}".format(self.subset, self.year), img["file_name"])
        return image_path

    def load_image(self, image_id) :
        image_path    = self.image_path(image_id)
        image         = skimage.io.imread(image_path)
        return image

    def load_panoptic_image(self, image_id) :
        ann = self.panoptic.anns[image_id]
        panoptic_image_path = os.path.join(self.dataset_dir, "images", "panoptic_{}{}".format(self.subset, self.year), ann["file_name"])
        image         = skimage.io.imread(panoptic_image_path)
        return image
        
    def show_examples(self, pos = None) :
        if pos is None:
            pos = np.random.randint(1,len(self.coco.imgs))
        coco_image_id = list(self.coco.imgs.keys())[pos-1]
        img           = self.coco.imgs[coco_image_id]
        image         = self.load_image(coco_image_id)
        fig = plt.figure(figsize=(15,10))
        gs = gridspec.GridSpec(2, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[0, 1])
        ax4 = plt.subplot(gs[1, 1])
        ax1.imshow(image)
        plt.sca(ax1)
        self.coco.showAnns(self.coco.loadAnns(self.coco.getAnnIds(imgIds=[coco_image_id])))
        Self.person_keypoints.showAnns(self.person_keypoints.loadAnns(self.person_keypoints.getAnnIds(imgIds=[coco_image_id])))
        panoptic_image = self.load_panoptic_image(coco_image_id)
        ax2.imshow(panoptic_image)
        ann = self.panoptic.anns[coco_image_id]
        print("Segments")
        for segment in ann['segments_info']:
            category = self.panoptic.cats[segment['category_id']]
            super_category_name = category['supercategory']
            print("\t{}<{}:{}".format(category["name"],super_category_name,segment))
        plt.show()
    """
