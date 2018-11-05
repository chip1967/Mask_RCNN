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

import numpy as np
import skimage.io
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import logging
from pycocotools import mask as mask_utils

############################################################
#  Abstract Data Layer
############################################################

class ImageData(object):
    """
    Abstract base class for image representations
    """
    def __init__(self):
        return
        
class JointPosition(object):

    def __init__(self,name) :
        self.name = name

    def __str__(self):
        return self.name

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        if isinstance(other, JointPosition):
            self.name == other.name
        else:
            False

class JointPositions(object):
    
    RIGHT_ANKLE = JointPosition("Right Ankle")
    RIGHT_KNEE  = JointPosition("Right Knee")
    RIGHT_HIP   = JointPosition("Right Hip")
    RIGHT_WRIST = JointPosition("Right Wrist")
    RIGHT_ELBOW = JointPosition("Right Elbow")
    RIGHT_SHOULDER = JointPosition("Right Shoulder")
    LEFT_ANKLE = JointPosition("Left Ankle")
    LEFT_KNEE  = JointPosition("Left Knee")
    LEFT_HIP   = JointPosition("Left Hip")
    LEFT_WRIST = JointPosition("Left Wrist")
    LEFT_ELBOW = JointPosition("Left Elbow")
    LEFT_SHOULDER = JointPosition("Left Shoulder")
    NECK_TOP      = JointPosition("Neck Top")
    HEAD_TOP      = JointPosition("Head Top")
    NOSE          = JointPosition("Nose")
    LEFT_EYE      = JointPosition("Left Eye")
    RIGHT_EYE     = JointPosition("Right Eye")
    LEFT_EAR      = JointPosition("Left Ear")
    RIGHT_EAR     = JointPosition("Right Ear")

    JOINT_CONNECTIONS = [
        [RIGHT_ANKLE, RIGHT_KNEE],
        [RIGHT_WRIST, RIGHT_ELBOW],
        [RIGHT_KNEE, RIGHT_HIP],
        [RIGHT_ELBOW, RIGHT_SHOULDER],

        [LEFT_ANKLE, LEFT_KNEE],
        [LEFT_WRIST, LEFT_ELBOW],
        [LEFT_KNEE,  LEFT_HIP],
        [LEFT_ELBOW, LEFT_SHOULDER],

        [LEFT_SHOULDER, NECK_TOP, NOSE, HEAD_TOP],
        [RIGHT_SHOULDER, NECK_TOP, NOSE, HEAD_TOP],
        [LEFT_HIP, NECK_TOP, NOSE, HEAD_TOP],
        [RIGHT_HIP, NECK_TOP, NOSE, HEAD_TOP]
    ]

class ImageFile(ImageData):
    """
    """
    
    def __init__(self, image_path):
        ImageData.__init__(self)
        self.image_path = image_path

    def __str__(self):
        return self.image_path

    def read_image(self):
        image = skimage.io.imread(self.image_path)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image        
        
class Joint(object):

    def __init__(self,
                 position,
                 x,
                 y,
                 visible=None):
        assert(isinstance(position,JointPosition), "Should be JointPosition")
        self.position = position
        self.x = x
        self.y = y
        self.visible = visible
        
    def __str__(self) :
        return "Joint({},({},{}),{})".format(self.position, self.x, self.y, self.visible)

class Person(object):

    def __init__(self,
                 joints = None,
                 regions = None,
                 bbox = None,
                 dp_data = None,
                 dp_mask = None):
        self.joints   = joints
        self.regions  = regions
        self.bbox     = bbox
        self.dp_data  = dp_data
        self.dp_mask  = dp_mask

    def get_joint(self,joint_position):
        assert(isinstance(joint_position, JointPosition))
        l = [joint for joint in self.joints if joint.position.name == joint_position.name]
        if len(l) == 0:
            return None
        else:
            return l[0]

    def regions_to_mask(self, image):
        rle = self.regions_to_rle(image)
        return mask_utils.decode(rle)
        
    def regions_to_rle(self, image):
        rles = mask_utils.frPyObjects(list([region.reshape(-1) for region in self.regions]), image.height, image.width)
        rle = mask_utils.merge(rles)
        return rle
        
class Image(object):

    def __init__(self,
                 image_data,
                 height,
                 width,
                 people,
                 occlusions=None):
        self.image_data = image_data
        self.people     = people
        self.height     = height
        self.width      = width
        self.occlusions = occlusions

    def create_occlusion_mask(self):
        mask = np.zeros((self.height, self.width), dtype='int')
        for occlusion in self.occlusions:
            occlusion.add_to_mask(mask)
        return mask

    def __str__(self):
        return "Image[{},num people:{}]".format(self.image_data, len(self.people))

    @property    
    def has_dp_data(self):
        return any([ person.dp_data is not None for person in self.people])

    def show_image(self, person_index):
        person = self.people[person_index]
        image  = self.image_data.read_image()
        plt.imshow(image)
        plt.axis('off')

    def show_joints(self, person_index):
        person = self.people[person_index]
        image  = self.image_data.read_image()
        if person.joints is not None:
            for connection in JointPositions.JOINT_CONNECTIONS:
                possible_joints   = [person.get_joint(joint) for joint in connection]
                joint_x = np.array([ joint.x for joint in possible_joints if joint is not None])
                joint_y = np.array([ joint.y for joint in possible_joints if joint is not None])
                plt.plot(joint_x, joint_y)
                
    def show_regions(self, person_index):
        person = self.people[person_index]
        image  = self.image_data.read_image()
        if person.regions is not None:
            for region in person.regions:
                region = np.append(region, [region[0]], axis=0)
                plt.plot(region[:,0], region[:,1])
    def show_dp_data(self, person_index):
        person = self.people[person_index]
        image  = self.image_data.read_image()
        if person.dp_data is not None and person.bbox is not None:
            point_x = np.array(person.dp_data['dp_x'])/ 255. * person.bbox[2] # Strech the points to current box.
            point_y = np.array(person.dp_data['dp_y'])/ 255. * person.bbox[3] # Strech the points to current box.
            #
            point_I = np.array(person.dp_data['dp_I'])
            point_U = np.array(person.dp_data['dp_U'])
            point_V = np.array(person.dp_data['dp_V'])
            #
            x1,y1,x2,y2 = person.bbox[0],person.bbox[1],person.bbox[0]+person.bbox[2],person.bbox[1]+person.bbox[3]
            x2 = min( [ x2,image.shape[1] ] ); y2 = min( [ y2,image.shape[0] ] )
            ###############
            point_x = point_x + x1 
            point_y = point_y + y1
            plt.scatter(point_x,point_y,22,point_I)
            #plt.scatter(point_x,point_y,22,point_U)
            #plt.scatter(point_x,point_y,22,point_V)
        if person.dp_mask is not None and person.bbox is not None:
            ################
            x1,y1,x2,y2 = int(person.bbox[0]),int(person.bbox[1]),int(person.bbox[0]+person.bbox[2]),int(person.bbox[1]+person.bbox[3])
            x2 = min( [ x2,image.shape[1] ] );  y2 = min( [ y2,image.shape[0] ] )
            ################
            mask_im = cv2.resize( person.dp_mask, (x2-x1,y2-y1) ,interpolation=cv2.INTER_NEAREST)
            mask_im_size = np.zeros((image.shape[0], image.shape[1]))
            mask_im_size[y1:y2,x1:x2] = mask_im
            mask_bool_size = mask_im_size!=0
            mask_vis = cv2.applyColorMap( (mask_im_size*15).astype(np.uint8) , cv2.COLORMAP_PARULA)[:,:,:]
            mask_vis = np.append(mask_vis, np.zeros((image.shape[0], image.shape[1],1)), axis=2)
            mask_vis[:,:,:] = mask_vis[:,:,:]/255.0
            mask_vis[:,:,3][mask_bool_size] = 0.25
            plt.imshow(mask_vis, extent=(0, image.shape[1], image.shape[0], 0))

    def show_occlusions(self, person_index):
        person = self.people[person_index]
        image  = self.image_data.read_image()            
        if self.occlusions is not None:
            mask = self.create_occlusion_mask()
            logging.debug("Created occlusion mask %s", np.sum(mask) / 1.0 / image.shape[0] / image.shape[1])
            mask_image = np.zeros((image.shape[0], image.shape[1], 4))
            mask_image[mask,0] = 0.5 
            mask_image[mask,1] = 0.5 
            mask_image[mask,3] = 0.8
            print(mask)
            plt.imshow(mask, extent=(0, image.shape[1], image.shape[0], 0))
            

class ImageSet(object):

    def __init__(self,
                 images):
        self.images = images

    @property    
    def num_images(self):
        return len(self.images)

    
