# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
import functools
import os.path
import random
import sys
import xml.etree.ElementTree
import numpy as np
import matplotlib.pyplot as plt
import skimage.data
import cv2
import PIL.Image
import logging

from abstract_data import *

class Occlusion(object):
    
    def __init__(self, occluder, scale_factor, center, person):
        self.occluder       = occluder
        self.scale_factor   = scale_factor
        self.center         = center
        self.person         = person
    
    def add_to_mask(self, foreground_mask):
        occluder = DataOccluder.resize_by_factor(self.occluder, self.scale_factor)
        DataOccluder.add_mask(occluder, foreground_mask, center=self.center)

    def paste_over(self, image):
        occluder = DataOccluder.resize_by_factor(self.occluder, self.scale_factor)
        #logging.debug("adding occlusion over image %s, %s, %s bbox %s", image.shape, occluder.shape, self.center, self.person.bbox)
        DataOccluder.paste_over(occluder, image, center=self.center)

class DataOccluder(object):

    def __init__(self,pascal_voc_root_path):
        logging.getLogger('PIL.PngImagePlugin').setLevel(logging.INFO)
        logging.debug("Loading occlusion data from %s", pascal_voc_root_path)
        occluders = []
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        annotation_paths = DataOccluder.list_filepaths(os.path.join(pascal_voc_root_path, 'Annotations'))
        for annotation_path in annotation_paths:

            # if len(occluders) > 100:
            #      continue
            
            xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
            is_segmented = (xml_root.find('segmented').text != '0')

            if not is_segmented:
                continue

            boxes = []
            for i_obj, obj in enumerate(xml_root.findall('object')):
                is_person = (obj.find('name').text == 'person')
                is_difficult = (obj.find('difficult').text != '0')
                is_truncated = (obj.find('truncated').text != '0')
                if not is_person and not is_difficult and not is_truncated:
                    bndbox = obj.find('bndbox')
                    box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                    boxes.append((i_obj, box))

            if not boxes:
                continue

            im_filename = xml_root.find('filename').text
            seg_filename = im_filename.replace('jpg', 'png')
            
            im_path = os.path.join(pascal_voc_root_path, 'JPEGImages', im_filename)
            seg_path = os.path.join(pascal_voc_root_path,'SegmentationObject', seg_filename)

            im = np.asarray(PIL.Image.open(im_path))
            labels = np.asarray(PIL.Image.open(seg_path))

            for i_obj, (xmin, ymin, xmax, ymax) in boxes:
                object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8)*255
                object_image = im[ymin:ymax, xmin:xmax]
                if cv2.countNonZero(object_mask) < 500:
                    # Ignore small objects
                    continue

            # Reduce the opacity of the mask along the border for smoother blending
            eroded = cv2.erode(object_mask, structuring_element)
            object_mask[eroded < object_mask] = 192
            object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)
            
            # Downscale for efficiency
            object_with_mask = DataOccluder.resize_by_factor(object_with_mask, 0.5)
            occluders.append(object_with_mask)
        self.occluders = occluders
        logging.debug("%s Occluders loaded", len(occluders))

    def make_occluded_image(self,image):
        image_data = image.image_data
        image_height = image.height
        image_width  = image.width
        image_people = image.people
        occlusions   = []
        for person in image_people:
            bbox = person.bbox
            x1,y1,x2,y2 = person.bbox[0],person.bbox[1],person.bbox[0]+person.bbox[2],person.bbox[1]+person.bbox[3]
            person_height = x2 - x1
            person_width  = y2 - y1
            if (min(image_width, image_height) > 10 * max(person_height, person_width)):                
                continue
            count = np.random.randint(0, 2)
            while count > 0:
                occluder = random.choice(self.occluders)
                occlusion_size  = max(occluder.shape[0], occluder.shape[1])
                im_scale_factor = max(person_height, person_width) / (occlusion_size * 1.0)
                random_scale_factor = np.random.uniform(0.4, 2.0)
                scale_factor = random_scale_factor * im_scale_factor
                occlusion_final_shape = np.array(occluder.shape).astype('float') * scale_factor
                # logging.debug("Scale_factor %s, occluder %s, %s, %s",scale_factor, occluder.shape, min(image_width, image_height), max(person_height, person_width))
                if min(occlusion_final_shape) > 5:
                    center = np.random.uniform([ int(x1-occlusion_final_shape[1]*0.5), int(y1-occlusion_final_shape[0]*0.5)],
                                               [ int(x2 + occlusion_final_shape[1]*0.5), int(y2 + occlusion_final_shape[0]*0.5)])
                    occlusion = Occlusion(occluder, scale_factor, center, person)
                    occlusions.append(occlusion)
                    count = count - 1
        if image. occlusions is not None:
            raise Exception("TODO:Cant occude image with foreground mask")
        image_data = OccludedImageData(image.image_data, occlusions)        
        return Image(image_data, image.height, image.width, image.people, occlusions)

    def add_mask(im_src, current_mask, center):
        """Pastes `im_src` onto `current_mask`

        Args:
            im_src: The RGBA image to be pasted onto `current_mask`. Its size can be arbitrary.
            current_mask: The target image.
            alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
                at each pixel. Large values mean more visibility for `im_src`.
            center: coordinates in `current_mask` where the center of `im_src` should be placed.
        """

        width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
        width_height_dst = np.asarray([current_mask.shape[1], current_mask.shape[0]])

        center = np.round(center).astype(np.int32)
        raw_start_dst = center - width_height_src // 2
        raw_end_dst = raw_start_dst + width_height_src
        
        start_dst = np.clip(raw_start_dst, 0, width_height_dst)
        end_dst = np.clip(raw_end_dst, 0, width_height_dst)
        region_dst = current_mask[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

        start_src = start_dst - raw_start_dst
        end_src = width_height_src + (end_dst - raw_end_dst)
        region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
        region_mask = region_src[..., 3] != 0
        
        current_mask[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]][region_mask] = 1

    def paste_over(im_src, im_dst, center):
        """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.

        Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
        `im_src` becomes visible).

        Args:
            im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
            im_dst: The target image.
                at each pixel. Large values mean more visibility for `im_src`.
            center: coordinates in `im_dst` where the center of `im_src` should be placed.
        """

        width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
        width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

        center = np.round(center).astype(np.int32)
        raw_start_dst = center - width_height_src // 2
        raw_end_dst = raw_start_dst + width_height_src
        
        start_dst = np.clip(raw_start_dst, 0, width_height_dst)
        end_dst = np.clip(raw_end_dst, 0, width_height_dst)
        region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

        start_src = start_dst - raw_start_dst
        end_src = width_height_src + (end_dst - raw_end_dst)
        region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
        color_src = region_src[..., 0:3]
        alpha = region_src[..., 3:].astype(np.float32)/255
        
        im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
            alpha * color_src + (1 - alpha) * region_dst)

    def resize_by_factor(im, factor):
        """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
            for downscaling.
        """
        new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
        interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
        return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)

    def list_filepaths(dirpath):
        names = os.listdir(dirpath)
        paths = [os.path.join(dirpath, name) for name in names]
        return sorted(filter(os.path.isfile, paths))

class OccludedImageData(ImageData):

    def __init__(self, image_data, occluders):
        ImageData.__init__(self)
        self.image_data = image_data
        self.occluders = occluders

    def read_image(self):
        image = self.image_data.read_image()
        for occluder in self.occluders:
            occluder.paste_over(image)
        return image

class OccludedImages(ImageSet):

    def __init__(self, occluder, imageset):
        occluded_images = [occluder.make_occluded_image(image) for image in imageset.images]
        ImageSet.__init__(self, occluded_images)

class OccludedData(object):

    def __init__(self, occluder, data):
        self.occluder = occluder
        self.data = data
        self.person_cat = data.person_cat

    def load_images(self):
        images = self.data.load_images()
        return OccludedImages(self.occluder, images)

if False:
    def occlude_with_objects(self, im, regions):        
        """Returns an augmented version of `im`, containing some occluders from the Pascal VOC dataset."""
        
        result = im.copy()
        width_height = np.asarray([im.shape[1], im.shape[0]])
        im_scale_factor = min(width_height) / 256
        count = np.random.randint(1, 8)

        for _ in range(count):
            occluder = random.choice(self.occluders)
            
            random_scale_factor = np.random.uniform(0.2, 1.0)
            scale_factor = random_scale_factor * im_scale_factor
            occluder = DataOccluder.resize_by_factor(occluder, scale_factor)
            
            center = np.random.uniform([0,0], width_height)
            DataOccluder.paste_over(im_src=occluder, im_dst=result, center=center)

        return result


    def paste_over(im_src, im_dst, center):
        """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.

        Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
        `im_src` becomes visible).

        Args:
            im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
            im_dst: The target image.
            alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
                at each pixel. Large values mean more visibility for `im_src`.
            center: coordinates in `im_dst` where the center of `im_src` should be placed.
        """

        width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
        width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

        center = np.round(center).astype(np.int32)
        raw_start_dst = center - width_height_src // 2
        raw_end_dst = raw_start_dst + width_height_src
        
        start_dst = np.clip(raw_start_dst, 0, width_height_dst)
        end_dst = np.clip(raw_end_dst, 0, width_height_dst)
        region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

        start_src = start_dst - raw_start_dst
        end_src = width_height_src + (end_dst - raw_end_dst)
        region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
        color_src = region_src[..., 0:3]
        alpha = region_src[..., 3:].astype(np.float32)/255
        
        im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
            alpha * color_src + (1 - alpha) * region_dst)


    def resize_by_factor(im, factor):
        """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
            for downscaling.
        """
        new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
        interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
        return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)

    def list_filepaths(dirpath):
        names = os.listdir(dirpath)
        paths = [os.path.join(dirpath, name) for name in names]
        return sorted(filter(os.path.isfile, paths))

