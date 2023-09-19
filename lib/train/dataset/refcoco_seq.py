import json
import os
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
import torch
import random
import numpy as np
from pycocotools.coco import COCO
from collections import OrderedDict
from lib.train.admin import env_settings
from lib.utils.string_utils import clean_string


class RefCOCOSeq(BaseVideoDataset):
    """ The COCO dataset. COCO is an image dataset. Thus, we treat each image as a sequence of length 1.

    Publication:
        Microsoft COCO: Common Objects in Context.
        Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona,
        Deva Ramanan, Piotr Dollar and C. Lawrence Zitnick
        ECCV, 2014
        https://arxiv.org/pdf/1405.0312.pdf

    Download the images along with annotations from http://cocodataset.org/#download. The root folder should be
    organized as follows.
        - coco_root
            - annotations
                - instances_train2014.json
                - instances_train2017.json
            - images
                - train2014
                - train2017

    Note: You also have to install the coco pythonAPI from https://github.com/cocodataset/cocoapi.
    """

    def __init__(self, root=None, refcoco_type=None, image_loader=jpeg4py_loader, data_fraction=None, 
                split="train", version="2014"):
        """
        args:
            root - path to the coco dataset.
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
            data_fraction (None) - Fraction of images to be used. The images are selected randomly. If None, all the
                                  images  will be used
            split - 'train' or 'val'.
            version - version of coco dataset (2014 or 2017)
        """
        root = env_settings().ref_coco_dir if root is None else root
        super().__init__('RefCOCO', root, image_loader)

        self.img_pth = os.path.join(root, 'images/{}{}/'.format(split, version))
        self.anno_path = os.path.join(root, 'annotations/{}/instances.json'.format(refcoco_type))

        # Load the COCO set.
        # self.anns_all = json.load(open(self.anno_path, 'r'))
        self.train_anns = json.load(open(self.anno_path, 'r'))['train']

        # self.coco_set = COCO(self.anno_path)
        # self.cats = self.coco_set.cats
        # self.class_list = self.get_class_list()

        self.sequence_list = self._get_sequence_list() # save image ids

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))
        # self.seq_per_class = self._build_seq_per_class()

        self.cats = self._get_cats_info()

    def _get_sequence_list(self):
        # ann_list = list(self.coco_set.anns.keys())
        # seq_list = [a for a in ann_list if self.coco_set.anns[a]['iscrowd'] == 0]
        
        seq_list = [ann['image_id'] for ann in self.train_anns]
        return seq_list

    def is_video_sequence(self):
        return False

    def get_num_classes(self):
        return len(self.class_list)

    def get_name(self):
        return 'coco'

    def has_class_info(self):
        return True

    def get_class_list(self):
        class_list = []
        for cat_id in self.cats.keys():
            class_list.append(self.cats[cat_id]['name'])
        return class_list

    def has_segmentation_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _build_seq_per_class(self):
        seq_per_class = {}
        for i, seq in enumerate(self.sequence_list):
            class_name = self.cats[self.coco_set.anns[seq]['category_id']]['name']
            if class_name not in seq_per_class:
                seq_per_class[class_name] = [i]
            else:
                seq_per_class[class_name].append(i)

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def get_sequence_info(self, seq_id):
        anno = self._get_anno(seq_id)

        bbox = torch.Tensor(anno['bbox']).view(1, 4)

        # mask = torch.Tensor(self.coco_set.annToMask(anno)).unsqueeze(dim=0)

        '''2021.1.3 To avoid too small bounding boxes. Here we change the threshold to 50 pixels'''
        valid = (bbox[:, 2] > 50) & (bbox[:, 3] > 50)

        visible = valid.clone().byte()

        # return {'bbox': bbox, 'mask': mask, 'valid': valid, 'visible': visible}
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_anno(self, seq_id):
        # anno = self.coco_set.anns[self.sequence_list[seq_id]]
        anno = self.train_anns[seq_id]

        return anno

    def _get_frames(self, seq_id):
        # path = self.coco_set.loadImgs([self.coco_set.anns[self.sequence_list[seq_id]]['image_id']])[0]['file_name']
        path = "COCO_train2014_%012d.jpg" % self.train_anns[seq_id]['image_id']
        img = self.image_loader(os.path.join(self.img_pth, path))
        return img

    def get_meta_info(self, seq_id):
        try:
            # cat_dict_current = self.cats[self.coco_set.anns[self.sequence_list[seq_id]]['category_id']]
            cat_dict_current = self.cats[self.train_anns[seq_id]['category_id'] + 1]
            expressions = self.train_anns[seq_id]['expressions']
            exp_str = expressions[np.random.choice(len(expressions))]
            exp_str = clean_string(exp_str)
            object_meta = OrderedDict({'object_class_name': cat_dict_current['name'],
                                       'motion_class': None,
                                       'major_class': cat_dict_current['supercategory'],
                                       'root_class': None,
                                       'motion_adverb': None,
                                       'exp_str': exp_str})
        except:
            return None
            # object_meta = OrderedDict({'object_class_name': None,
            #                            'motion_class': None,
            #                            'major_class': None,
            #                            'root_class': None,
            #                            'motion_adverb': None,
            #                            'exp_str': None})
            
        return object_meta


    def get_class_name(self, seq_id):
        cat_dict_current = self.cats[self.coco_set.anns[self.sequence_list[seq_id]]['category_id']]
        return cat_dict_current['name']

    def get_frames(self, seq_id=None, frame_ids=None, anno=None):
        # COCO is an image dataset. Thus we replicate the image denoted by seq_id len(frame_ids) times, and return a
        # list containing these replicated images.
        frame = self._get_frames(seq_id)

        frame_list = [frame.copy() for _ in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0, ...] for _ in frame_ids]

        object_meta = self.get_meta_info(seq_id)

        return frame_list, anno_frames, object_meta

    def _get_cats_info(self):
        cats_info = {1: {'supercategory': 'person', 'id': 1, 'name': 'person'}, 
                    2: {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
                    3: {'supercategory': 'vehicle', 'id': 3, 'name': 'car'}, 
                    4: {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'}, 
                    5: {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}, 
                    6: {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'}, 
                    7: {'supercategory': 'vehicle', 'id': 7, 'name': 'train'}, 
                    8: {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'}, 
                    9: {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'}, 
                    10: {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'}, 
                    11: {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'}, 
                    13: {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'}, 
                    14: {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'}, 
                    15: {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'}, 
                    16: {'supercategory': 'animal', 'id': 16, 'name': 'bird'}, 
                    17: {'supercategory': 'animal', 'id': 17, 'name': 'cat'}, 
                    18: {'supercategory': 'animal', 'id': 18, 'name': 'dog'}, 
                    19: {'supercategory': 'animal', 'id': 19, 'name': 'horse'}, 
                    20: {'supercategory': 'animal', 'id': 20, 'name': 'sheep'}, 
                    21: {'supercategory': 'animal', 'id': 21, 'name': 'cow'}, 
                    22: {'supercategory': 'animal', 'id': 22, 'name': 'elephant'}, 
                    23: {'supercategory': 'animal', 'id': 23, 'name': 'bear'}, 
                    24: {'supercategory': 'animal', 'id': 24, 'name': 'zebra'}, 
                    25: {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'}, 
                    27: {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'}, 
                    28: {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'}, 
                    31: {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'}, 
                    32: {'supercategory': 'accessory', 'id': 32, 'name': 'tie'}, 
                    33: {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'}, 
                    34: {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'}, 
                    35: {'supercategory': 'sports', 'id': 35, 'name': 'skis'}, 
                    36: {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'}, 
                    37: {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'}, 
                    38: {'supercategory': 'sports', 'id': 38, 'name': 'kite'}, 
                    39: {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'}, 
                    40: {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'}, 
                    41: {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'}, 
                    42: {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'}, 
                    43: {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'}, 
                    44: {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'}, 
                    46: {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'}, 
                    47: {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'}, 
                    48: {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'}, 
                    49: {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'}, 
                    50: {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'}, 
                    51: {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'}, 
                    52: {'supercategory': 'food', 'id': 52, 'name': 'banana'}, 
                    53: {'supercategory': 'food', 'id': 53, 'name': 'apple'}, 
                    54: {'supercategory': 'food', 'id': 54, 'name': 'sandwich'}, 
                    55: {'supercategory': 'food', 'id': 55, 'name': 'orange'}, 
                    56: {'supercategory': 'food', 'id': 56, 'name': 'broccoli'}, 
                    57: {'supercategory': 'food', 'id': 57, 'name': 'carrot'}, 
                    58: {'supercategory': 'food', 'id': 58, 'name': 'hot dog'}, 
                    59: {'supercategory': 'food', 'id': 59, 'name': 'pizza'}, 
                    60: {'supercategory': 'food', 'id': 60, 'name': 'donut'}, 
                    61: {'supercategory': 'food', 'id': 61, 'name': 'cake'}, 
                    62: {'supercategory': 'furniture', 'id': 62, 'name': 'chair'}, 
                    63: {'supercategory': 'furniture', 'id': 63, 'name': 'couch'}, 
                    64: {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'}, 
                    65: {'supercategory': 'furniture', 'id': 65, 'name': 'bed'}, 
                    67: {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'}, 
                    70: {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'}, 
                    72: {'supercategory': 'electronic', 'id': 72, 'name': 'tv'}, 
                    73: {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'}, 
                    74: {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'}, 
                    75: {'supercategory': 'electronic', 'id': 75, 'name': 'remote'}, 
                    76: {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'}, 
                    77: {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'}, 
                    78: {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'}, 
                    79: {'supercategory': 'appliance', 'id': 79, 'name': 'oven'}, 
                    80: {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'}, 
                    81: {'supercategory': 'appliance', 'id': 81, 'name': 'sink'}, 
                    82: {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'}, 
                    84: {'supercategory': 'indoor', 'id': 84, 'name': 'book'}, 
                    85: {'supercategory': 'indoor', 'id': 85, 'name': 'clock'}, 
                    86: {'supercategory': 'indoor', 'id': 86, 'name': 'vase'}, 
                    87: {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'}, 
                    88: {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'}, 
                    89: {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'}, 
                    90: {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}}
        return cats_info