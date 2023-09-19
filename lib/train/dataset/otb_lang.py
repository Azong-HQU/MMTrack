import numpy as np
from lib.test.utils.load_text import load_text
from lib.train.data import jpeg4py_loader
import os
from .base_video_dataset import BaseVideoDataset
import torch
import cv2
import random
from collections import OrderedDict
from lib.utils.string_utils import clean_string


class OTB_Lang(BaseVideoDataset):
    """ OTB-Sentence VL dataset
    Publication:
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, data_fraction=None):
        super().__init__('OTB_Lang', root, image_loader)

        self.base_path = root
        self.video_path = os.path.join(root, 'OTB_videos')
        self.split = split
        if self.split == 'train':
            self.sequence_list = self._get_sequence_info_list_train()
            self.nlp_root = os.path.join(self.base_path, 'OTB_query_train')
        elif self.split == 'test':
            self.sequence_list = self._get_sequence_info_list_test()
            self.nlp_root = os.path.join(self.base_path, 'OTB_query_test')

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def __len__(self):
        return len(self.sequence_info_list)

    def get_name(self):
        return 'OTB_Lang'

    def get_num_classes(self):
        return False

    def get_class_list(self):
        return False

    def _get_sequence_path(self, seq_id):
        path = self.sequence_list[seq_id]['path']
        return os.path.join(self.video_path, path)

    def _read_bb_anno(self, seq_id):
        bb_anno_file = self.sequence_list[seq_id]['anno_path']
        bb_path = os.path.join(self.video_path, bb_anno_file)

        gt = load_text(str(bb_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        start_frame = self.sequence_list[seq_id]['startFrame'] - 1
        end_frame = self.sequence_list[seq_id]['endFrame'] - 1

        construct_gt = np.zeros((end_frame + 1, 4))
        construct_gt[start_frame: end_frame + 1, :] = gt

        return torch.tensor(gt)

    def get_sequence_info(self, seq_id):
        bbox = self._read_bb_anno(seq_id)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:04}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, seq_id, frame_id):
        start_frame = self.sequence_list[seq_id]['startFrame'] - 1
        frame_id = frame_id + start_frame
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def _get_expression(self, seq_id):
        # read expression data
        exp_path = os.path.join(self.nlp_root, self.sequence_list[seq_id]['name']+'.txt')
        exp_str = ''
        try:
            with open(exp_path, 'r') as f:
                for line in f.readlines():
                    exp_str += line
        except Exception as e:
            print(e)
            return None
        
        assert (exp_str != '' and not exp_str is None), 'ERROR: Language File is None: "{}"'.format(exp_path)
        exp_str = clean_string(exp_str)
        return exp_str

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, seq_id, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        exp_str = self._get_expression(seq_id) # read expression data

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None,
                                   'exp_str': exp_str})

        return frame_list, anno_frames, object_meta

    def _get_sequence_info_list_train(self):
        sequence_info_list = [
            {"name": "Basketball", "path": "Basketball/img", "startFrame": 1, "endFrame": 725, "nz": 4, "ext": "jpg",
             "anno_path": "Basketball/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Bolt", "path": "Bolt/img", "startFrame": 1, "endFrame": 350, "nz": 4, "ext": "jpg",
             "anno_path": "Bolt/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Boy", "path": "Boy/img", "startFrame": 1, "endFrame": 602, "nz": 4, "ext": "jpg",
             "anno_path": "Boy/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Car4", "path": "Car4/img", "startFrame": 1, "endFrame": 659, "nz": 4, "ext": "jpg",
             "anno_path": "Car4/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "CarDark", "path": "CarDark/img", "startFrame": 1, "endFrame": 393, "nz": 4, "ext": "jpg",
             "anno_path": "CarDark/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "CarScale", "path": "CarScale/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg",
             "anno_path": "CarScale/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "Coke", "path": "Coke/img", "startFrame": 1, "endFrame": 291, "nz": 4, "ext": "jpg",
             "anno_path": "Coke/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Couple", "path": "Couple/img", "startFrame": 1, "endFrame": 140, "nz": 4, "ext": "jpg",
             "anno_path": "Couple/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Crossing", "path": "Crossing/img", "startFrame": 1, "endFrame": 120, "nz": 4, "ext": "jpg",
             "anno_path": "Crossing/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "David", "path": "David/img", "startFrame": 300, "endFrame": 770, "nz": 4, "ext": "jpg",
             "anno_path": "David/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "David2", "path": "David2/img", "startFrame": 1, "endFrame": 537, "nz": 4, "ext": "jpg",
             "anno_path": "David2/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "David3", "path": "David3/img", "startFrame": 1, "endFrame": 252, "nz": 4, "ext": "jpg",
             "anno_path": "David3/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Deer", "path": "Deer/img", "startFrame": 1, "endFrame": 71, "nz": 4, "ext": "jpg",
             "anno_path": "Deer/groundtruth_rect.txt",
             "object_class": "mammal"},
            {"name": "Dog1", "path": "Dog1/img", "startFrame": 1, "endFrame": 1350, "nz": 4, "ext": "jpg",
             "anno_path": "Dog1/groundtruth_rect.txt",
             "object_class": "dog"},
            {"name": "Doll", "path": "Doll/img", "startFrame": 1, "endFrame": 3872, "nz": 4, "ext": "jpg",
             "anno_path": "Doll/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Dudek", "path": "Dudek/img", "startFrame": 1, "endFrame": 1145, "nz": 4, "ext": "jpg",
             "anno_path": "Dudek/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "FaceOcc1", "path": "FaceOcc1/img", "startFrame": 1, "endFrame": 892, "nz": 4, "ext": "jpg",
             "anno_path": "FaceOcc1/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "FaceOcc2", "path": "FaceOcc2/img", "startFrame": 1, "endFrame": 812, "nz": 4, "ext": "jpg",
             "anno_path": "FaceOcc2/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Fish", "path": "Fish/img", "startFrame": 1, "endFrame": 476, "nz": 4, "ext": "jpg",
             "anno_path": "Fish/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "FleetFace", "path": "FleetFace/img", "startFrame": 1, "endFrame": 707, "nz": 4, "ext": "jpg",
             "anno_path": "FleetFace/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Football", "path": "Football/img", "startFrame": 1, "endFrame": 362, "nz": 4, "ext": "jpg",
             "anno_path": "Football/groundtruth_rect.txt",
             "object_class": "person head"},
            {"name": "Football1", "path": "Football1/img", "startFrame": 1, "endFrame": 74, "nz": 4, "ext": "jpg",
             "anno_path": "Football1/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Freeman1", "path": "Freeman1/img", "startFrame": 1, "endFrame": 326, "nz": 4, "ext": "jpg",
             "anno_path": "Freeman1/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Freeman3", "path": "Freeman3/img", "startFrame": 1, "endFrame": 460, "nz": 4, "ext": "jpg",
             "anno_path": "Freeman3/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Freeman4", "path": "Freeman4/img", "startFrame": 1, "endFrame": 283, "nz": 4, "ext": "jpg",
             "anno_path": "Freeman4/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Girl", "path": "Girl/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg",
             "anno_path": "Girl/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Ironman", "path": "Ironman/img", "startFrame": 1, "endFrame": 166, "nz": 4, "ext": "jpg",
             "anno_path": "Ironman/groundtruth_rect.txt",
             "object_class": "person head"},
            {"name": "Jogging-1", "path": "Jogging/img", "startFrame": 1, "endFrame": 307, "nz": 4, "ext": "jpg",
             "anno_path": "Jogging/groundtruth_rect.1.txt",
             "object_class": "person"},
            {"name": "Jogging-2", "path": "Jogging/img", "startFrame": 1, "endFrame": 307, "nz": 4, "ext": "jpg",
             "anno_path": "Jogging/groundtruth_rect.2.txt",
             "object_class": "person"},
            {"name": "Jumping", "path": "Jumping/img", "startFrame": 1, "endFrame": 313, "nz": 4, "ext": "jpg",
             "anno_path": "Jumping/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Lemming", "path": "Lemming/img", "startFrame": 1, "endFrame": 1336, "nz": 4, "ext": "jpg",
             "anno_path": "Lemming/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Liquor", "path": "Liquor/img", "startFrame": 1, "endFrame": 1741, "nz": 4, "ext": "jpg",
             "anno_path": "Liquor/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Matrix", "path": "Matrix/img", "startFrame": 1, "endFrame": 100, "nz": 4, "ext": "jpg",
             "anno_path": "Matrix/groundtruth_rect.txt",
             "object_class": "person head"},
            {"name": "Mhyang", "path": "Mhyang/img", "startFrame": 1, "endFrame": 1490, "nz": 4, "ext": "jpg",
             "anno_path": "Mhyang/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "MotorRolling", "path": "MotorRolling/img", "startFrame": 1, "endFrame": 164, "nz": 4,
             "ext": "jpg", "anno_path": "MotorRolling/groundtruth_rect.txt",
             "object_class": "vehicle"},
            {"name": "MountainBike", "path": "MountainBike/img", "startFrame": 1, "endFrame": 228, "nz": 4,
             "ext": "jpg", "anno_path": "MountainBike/groundtruth_rect.txt",
             "object_class": "bicycle"},
            {"name": "Shaking", "path": "Shaking/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg",
             "anno_path": "Shaking/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Singer1", "path": "Singer1/img", "startFrame": 1, "endFrame": 351, "nz": 4, "ext": "jpg",
             "anno_path": "Singer1/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Singer2", "path": "Singer2/img", "startFrame": 1, "endFrame": 366, "nz": 4, "ext": "jpg",
             "anno_path": "Singer2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Skating1", "path": "Skating1/img", "startFrame": 1, "endFrame": 400, "nz": 4, "ext": "jpg",
             "anno_path": "Skating1/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Skiing", "path": "Skiing/img", "startFrame": 1, "endFrame": 81, "nz": 4, "ext": "jpg",
             "anno_path": "Skiing/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Soccer", "path": "Soccer/img", "startFrame": 1, "endFrame": 392, "nz": 4, "ext": "jpg",
             "anno_path": "Soccer/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Subway", "path": "Subway/img", "startFrame": 1, "endFrame": 175, "nz": 4, "ext": "jpg",
             "anno_path": "Subway/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Suv", "path": "Suv/img", "startFrame": 1, "endFrame": 945, "nz": 4, "ext": "jpg",
             "anno_path": "Suv/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "Sylvester", "path": "Sylvester/img", "startFrame": 1, "endFrame": 1345, "nz": 4, "ext": "jpg",
             "anno_path": "Sylvester/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Tiger1", "path": "Tiger1/img", "startFrame": 1, "endFrame": 354, "nz": 4, "ext": "jpg",
             "anno_path": "Tiger1/groundtruth_rect.txt", "initOmit": 5,
             "object_class": "other"},
            {"name": "Tiger2", "path": "Tiger2/img", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg",
             "anno_path": "Tiger2/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Trellis", "path": "Trellis/img", "startFrame": 1, "endFrame": 569, "nz": 4, "ext": "jpg",
             "anno_path": "Trellis/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Walking", "path": "Walking/img", "startFrame": 1, "endFrame": 412, "nz": 4, "ext": "jpg",
             "anno_path": "Walking/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Walking2", "path": "Walking2/img", "startFrame": 1, "endFrame": 500, "nz": 4, "ext": "jpg",
             "anno_path": "Walking2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Woman", "path": "Woman/img", "startFrame": 1, "endFrame": 597, "nz": 4, "ext": "jpg",
             "anno_path": "Woman/groundtruth_rect.txt",
             "object_class": "person"}
        ]

        return sequence_info_list

    def _get_sequence_info_list_test(self):
        sequence_info_list = [
            {"name": "Biker", "path": "Biker/img", "startFrame": 1, "endFrame": 142, "nz": 4, "ext": "jpg",
             "anno_path": "Biker/groundtruth_rect.txt",
             "object_class": "person head"},
            {"name": "Bird1", "path": "Bird1/img", "startFrame": 1, "endFrame": 408, "nz": 4, "ext": "jpg",
             "anno_path": "Bird1/groundtruth_rect.txt",
             "object_class": "bird"},
            {"name": "Bird2", "path": "Bird2/img", "startFrame": 1, "endFrame": 99, "nz": 4, "ext": "jpg",
             "anno_path": "Bird2/groundtruth_rect.txt",
             "object_class": "bird"},
            {"name": "BlurBody", "path": "BlurBody/img", "startFrame": 1, "endFrame": 334, "nz": 4, "ext": "jpg",
             "anno_path": "BlurBody/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "BlurCar1", "path": "BlurCar1/img", "startFrame": 247, "endFrame": 988, "nz": 4, "ext": "jpg",
             "anno_path": "BlurCar1/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "BlurCar2", "path": "BlurCar2/img", "startFrame": 1, "endFrame": 585, "nz": 4, "ext": "jpg",
             "anno_path": "BlurCar2/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "BlurCar3", "path": "BlurCar3/img", "startFrame": 3, "endFrame": 359, "nz": 4, "ext": "jpg",
             "anno_path": "BlurCar3/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "BlurCar4", "path": "BlurCar4/img", "startFrame": 18, "endFrame": 397, "nz": 4, "ext": "jpg",
             "anno_path": "BlurCar4/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "BlurFace", "path": "BlurFace/img", "startFrame": 1, "endFrame": 493, "nz": 4, "ext": "jpg",
             "anno_path": "BlurFace/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "BlurOwl", "path": "BlurOwl/img", "startFrame": 1, "endFrame": 631, "nz": 4, "ext": "jpg",
             "anno_path": "BlurOwl/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Board", "path": "Board/img", "startFrame": 1, "endFrame": 698, "nz": 5, "ext": "jpg",
             "anno_path": "Board/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Bolt2", "path": "Bolt2/img", "startFrame": 1, "endFrame": 293, "nz": 4, "ext": "jpg",
             "anno_path": "Bolt2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Box", "path": "Box/img", "startFrame": 1, "endFrame": 1161, "nz": 4, "ext": "jpg",
             "anno_path": "Box/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Car1", "path": "Car1/img", "startFrame": 1, "endFrame": 1020, "nz": 4, "ext": "jpg",
             "anno_path": "Car1/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "Car2", "path": "Car2/img", "startFrame": 1, "endFrame": 913, "nz": 4, "ext": "jpg",
             "anno_path": "Car2/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "Car24", "path": "Car24/img", "startFrame": 1, "endFrame": 3059, "nz": 4, "ext": "jpg",
             "anno_path": "Car24/groundtruth_rect.txt",
             "object_class": "car"},
            {"name": "Coupon", "path": "Coupon/img", "startFrame": 1, "endFrame": 327, "nz": 4, "ext": "jpg",
             "anno_path": "Coupon/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Crowds", "path": "Crowds/img", "startFrame": 1, "endFrame": 347, "nz": 4, "ext": "jpg",
             "anno_path": "Crowds/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Dancer", "path": "Dancer/img", "startFrame": 1, "endFrame": 225, "nz": 4, "ext": "jpg",
             "anno_path": "Dancer/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Dancer2", "path": "Dancer2/img", "startFrame": 1, "endFrame": 150, "nz": 4, "ext": "jpg",
             "anno_path": "Dancer2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Diving", "path": "Diving/img", "startFrame": 1, "endFrame": 215, "nz": 4, "ext": "jpg",
             "anno_path": "Diving/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Dog", "path": "Dog/img", "startFrame": 1, "endFrame": 127, "nz": 4, "ext": "jpg",
             "anno_path": "Dog/groundtruth_rect.txt",
             "object_class": "dog"},
            {"name": "DragonBaby", "path": "DragonBaby/img", "startFrame": 1, "endFrame": 113, "nz": 4, "ext": "jpg",
             "anno_path": "DragonBaby/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Girl2", "path": "Girl2/img", "startFrame": 1, "endFrame": 1500, "nz": 4, "ext": "jpg",
             "anno_path": "Girl2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Gym", "path": "Gym/img", "startFrame": 1, "endFrame": 767, "nz": 4, "ext": "jpg",
             "anno_path": "Gym/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human2", "path": "Human2/img", "startFrame": 1, "endFrame": 1128, "nz": 4, "ext": "jpg",
             "anno_path": "Human2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human3", "path": "Human3/img", "startFrame": 1, "endFrame": 1698, "nz": 4, "ext": "jpg",
             "anno_path": "Human3/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human4", "path": "Human4/img", "startFrame": 1, "endFrame": 667, "nz": 4, "ext": "jpg",
             "anno_path": "Human4/groundtruth_rect.2.txt",
             "object_class": "person"},
            {"name": "Human5", "path": "Human5/img", "startFrame": 1, "endFrame": 713, "nz": 4, "ext": "jpg",
             "anno_path": "Human5/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human6", "path": "Human6/img", "startFrame": 1, "endFrame": 792, "nz": 4, "ext": "jpg",
             "anno_path": "Human6/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human7", "path": "Human7/img", "startFrame": 1, "endFrame": 250, "nz": 4, "ext": "jpg",
             "anno_path": "Human7/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human8", "path": "Human8/img", "startFrame": 1, "endFrame": 128, "nz": 4, "ext": "jpg",
             "anno_path": "Human8/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Human9", "path": "Human9/img", "startFrame": 1, "endFrame": 305, "nz": 4, "ext": "jpg",
             "anno_path": "Human9/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Jump", "path": "Jump/img", "startFrame": 1, "endFrame": 122, "nz": 4, "ext": "jpg",
             "anno_path": "Jump/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "KiteSurf", "path": "KiteSurf/img", "startFrame": 1, "endFrame": 84, "nz": 4, "ext": "jpg",
             "anno_path": "KiteSurf/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Man", "path": "Man/img", "startFrame": 1, "endFrame": 134, "nz": 4, "ext": "jpg",
             "anno_path": "Man/groundtruth_rect.txt",
             "object_class": "face"},
            {"name": "Panda", "path": "Panda/img", "startFrame": 1, "endFrame": 1000, "nz": 4, "ext": "jpg",
             "anno_path": "Panda/groundtruth_rect.txt",
             "object_class": "mammal"},
            {"name": "RedTeam", "path": "RedTeam/img", "startFrame": 1, "endFrame": 1918, "nz": 4, "ext": "jpg",
             "anno_path": "RedTeam/groundtruth_rect.txt",
             "object_class": "vehicle"},
            {"name": "Rubik", "path": "Rubik/img", "startFrame": 1, "endFrame": 1997, "nz": 4, "ext": "jpg",
             "anno_path": "Rubik/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Skater", "path": "Skater/img", "startFrame": 1, "endFrame": 160, "nz": 4, "ext": "jpg",
             "anno_path": "Skater/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Skater2", "path": "Skater2/img", "startFrame": 1, "endFrame": 435, "nz": 4, "ext": "jpg",
             "anno_path": "Skater2/groundtruth_rect.txt",
             "object_class": "person"},
            {"name": "Skating2-1", "path": "Skating2-1/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg",
             "anno_path": "Skating2-1/groundtruth_rect.1.txt",
             "object_class": "person"},
            {"name": "Skating2-2", "path": "Skating2-2/img", "startFrame": 1, "endFrame": 473, "nz": 4, "ext": "jpg",
             "anno_path": "Skating2-2/groundtruth_rect.2.txt",
             "object_class": "person"},
            {"name": "Surfer", "path": "Surfer/img", "startFrame": 1, "endFrame": 376, "nz": 4, "ext": "jpg",
             "anno_path": "Surfer/groundtruth_rect.txt",
             "object_class": "person head"},
            {"name": "Toy", "path": "Toy/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg",
             "anno_path": "Toy/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Trans", "path": "Trans/img", "startFrame": 1, "endFrame": 124, "nz": 4, "ext": "jpg",
             "anno_path": "Trans/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Twinnings", "path": "Twinnings/img", "startFrame": 1, "endFrame": 472, "nz": 4, "ext": "jpg",
             "anno_path": "Twinnings/groundtruth_rect.txt",
             "object_class": "other"},
            {"name": "Vase", "path": "Vase/img", "startFrame": 1, "endFrame": 271, "nz": 4, "ext": "jpg",
             "anno_path": "Vase/groundtruth_rect.txt",
             "object_class": "other"}
        ]

        return sequence_info_list