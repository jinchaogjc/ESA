import os
import os.path as osp
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import pickle

class vocsbdDataSet(data.Dataset):
    def __init__(
            self,
            data_root,
            data_list,
            max_iters=None,
            num_classes=21,
            split="train",
            transform=None,
            ignore_label=255,
            debug=False,
            # cfg=None,
            # empty=False,
    ):
        # self.active = True if split == 'active' else False
        # if split == 'active':
        #     split = 'train'
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        # self.cfg = cfg
        # self.empty = empty
        with open(data_list, "r") as handle:
            content = handle.readlines()

        self.data_list = []
        
        self.img_ids = [i_id.strip()+".png" for i_id in content]
        self.debug = debug
        if max_iters is not None:
            self.label_to_file, self.file_to_label = pickle.load(open(osp.join(data_root, "vocsbd.pickle"), "rb"))
            if self.debug:
                self.file_to_label = {i:self.file_to_label[i] for i in self.img_ids}
                self.label_to_file = [[] for i in range(self.NUM_CLASS)]
                for i in self.file_to_label.keys():
                    for j in self.file_to_label[i]:
                        self.label_to_file[j].append(i)
            self.img_ids = []
            SUB_EPOCH_SIZE = 3000
            if self.debug:
                SUB_EPOCH_SIZE = 30
            tmp_list = []
            ind = dict()
            for i in range(self.NUM_CLASS):
                ind[i] = 0
            # ind = {i:0 for i in range(self.NUM_CLASS)}
            for e in range(int(max_iters / SUB_EPOCH_SIZE) + 1):
                cur_class_dist = np.zeros(self.NUM_CLASS)
                for i in range(SUB_EPOCH_SIZE):
                    if cur_class_dist.sum() == 0:
                        dist1 = cur_class_dist.copy()
                    else:
                        dist1 = cur_class_dist / cur_class_dist.sum()
                    w = 1 / np.log(1 + 1e-2 + dist1)
                    w = w / w.sum()
                    c = np.random.choice(self.NUM_CLASS, p=w)


                    try:
                        if ind[c] > (len(self.label_to_file[c]) - 1):
                            np.random.shuffle(self.label_to_file[c])
                            ind[c] = ind[c] % (len(self.label_to_file[c]) - 1)
                        # 00029 class 18
                        c_file = self.label_to_file[c][ind[c]]
                    except Exception:
                        pass
                    
                    tmp_list.append(c_file)
                    ind[c] = ind[c] + 1
                    cur_class_dist[self.file_to_label[c_file]] += 1

            self.img_ids = tmp_list

        for name in self.img_ids:
            self.data_list.append(
                {
                    #TODO, JPEGImages
                    "img": os.path.join(self.data_root, "JPEGImages/%s" % name.split('.')[0] + ".jpg"),
                    "label": os.path.join(self.data_root, "SegmentationClassAug/%s" % name),
                    "name": name,
                }
            )
            
        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        # --------------------------------------------------------------------------------
        # A list of all labels
        # --------------------------------------------------------------------------------

        # Please adapt the train IDs as appropriate for your approach.
        # Note that you might want to ignore labels with ID 255 during training.
        # Further note that the current train IDs are only a suggestion. You can use whatever you like.
        # Make sure to provide your results using the original IDs and not the training IDs.
        # Note that many IDs are ignored in evaluation and thus you never need to predict these!

        # labels = [
        #     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        #     Label('unlabeled',                0,      255,    'void',             0,          False,          True,       (0, 0, 0)),
        #     Label('ego vehicle',              1,      255,    'void',             0,          False,          True,       (0, 0, 0)),
        #     Label('rectification border',     2,      255,    'void',             0,          False,          True,       (0, 0, 0)),
        #     Label('out of roi',               3,      255,    'void',             0,          False,          True,       (0, 0, 0)),
        #     Label('static',                   4,      255,    'void',             0,          False,          True,       (0, 0, 0)),
        #     Label('dynamic',                  5,      255,    'void',             0,          False,          True,       (111, 74, 0)),
        #     Label('ground',                   6,      255,    'void',             0,          False,          True,       (81, 0, 81)),
        #     Label('road',                     7,      0,      'flat',             1,          False,          False,      (128, 64, 128)),
        #     Label('sidewalk',                 8,      1,      'flat',             1,          False,          False,      (244, 35, 232)),
        #     Label('parking',                  9,      255,    'flat',             1,          False,          True,       (250, 170, 160)),
        #     Label('rail track',               10,     255,    'flat',             1,          False,          True,       (230, 150, 140)),
        #     Label('building',                 11,     2,      'construction',     2,          False,          False,      (70, 70, 70)),
        #     Label('wall',                     12,     3,      'construction',     2,          False,          False,      (102, 102, 156)),
        #     Label('fence',                    13,     4,      'construction',     2,          False,          False,      (190, 153, 153)),
        #     Label('guard rail',               14,     255,    'construction',     2,          False,          True,       (180, 165, 180)),
        #     Label('bridge',                   15,     255,    'construction',     2,          False,          True,       (150, 100, 100)),
        #     Label('tunnel',                   16,     255,    'construction',     2,          False,          True,       (150, 120, 90)),
        #     Label('pole',                     17,     5,      'object',           3,          False,          False,      (153, 153, 153)),
        #     Label('polegroup',                18,     255,    'object',           3,          False,          True,       (153, 153, 153)),
        #     Label('traffic light',            19,     6,      'object',           3,          False,          False,      (250, 170, 30)),
        #     Label('traffic sign',             20,     7,      'object',           3,          False,          False,      (220, 220, 0)),
        #     Label('vegetation',               21,     8,      'nature',           4,          False,          False,      (107, 142, 35)),
        #     Label('terrain',                  22,     9,      'nature',           4,          False,          False,      (152, 251, 152)),
        #     Label('sky',                      23,     10,     'sky',              5,          False,          False,      (70, 130, 180)),
        #     Label('person',                   24,     11,     'human',            6,          True,           False,      (220, 20, 60)),
        #     Label('rider',                    25,     12,     'human',            6,          True,           False,      (255, 0, 0)),
        #     Label('car',                      26,     13,     'vehicle',          7,          True,           False,      (0, 0, 142)),
        #     Label('truck',                    27,     14,     'vehicle',          7,          True,           False,      (0, 0, 70)),
        #     Label('bus',                      28,     15,     'vehicle',          7,          True,           False,      (0, 60, 100)),
        #     Label('caravan',                  29,     255,    'vehicle',          7,          True,           True,       (0, 0, 90)),
        #     Label('trailer',                  30,     255,    'vehicle',          7,          True,           True,       (0, 0, 110)),
        #     Label('train',                    31,     16,     'vehicle',          7,          True,           False,      (0, 80, 100)),
        #     Label('motorcycle',               32,     17,     'vehicle',          7,          True,           False,      (0, 0, 230)),
        #     Label('bicycle',                  33,     18,     'vehicle',          7,          True,           False,      (119, 11, 32)),
        #     Label('license plate',             -1,    -1,     'vehicle',          7,          False,          True,       (0, 0, 142)),
        # ]

        # GTAV
        self.id_to_trainid = {
            7: 0,
            8: 1,
            11: 2,
            12: 3,
            13: 4,
            17: 5,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            31: 16,
            32: 17,
            33: 18,
        }
        self.trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "terrain",
            10: "sky",
            11: "person",
            12: "rider",
            13: "car",
            14: "truck",
            15: "bus",
            16: "train",
            17: "motocycle",
            18: "bicycle",
        }
        if self.NUM_CLASS == 16:  # SYNTHIA
            self.id_to_trainid = {
                7: 0,
                8: 1,
                11: 2,
                12: 3,
                13: 4,
                17: 5,
                19: 6,
                20: 7,
                21: 8,
                23: 9,
                24: 10,
                25: 11,
                26: 12,
                28: 13,
                32: 14,
                33: 15,
            }
            self.trainid2name = {
                0: "road",
                1: "sidewalk",
                2: "building",
                3: "wall",
                4: "fence",
                5: "pole",
                6: "light",
                7: "sign",
                8: "vegetation",
                9: "sky",
                10: "person",
                11: "rider",
                12: "car",
                13: "bus",
                14: "motocycle",
                15: "bicycle",
            }
        if self.NUM_CLASS == 21:  # VOCSBD
            self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20}
            self.trainid2name = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
        self.transform = transform

        self.ignore_label = ignore_label


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = np.array(Image.open(datafiles["label"]), dtype=np.uint8)
        # name = datafiles["name"]

        # re-assign labels to match the format of Cityscapes
        # label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)
        # for k, v in self.id_to_trainid.items():
        #     label_copy[label == k] = v

        # label = Image.fromarray(label_copy)

        if self.transform is not None:
            image, label = self.transform(image, label)

        ret_data = {
            "img": image,
            'label': label,
            'index': index,
            'datafiles': datafiles,
        }
        return ret_data