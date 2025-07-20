import os
import numpy as np
import torch
# import argparse
from torch.utils import data
from PIL import Image
# from . import transform
# import torch.nn.functional as F
from tools.utils import resize_tsr_img, resize_cv_img
from torchvision.transforms import functional as F

class VOC12DataSet(data.Dataset):
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
            cfg=None,
            empty=False,
            source_dataset="coco",
            # ES_LABEL="SegmentationClassAug_es"
    ):
        self.active = True if split == 'active' else False
        if split == 'active':
            split = 'train'
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.cfg = cfg
        self.empty = empty
        self.source_dataset = source_dataset
        self.valmode = 'val' in data_list
        with open(data_list, "r") as handle:
            content = handle.readlines()

        self.data_list = []
        if empty:
            self.data_list.append(
                {
                    "img": "",
                    "label": "",
                    "label_mask": "",
                    "name": "",
                }
            )
        else:
            for fname in content:
                name = fname.strip()
                self.data_list.append(
                    {
                        "img": os.path.join(
                            self.data_root, "JPEGImages/%s" % (name+".jpg")
                        ),
                        "label": os.path.join(
                            self.data_root,
                            "SegmentationClassAug/%s" % ( name + ".png"),
                        ),
                        # mode in val, remove label_es
                        
                        "label_es": os.path.join(
                            self.data_root,
                            cfg.ACTIVE.ES_LABEL+"/%s"
                            % (name + "_gtFine_labelTrainIds.png",
                            ),
                        ),
                        "label_mask": os.path.join(
                            self.cfg.OUTPUT_DIR,
                            "gtMask/%s/%s"
                            % (
                                self.split,
                                name + "_gtFine_labelIds.png",
                            ),
                        ),
                        # "label_mask": os.path.join(
                        #     self.cfg.OUTPUT_DIR,
                        #     "gtMask/%s/%s"
                        #     % ( "train", name + ".png"),
                            
                        # ),
                        "name": name,
                        'indicator': os.path.join(
                            cfg.OUTPUT_DIR,
                            "gtIndicator/%s/%s"
                            % (
                                "train",
                                name + "_indicator.pth",
                            ),
                        )
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

        # VOC
        self.trainid2name = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
        # GTAV
        # self.id_to_trainid = {
        #     7: 0,
        #     8: 1,
        #     11: 2,
        #     12: 3,
        #     13: 4,
        #     17: 5,
        #     19: 6,
        #     20: 7,
        #     21: 8,
        #     22: 9,
        #     23: 10,
        #     24: 11,
        #     25: 12,
        #     26: 13,
        #     27: 14,
        #     28: 15,
        #     31: 16,
        #     32: 17,
        #     33: 18,
        # }
        
        # self.trainid2name = {
        #     0: "road",
        #     1: "sidewalk",
        #     2: "building",
        #     3: "wall",
        #     4: "fence",
        #     5: "pole",
        #     6: "light",
        #     7: "sign",
        #     8: "vegetation",
        #     9: "terrain",
        #     10: "sky",
        #     11: "person",
        #     12: "rider",
        #     13: "car",
        #     14: "truck",
        #     15: "bus",
        #     16: "train",
        #     17: "motocycle",
        #     18: "bicycle",
        # }
        # if self.NUM_CLASS == 16:  # SYNTHIA
        #     self.id_to_trainid = {
        #         7: 0,
        #         8: 1,
        #         11: 2,
        #         12: 3,
        #         13: 4,
        #         17: 5,
        #         19: 6,
        #         20: 7,
        #         21: 8,
        #         23: 9,
        #         24: 10,
        #         25: 11,
        #         26: 12,
        #         28: 13,
        #         32: 14,
        #         33: 15,
        #     }
        #     self.trainid2name = {
        #         0: "road",
        #         1: "sidewalk",
        #         2: "building",
        #         3: "wall",
        #         4: "fence",
        #         5: "pole",
        #         6: "light",
        #         7: "sign",
        #         8: "vegetation",
        #         9: "sky",
        #         10: "person",
        #         11: "rider",
        #         12: "car",
        #         13: "bus",
        #         14: "motocycle",
        #         15: "bicycle",
        #     }
        # if self.NUM_CLASS == 21:  # VOCSBD
        #     self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20}
        #     self.trainid2name = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
        
        
        # if self.source_dataset == "coco":
        #     self.id_to_trainid = {0: 0, 1: 5, 2: 2, 3: 15, 4: 9, 5: 40, 6: 6, 7: 3, 8: 16, 9: 57, 10: 20, 11: 61, 12: 17, 13: 18, 14: 4, 15: 1, 16: 59, 17: 19, 18: 58, 19: 7, 20: 63}
        #     # TODO change class follow id_to_trainid
        #     # change COCO id 5, trainid2name[5]="aeroplane", output class aeroplane.
        #     # self.trainid2name = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
        #     self.trainid2name = {0: 'background', 
        #                          5: 'aeroplane', 
        #                          2: 'bicycle', 
        #                          15: 'bird', 
        #                          9: 'boat', 
        #                          40: 'bottle', 
        #                          6: 'bus', 
        #                          3: 'car', 
        #                          16: 'cat', 
        #                          57: 'chair', 
        #                          20: 'cow', 
        #                          61: 'diningtable', 
        #                          17: 'dog', 
        #                          18: 'horse', 
        #                          4: 'motorbike', 
        #                          1: 'person', 
        #                          59: 'pottedplant', 
        #                          19: 'sheep', 
        #                          58: 'sofa', 
        #                          7: 'train', 
        #                          63: 'tvmonitor'}
    
        
        self.transform = transform

        self.ignore_label = ignore_label

        self.debug = debug

    def __len__(self):
        return len(self.data_list)
    
    
    def __getitem__(self, index):
        # if self.debug:
        #     index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        origin_shape = image.size
        # label = np.array(Image.open(datafiles["label"]), dtype=np.uint8)
        label = Image.open(datafiles["label"])
        
        # image = F.resize(image, (512, 512), Image.BICUBIC)
        # label = F.resize(label, (512, 512), Image.BICUBIC)
        image = resize_cv_img(image)
        label = resize_cv_img(label)
        
        label = np.array(label, dtype=np.uint8)
        
        label_mask = None
        if self.split == 'train':
            # label_mask = np.array(Image.open(datafiles["label_mask"]), dtype=np.uint8)
            label_mask = Image.open(datafiles["label_mask"])
            label_mask = resize_cv_img(label_mask)
            label_mask = np.array(label_mask, dtype=np.uint8)
        else:
            # test or val, mask is useless
            label_mask = np.ones_like(label, dtype=np.uint8) * 255

        # for generate new mask
        origin_mask = torch.from_numpy(label_mask).long()

        active_indicator = torch.tensor([0])
        active_selected = torch.tensor([0])
        if self.active:
            indicator = torch.load(datafiles['indicator'])
            active_indicator = indicator['active']
            active_selected = indicator['selected']
            # if first time load, initialize it
            if active_indicator.size() == (1,):
                active_indicator = torch.zeros_like(origin_mask, dtype=torch.bool)
                active_selected = torch.zeros_like(origin_mask, dtype=torch.bool)

        # re-assign labels to match the format of VOC
        # label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)
        # print(np.unique(label))
        # for k, v in self.id_to_trainid.items():
            # label_copy[label == k] = v
        # label = np.array(label_copy, dtype=np.uint8)

        origin_label = torch.from_numpy(label).long()

        label.resize(label.shape[0], label.shape[1], 1)
        label_mask.resize(label_mask.shape[0], label_mask.shape[1], 1)

        h, w = label.shape[0], label.shape[1]

        if label.shape != label_mask.shape:
            print("err")
        mask_aggregation = np.concatenate((label, label_mask), axis=2)
        mask_aggregation = Image.fromarray(mask_aggregation)

        if self.transform is not None:
            image, mask_aggregation = self.transform(image, mask_aggregation)
            label = mask_aggregation[:, :, 0]
            label_mask = mask_aggregation[:, :, 1]

        # label_es = np.array(Image.open(datafiles["label_es"]), dtype=np.uint8)
        # if not self.valmode:
        #     label_es = Image.open(datafiles["label_es"])
        #     label_es = resize_cv_img(label_es)
        #     label_es = np.array(label_es, dtype=np.uint8)
        # else:
        #     label_es = torch.zeros_like(label_mask)   
        
        
        origin_mask = resize_tsr_img(origin_mask)
        origin_label = resize_tsr_img(origin_label)
        
        
        ret_data = {
        "img": image,  # data
        'label': label,  # for test
        'mask': label_mask,  # for train
        'name': datafiles['name'],  # for test to store the results
        'path_to_mask': datafiles['label_mask'],  # for active to store new mask
        'path_to_indicator': datafiles['indicator'],  # store new indicator
        'size': torch.tensor([h, w]),  # for active to interpolate the output to original size
        'origin_mask': origin_mask,  # mask without transforms for active
        'origin_label': origin_label,  # label without transforms for active
        'active': active_indicator,  # indicate region or pixels can not be selected
        'selected': active_selected, # indicate the pixel have been selected, can calculate the class-wise ratio of selected samples
        # 'label_es': label_es,
        'origin_shape': origin_shape
        }
        return ret_data
    
    
if __name__ == "__main__":
    import sys
    sys.path.append('../datasets')
    import transform
    import cv2
    
    # (args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
    #                            split=mode, transform=transform, debug=cfg.DEBUG)
    w, h = (1280, 720)
    
    trans_list = [
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_bgr255=False)
    ]
    trans_list = [transform.Resize((h, w)), ] + trans_list
    trans = transform.Compose(trans_list)
    
    import sys
    sys.path.append("/home/jc/Codes/ripu")
    from core.configs import cfg
    
    config_file = "configs/coco/e024_deeplabv3plus_r101_cocovoc_PA_40_d0.yaml"
    # parser = argparse.ArgumentParser(description="Active Domain Adaptive Semantic Segmentation Training")
    # args = parser.parse_args()
    # args.OUTPUT_DIR = "results/v3plus_cocovoc_pa_40"
    cfg.merge_from_file(config_file)
    # cfg.merge_from_list(args.OUTPUT_DIR)
    cfg.OUTPUT_DIR = "results/e155_deeplabv3plus_r101_cocovoc_SAM_all_d0"
    voc_dataset = VOC12DataSet(data_root='datasets/VOCdevkit/VOC2012', 
                               data_list='datasets/voc_train_list.txt',
                               max_iters=62500,
                               transform=trans,
                               cfg=cfg)
    
    for batch_index, tgt_data in enumerate(voc_dataset):
        print(batch_index)
        # print(tgt_data['img'].shape)
        # print(tgt_data['label'].shape)
        # print(tgt_data['mask'].shape)
        # print(tgt_data['origin_mask'].shape)
        # print(tgt_data['origin_label'].shape)
        # print(tgt_data['active'].shape)
        # print(tgt_data['selected'].shape)
        # print(tgt_data['name'])
        # print(tgt_data['path_to_mask'])
        # print(tgt_data['path_to_indicator'])
        # print(tgt_data['size'])
        # print(tgt_data['origin_shape'])
        # break
    # gtav_dataset = GTAVDataSet(data_root='datasets/gtav', 
    #                            data_list='datasets/gtav_train_list.txt',
    #                            # max_iters=62500,
    #                            transform=trans)

    # Select a sample from the dataset
    sample = voc_dataset[0]
    img = sample['img']
    
    import torchvision.transforms as transforms
    from PIL import Image
    
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(img)
    
    pil_image.save("voc_sample_image.jpg")
    # print(cv2.imwrite("gtav_sample_image.jpg", np.array(img)))
    label = sample['label']
    print(cv2.imwrite("voc_sample_label.jpg", np.array(label)))
    # print(sample['index'])
    # print(sample['datafiles'])
    # Access the image and label of the sample
    # image, label = sample

    # Plot the image
    # plt.imshow(img[0, :, :], cmap='gray')
    # plt.title(f'Label: {label}')
    # plt.show()