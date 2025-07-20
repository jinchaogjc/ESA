import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange
import os
import copy
from pycocotools.coco import COCO
from pycocotools import mask
from torchvision import transforms
# import coco_transforms as tr
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


import os
import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image, ImageFile
import pickle
import torch


import torchvision
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True


class COCODataSetVOCTrainId(Dataset):
    VOC_NUM_CLASSES = 21
    def __init__(self,
                 data_root,
                 data_list,
                 max_iters=None,
                 num_classes=81,
                 split="train",
                 year='2017',
                 transform=None,
                 ignore_label=255,
                 debug=False, 
                 batch_size=512,
                 crop_size=512,
                 coco_others_label=0):
        # if debug=True, it will only load 5000, train images.
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.data_list = []
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.coco_others_label = coco_others_label
        self.preprocess_data_list = data_list
        with open(data_list, "r") as handle:
            content = handle.readlines() # handle.read().splitlines()
        self.img_ids = [i_id.strip() for i_id in content]
        self.debug = debug
        if max_iters is not None:
            # num_classes_new = self.NUM_CLASS
            num_classes_new = COCODataSetVOCTrainId.VOC_NUM_CLASSES - 1
            self.label_to_file, self.file_to_label = pickle.load(open(osp.join("", "datasets/coco_label_info.pickle"), "rb"))
            if self.debug:
                self.label_to_file, self.file_to_label = pickle.load(open(osp.join("", "datasets/coco_label_info_debug.pickle"), "rb"))
                self.file_to_label = {i:self.file_to_label[i] for i in self.img_ids}
                self.label_to_file = [[] for i in range(num_classes_new)]
                for i in self.file_to_label.keys():
                    for j in self.file_to_label[i]:
                        self.label_to_file[j].append(i)
            # del self.label_to_file[0]
            # num_classes_new = COCODataSet.VOC_NUM_CLASSES - 1
            self.img_ids = []
            SUB_EPOCH_SIZE = 3000
            if self.debug:
                SUB_EPOCH_SIZE = 30
            tmp_list = []
            ind = dict()
            for i in range(num_classes_new):
                ind[i] = 0
            # ind = {i:0 for i in range(num_classes_new)}
            for e in range(int(max_iters / SUB_EPOCH_SIZE) + 1):
                cur_class_dist = np.zeros(num_classes_new)
                for i in range(SUB_EPOCH_SIZE):
                    if cur_class_dist.sum() == 0:
                        dist1 = cur_class_dist.copy()
                    else:
                        dist1 = cur_class_dist / cur_class_dist.sum()
                    w = 1 / np.log(1 + 1e-2 + dist1)
                    w = w / w.sum()
                    c = np.random.choice(num_classes_new, p=w)

                    # c -= 1
                    if ind[c] > (len(self.label_to_file[c]) - 1):
                        np.random.shuffle(self.label_to_file[c])
                        ind[c] = ind[c] % (len(self.label_to_file[c]) - 1)
                    # 00029 class 18
                    if debug:
                        if len(self.label_to_file[c]) == 0:
                            continue
                    c_file = self.label_to_file[c][ind[c]]
                    tmp_list.append(c_file)
                    ind[c] = ind[c] + 1
                    cur_class_dist[self.file_to_label[c_file]] += 1

            self.img_ids = tmp_list

        for name in self.img_ids:
            self.data_list.append(
                {
                    "img": os.path.join(self.data_root, "images", split + year+ "/%s" % name),
                    "label": os.path.join(self.data_root, "labels", split + year+ "/%s" % name),
                    "name": name,
                }
            )

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
  
        # [0,5,2,15,9,40,6,3,16,57,20,61,17,18,4,1,59,19,58,7,63]
        self.id_to_trainid = {  0:  0,
                                5:  1,
                                2:  2,
                                15: 3,
                                9:  4,
                                40: 5,
                                6:  6,
                                3:  7,
                                16: 8,
                                57: 9,
                                20: 10,
                                61: 11,
                                17: 12,
                                18: 13,
                                4:  14,
                                1:  15,
                                59: 16,
                                19: 17,
                                58: 18,
                                7:  19,
                                63: 20}
        
       
        self.trainid2name = {
            0: 'background',
            1: 'aeroplane',
            2: 'bicycle',
            3: 'bird',
            4: 'boat',
            5: 'bottle',
            6: 'bus',
            7: 'car',
            8: 'cat',
            9: 'chair',
            10: 'cow',
            11: 'diningtable',
            12: 'dog',
            13: 'horse',
            14: 'motorbike',
            15: 'person',
            16: 'pottedplant',
            17: 'sheep',
            18: 'sofa',
            19: 'train',
            20: 'tvmonitor',
        }
        
     
        self.transform = transform
        self.ignore_label = ignore_label

        self.initcoco()

    def initcoco(self,
                 base_dir="datasets/coco",
                 split='train',
                 year='2017'):
        # super().__init__()
        ann_file = os.path.join(base_dir, 'annotations/instances_{}{}.json'.format(split, year))
        ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}.pth'.format(split, year))
        self.img_dir = os.path.join(base_dir, 'images/{}{}'.format(split, year))
        self.split = split
        self.coco = COCO(ann_file)
        self.coco_mask = mask
        if self.debug:
            ids_file = os.path.join(base_dir, 'annotations/{}_ids_{}'.format(split, year)+'_debug.pth')
        if os.path.exists(ids_file):
            self.ids = torch.load(ids_file)
        else:
            # ids = list(self.coco.imgs.keys())
            # self.ids = self._preprocess(ids, ids_file)
            
            self.ids = self._preprocess(ids_file)
        

    # def get_cat_ids(self):
        # CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        #        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        #        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        #        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        #        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        #        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        #        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        #        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        #        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        #        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        #        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        #        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        #        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        #        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
        # from pycocotools.coco import COCO as _COCO
        # # cat_ids = _COCO.getCatIds(cat_names=CLASSES, sup_names=[], cat_ids=[])
        # cat_ids = _COCO.getCatIds(self, catNums=CLASSES, supNms=[], catIds=[])
        # _COCO.getCatIds(catNms=[], supNms=[], catIds=[])
        # cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        # return cat2label
        
    def getitem(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        # sample = {'image': _img, 'label': _target}
        return _img, _target
        # if self.split == "train":
        #     return self.transform_tr(sample)
        # elif self.split == 'val':
        #     return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        coco = self.coco
        # img_id = self.ids[index]
        img_id = int(self.data_list[index]['name'].split(".")[0])
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _target = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))

        return _img, _target

    def save_trainlist(self, input):
        index = 0
    
        with open("datasets/coco_train_list_subset.txt", "w") as file:
            for path in input:
                print(path.split('/')[-1])
                file.write(str(path.split('/')[-1]) + "\n")
                index += 1
                if index == 5000:
                    break
                
    # def _preprocess(self, ids, ids_file):
    #     print("Preprocessing mask, this will take a while. " + \
    #           "But don't worry, it only run once for each split.")
    #     tbar = trange(len(ids))
    #     new_ids = []
        
        
    #     for i in tbar:
    #         img_id = ids[i]
    #         cocotarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
    #         img_metadata = self.coco.loadImgs(img_id)[0]
    #         mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
    #                                   img_metadata['width'])
            
    #         # TODO remove image only contain background
    #         tmp_labels = list(self.id_to_trainid.keys())
    #         del tmp_labels[0]
    #         if not bool(set(np.unique(mask)) & set(tmp_labels)):
    #             continue
            
    #         # more than 1k pixels
    #         if self.debug:
    #             if (mask > 0).sum() > 1000:
    #                 new_ids.append(img_id)
    #         else:
    #             new_ids.append(img_id)
    #         tbar.set_description('Doing: {}/{}, got {} qualified images'. \
    #                              format(i, len(ids), len(new_ids)))
            
    #         if self.debug:
    #             if len(new_ids) > 5000:
    #                 break
                
    #     if self.debug:
    #         trainlist = ["{:012d}.jpg".format(id) for id in new_ids]
    #         self.save_trainlist(trainlist)
            
    #     print('Found number of qualified images: ', len(new_ids))
    #     torch.save(new_ids, ids_file)
    #     return new_ids

    def _preprocess(self, ids_file):
        print("Preprocessing mask, this will take a while. " + \
              "But don't worry, it only run once for each split.")
        import sys
        sys.path.append("/home/jc/Codes/ripu")
        from core.datasets.utils import get_files_ids_from_txt
        new_ids = get_files_ids_from_txt(self.preprocess_data_list)
        torch.save(new_ids, ids_file)
        return new_ids
    
    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            
            # if cat in [44, 62, 64, 63, 72]:
            #     print("contain")
            #     input("contain 44 62 64 63 72, ")
            # reindex, from 90 label to 81, 0 is background, 1-80
            # category_to_label = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79}
            category_to_label = {0:0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
            c = category_to_label[cat]
            # if cat in self.CAT_LIST:
            #     c = self.CAT_LIST.index(cat)
            # else:
            #     continue
            # c = cat
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        # print(np.unique(mask))
        return mask

    # def transform_tr(self, sample):
    #     composed_transforms = transforms.Compose([
    #         tr.RandomHorizontalFlip(),
    #         tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.crop_size),
    #         tr.RandomGaussianBlur(),
    #         tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         tr.ToTensor()])

    #     return composed_transforms(sample)

    # def transform_val(self, sample):

    #     composed_transforms = transforms.Compose([
    #         tr.FixScaleCrop(crop_size=self.crop_size),
    #         tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         tr.ToTensor()])

    #     return composed_transforms(sample)


    # def len(self):
    #     return len(self.ids)


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # if self.debug:
        #     index = 0
        datafiles = self.data_list[index]

        # image = Image.open(datafiles["img"]).convert('RGB')
        # label = np.array(Image.open(datafiles["label"]), dtype=np.uint8)
        
        image, label = self.getitem(index)
        # name = datafiles["name"]
        # image = image.convert('RGB')
        label = np.array(label, dtype=np.uint8)
        # re-assign labels to match the format of Cityscapes
        # from coco_id to voc_id, and the (rest of 21-79 of coco_id to 255 first?)
        # label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)
        
        label_copy = copy.copy(label)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        # and the (rest of 21-80 of coco_id to 255 first?) or set them to 0 as background.
        for idx in range(len(self.id_to_trainid.items()), self.NUM_CLASS):
            label_copy[label_copy == idx] = self.coco_others_label
        
        label = Image.fromarray(label_copy)

        if self.transform is not None:
            image, label = self.transform(image, label)

        ret_data = {
            "img": image,
            'label': label,
            'index': index,
            'datafiles': datafiles,
        }

        return ret_data

def test_coco(coco_dataset):
    import cv2
    # gtav_dataset = GTAVDataSet(data_root='datasets/gtav', 
    #                            data_list='datasets/gtav_train_list.txt',
    #                            max_iters=62500,
    #                            transform=trans)

    # Select a sample from the dataset
    sample = coco_dataset[0]
    img = sample['img']
    
    import torchvision.transforms as transforms
    from PIL import Image
    
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(img)
    
    pil_image.save("coco_sample_image.jpg")
    # print(cv2.imwrite("gtav_sample_image.jpg", np.array(img)))
    label = sample['label']
    print(cv2.imwrite("coco_sample_label.jpg", np.array(label)))
    print(sample['index'])
    print(sample['datafiles'])
    
    
if __name__ == "__main__":
    import coco_transforms as tr
    from tools.utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    # import argparse

    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # args.base_size = 513
    # args.crop_size = 513
    import sys
    sys.path.append('../datasets')
    import transform
    
    w, h = (1280, 720)
    
    trans_list = [
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_bgr255=False)
    ]
    trans_list = [transform.Resize((h, w)), ] + trans_list
    trans = transform.Compose(trans_list)
    
    # coco_val = COCOSegmentation(args, split='val', year='2017')
    debug = False
    if debug:
        data_list="datasets/coco_train_list_subset.txt"
    else:
        data_list="datasets/coco_train_list.txt"
        
    coco_val = COCODataSetVOCTrainId(data_root="datasets/coco", 
                                data_list=data_list, 
                                split='train',
                                year="2017",
                                max_iters=62500,
                                batch_size=512,
                                crop_size=512,
                                transform=trans,
                                debug=debug,
                                coco_others_label=0)
    # coco_val.get_cat_ids()
    test_coco(coco_val)
    dataloader = DataLoader(coco_val, batch_size=1, shuffle=True, num_workers=0)

    # for ii, sample in enumerate(dataloader):
    #     for jj in range(sample["img"].size()[0]):
    #         # print(ii)
    #         img = sample['img'].numpy()
    #         image = img[0]
    #         gt = sample['label'].numpy()
    #         label = gt[0]
    #         labels = np.unique(label)
    #         if 5 in labels or 9 in labels or 16 in labels or 18 in labels or 20 in labels:
    #                 print("Contain: "+ str(labels))
    #                 # input("contain 5,9,16,18,20")
                    
    for ii, sample in enumerate(dataloader):
        print(ii)
        if ii == 3:
            print("")
        for jj in range(sample["img"].size()[0]):
            img = sample['img'].numpy()
            image1 = img[0]
            gt = sample['label'].numpy()
            gt1 = gt[0]
            if not len(np.unique(gt1)) == 1:
                continue
            print(np.unique(gt1))
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='coco')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)
            plt.show()
        # if ii == 1:
        #     break

    plt.show(block=True)