from tools.utils import *
import tqdm
import cv2
import numpy as np
import pickle
# from coco_segmentation import COCOSegmentation
from coco import COCODataSet
from tools.utils import get_files_from_txt
from tools.utils import generate_coco_train_list
    
def generate_coco_label_jump_bg():
    
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
    # coco_val = COCOSegmentation(data_root="datasets/coco", 
    #                             data_list="datasets/coco_train_list.txt", 
    #                             split='train',
    #                             year="2017",
    #                             batch_size=512,
    #                             crop_size=512,
    #                             transform=trans,
    #                             debug=True)
    debug=True
    generate_coco_train_list(debug)
    
    if debug:
        data_list="datasets/coco_train_list_subset.txt"
    else:
        data_list="datasets/coco_train_list.txt"
        
    coco_val = COCODataSet(data_root="datasets/coco", 
                                data_list=data_list, 
                                split='train',
                                year="2017",
                                batch_size=512,
                                crop_size=512,
                                transform=trans,
                                debug=debug,
                                coco_others_label=0
                                )
    
    # from coco_segmentation import COCOSegmentation
    # import argparse

    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # args.base_size = 513
    # args.crop_size = 513
    # coco_val = COCOSegmentation(args, split='val', year='2017')

    # test_coco(coco_val)
    dataloader = DataLoader(coco_val, batch_size=1, shuffle=False, num_workers=0)

    # for ii, sample in enumerate(dataloader):
    #     for jj in range(sample["img"].size()[0]):
    #         image = sample['img'].numpy()
    #         # image1 = img[0]
    #         label = sample['label'].numpy()
            # gt1 = gt[0]
    #         tmp = np.array(gt[jj]).astype(np.uint8)
    #         segmap = decode_segmap(tmp, dataset='coco')
    #         img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
    #         img_tmp *= (0.229, 0.224, 0.225)
    #         img_tmp += (0.485, 0.456, 0.406)
    #         img_tmp *= 255.0
    #         img_tmp = img_tmp.astype(np.uint8)
    #         plt.figure()
    #         plt.title('display')
    #         plt.subplot(211)
    #         plt.imshow(img_tmp)
    #         plt.subplot(212)
    #         plt.imshow(segmap)

    #     if ii == 1:
    #         break

    # plt.show(block=True)
    
    # vocsbdPath = "/home/jc/Codes/ripu/datasets/VOCdevkit/VOC2012/JPEGImages_vocsdb"
    # vocsbdPath = "datasets/VOCdevkit/VOC2012/"
    # gtDIR = "SegmentationClassAug"
    coco_path = "datasets/coco/images/train2017"
    # coco_train_list = "datasets/coco_train_list_subset.txt"
    coco_train_list = data_list
    if debug:
        CLASS_NUM = 81
        opt_file = "datasets/coco_label_info_debug.pickle"
    else:
        CLASS_NUM = 21
        opt_file = "datasets/coco_label_info.pickle"
    
    coco_img_list = get_files_from_txt(coco_train_list)
    file_to_label = {}
    label_to_file = [[] for i in range(CLASS_NUM)]
    
    x=()
    # for vocsbd_img in tqdm.tqdm(vocsbd_img_list):
    train_list = []
    # leave background out, just save 1-80, 
    for ii, sample in enumerate(dataloader):
        print(ii)
        # for jj in range(sample["img"].size()[0]):
        # image = sample['image'].numpy()
        image = sample['img'].numpy()
        image = image[0]
        #cv2.imwrite("test6.png", sample['img'][0].permute(1,2,0).numpy())
        label = sample['label'].numpy()
        label = label[0]
        # image_name = vocsbd_img+".png"
        image_name = sample['datafiles']['name'][0]
        image_path = sample['datafiles']['img'][0]
        # image_name = str(ii) + '.jpg'
        # # img = cv2.imread(image_path)
        # if debug and (not image_name in coco_img_list):
        #     continue
            
        # print(img.shape)
        labels = np.unique(label).tolist()
        
        # if 5 in labels or 9 in labels or 16 in labels or 18 in labels or 20 in labels:
        #     print("Contain")
        #     input("contain 5,9,16,18,20")
        # leave the images only contain background out
        if len(labels) == 1 and labels[0] == 0:
            continue
             
        train_list.append(image_name)         
        
        if len(labels) > 1 and labels[0] == 0:
            del labels[0]
                
        file_to_label[image_name] = [i - 1 for i in labels]
        for l in labels:
            label_to_file[l].append(image_name)
    del label_to_file[0]
    x=x+(label_to_file,)
    x=x+(file_to_label,)
    with open(opt_file, "wb") as f:
        pickle.dump(x, f)
    
    with open(opt_file, "rb") as f:
        x = pickle.load(f)
        pass
    
    
    # save_file = "datasets/coco_train_list.txt"
    # if debug:
    #     save_file = "datasets/coco_train_list_subset.txt"
    save_file = data_list
    index = 0
    with open(save_file, "w") as file:
        for path in tqdm.tqdm(train_list):
            print(path.split('/')[-1])
            file.write(str(path.split('/')[-1]) + "\n")
            if debug:
                index += 1
                if index == 5000:
                    break
        
if __name__ == '__main__':
    
    
    # import coco_transforms as tr
    # from utils import decode_segmap
    from torch.utils.data import DataLoader
    # from torchvision import transforms
    # import matplotlib.pyplot as plt
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
    # coco_val = COCOSegmentation(data_root="datasets/coco", 
    #                             data_list="datasets/coco_train_list.txt", 
    #                             split='train',
    #                             year="2017",
    #                             batch_size=512,
    #                             crop_size=512,
    #                             transform=trans,
    #                             debug=True)
    debug=False
    generate_coco_train_list(debug)
    
    if debug:
        data_list="datasets/coco_train_list_subset.txt"
    else:
        data_list="datasets/coco_train_list.txt"
        
    coco_val = COCODataSet(data_root="datasets/coco", 
                                data_list=data_list, 
                                split='train',
                                year="2017",
                                batch_size=512,
                                crop_size=512,
                                transform=trans,
                                debug=debug,
                                # coco_others_label=0
                                )
    
    # from coco_segmentation import COCOSegmentation
    # import argparse

    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # args.base_size = 513
    # args.crop_size = 513
    # coco_val = COCOSegmentation(args, split='val', year='2017')

    # test_coco(coco_val)
    dataloader = DataLoader(coco_val, batch_size=1, shuffle=False, num_workers=0)

    # for ii, sample in enumerate(dataloader):
    #     for jj in range(sample["img"].size()[0]):
    #         image = sample['img'].numpy()
    #         # image1 = img[0]
    #         label = sample['label'].numpy()
            # gt1 = gt[0]
    #         tmp = np.array(gt[jj]).astype(np.uint8)
    #         segmap = decode_segmap(tmp, dataset='coco')
    #         img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
    #         img_tmp *= (0.229, 0.224, 0.225)
    #         img_tmp += (0.485, 0.456, 0.406)
    #         img_tmp *= 255.0
    #         img_tmp = img_tmp.astype(np.uint8)
    #         plt.figure()
    #         plt.title('display')
    #         plt.subplot(211)
    #         plt.imshow(img_tmp)
    #         plt.subplot(212)
    #         plt.imshow(segmap)

    #     if ii == 1:
    #         break

    # plt.show(block=True)
    
    # vocsbdPath = "/home/jc/Codes/ripu/datasets/VOCdevkit/VOC2012/JPEGImages_vocsdb"
    # vocsbdPath = "datasets/VOCdevkit/VOC2012/"
    # gtDIR = "SegmentationClassAug"
    coco_path = "datasets/coco/images/train2017"
    # coco_train_list = "datasets/coco_train_list_subset.txt"
    coco_train_list = data_list
    CLASS_NUM = 81
    if debug:
        
        opt_file = "datasets/coco_label_info_debug.pickle"
    else:
        # CLASS_NUM = 21
        opt_file = "datasets/coco_label_info.pickle"
    
    coco_img_list = get_files_from_txt(coco_train_list)
    file_to_label = {}
    label_to_file = [[] for i in range(CLASS_NUM)]
    
    x=()
    # for vocsbd_img in tqdm.tqdm(vocsbd_img_list):
    train_list = []
    # leave background out, just save 1-80, 
    for ii, sample in enumerate(dataloader):
        print(ii)
        # for jj in range(sample["img"].size()[0]):
        # image = sample['image'].numpy()
        image = sample['img'].numpy()
        image = image[0]
        #cv2.imwrite("test6.png", sample['img'][0].permute(1,2,0).numpy())
        label = sample['label'].numpy()
        label = label[0]
        # image_name = vocsbd_img+".png"
        image_name = sample['datafiles']['name'][0]
        image_path = sample['datafiles']['img'][0]
        # image_name = str(ii) + '.jpg'
        # # img = cv2.imread(image_path)
        # if debug and (not image_name in coco_img_list):
        #     continue
            
        # print(img.shape)
        labels = np.unique(label).tolist()
        
        # if 5 in labels or 9 in labels or 16 in labels or 18 in labels or 20 in labels:
        #     print("Contain")
        #     input("contain 5,9,16,18,20")
        # leave the images only contain background out
        # if len(labels) == 1 and labels[0] == 0:
        #     continue
             
        train_list.append(image_name)         
        
        # if len(labels) > 1 and labels[0] == 0:
        #     del labels[0]
                
        file_to_label[image_name] = [i for i in labels]
        for l in labels:
            label_to_file[l].append(image_name)
    # del label_to_file[0]
    x=x+(label_to_file,)
    x=x+(file_to_label,)
    with open(opt_file, "wb") as f:
        pickle.dump(x, f)
    
    with open(opt_file, "rb") as f:
        x = pickle.load(f)
        pass
    
    
    # save_file = "datasets/coco_train_list.txt"
    # if debug:
    #     save_file = "datasets/coco_train_list_subset.txt"
    save_file = data_list
    index = 0
    with open(save_file, "w") as file:
        for path in tqdm.tqdm(train_list):
            print(path.split('/')[-1])
            file.write(str(path.split('/')[-1]) + "\n")
            if debug:
                index += 1
                if index == 5000:
                    break
        