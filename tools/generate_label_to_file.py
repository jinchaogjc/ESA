from tools.utils import *
import tqdm
import cv2
import numpy as np
import pickle

if __name__ == '__main__':
   
    # vocsbdPath = "/home/jc/Codes/ripu/datasets/VOCdevkit/VOC2012/JPEGImages_vocsdb"
    vocsbdPath = "datasets/VOCdevkit/VOC2012/"
    gtDIR = "SegmentationClassAug"
    data_type = "datasets/vocsbd_train_list_subset.txt"
    opt_file = "datasets/VOCdevkit/VOC2012/vocsbd.pickle"
    CLASS_NUM = 21
    vocsbd_img_list = get_files_from_txt(data_type)
    file_to_label = {}
    label_to_file = [[] for i in range(CLASS_NUM)]
    
    x=()
    for vocsbd_img in tqdm.tqdm(vocsbd_img_list):
        image_name = vocsbd_img+".png"
        img = cv2.imread(os.path.join(vocsbdPath, gtDIR, image_name))
        # print(img.shape)
        labels = np.unique(img).tolist()
        file_to_label[image_name] = labels
        for l in labels:
            label_to_file[l].append(image_name)
    
    x=x+(label_to_file,)
    x=x+(file_to_label,)
    with open(opt_file, "wb") as f:
        pickle.dump(x, f)
    
    with open(opt_file, "rb") as f:
        x = pickle.load(f)
        pass