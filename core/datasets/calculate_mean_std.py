import numpy as np
import glob
import tqdm
import os
import cv2
from torchvision import transforms
from PIL import Image

def show_normalized_image():
    image = Image.open("2007_002055.jpg")
    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]
    transform = transforms.ToTensor()
    images = transform(image)#.permute(1,2,0)
    image = F.normalize(images, mean=PIXEL_MEAN, std=PIXEL_STD)
    # TODO delete
    import torchvision.transforms as transforms
    from PIL import Image
    
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(image)
    
    pil_image.save("image_normalize2.jpg")
    
    
if __name__ == "__main__":
    # input = ["/home/jc/Data/GTAV/images_subset/*.png"]
    # -----------------------select dataset---------------------
    dataset = 'voc12'
    if dataset == 'gtav':
        input = ["/home/jc/Data/GTAV/images/*.png"]
    elif dataset == 'coco':
        input = ["/home/jc/Codes/ripu/datasets/coco/train2017/*.jpg"]
    elif dataset == 'voc12':
        input = ["/home/jc/Codes/ripu/datasets/VOCdevkit/VOC2012/JPEGImages/*.jpg"]
    else:
        print("please select gtav or coco or voc12")
    input = glob.glob(os.path.expanduser(input[0]))
    print(input)
    
    transform = transforms.ToTensor()
    
    mean = np.zeros(3)
    std = np.zeros(3)
    count = 0
    for image_path in tqdm.tqdm(input):
        image = np.array(transform(Image.open(image_path)).permute(1,2,0))
        mean += np.mean(image, axis=(0,1))
        std += np.std(image, axis=(0,1))
        count += 1
    mean /= count
    std /= count
    # images1 = [np.array(Image.open(image_path)) for image_path in input]
    # images = [np.array(transform(Image.open(image_path)).permute(1,2,0)) for image_path in input]
    # images = np.stack(images)
    # mean = np.mean(images, axis=(0,1,2))
    # std = np.std(images, axis=(0,1,2))
    print("mean:", mean)
    print("std:", std)
    # for path in tqdm.tqdm(input):
    #     pass
        # print(path)
        # images = 
        
    
    #_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
    #_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
    
    
    # calculate 0-200 image of GTAV
    # mean: [0.4920809 0.4680615 0.4333721]
    # std: [0.27053967 0.2627568  0.2521798 ]
    # mean: [0.4920832  0.46805956 0.43337135]
    # std: [0.23586252 0.22937044 0.22416745]
    
    # mean: [0.49537453 0.47597364 0.4413587 ]
    # std: [0.26796493 0.26299366 0.25487134]
    
    # calculate 24000 images of GTAV
    # mean: [0.44279135 0.43852617 0.42521505]
    # std: [0.23788825 0.23334042 0.23139835]
    
    