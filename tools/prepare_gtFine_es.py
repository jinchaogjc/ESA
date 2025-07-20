import sys
import os
import glob
import numpy as np
from PIL import Image


if __name__ == "__main__":
    cityscapesPath = "/home/jc/Documents/Data/cityscapes/gtFine_es_vis"
    # searchFine   = os.path.join( cityscapesPath , "labels", "*.png" )
    searchFine   = os.path.join( cityscapesPath , "train", "*", "*_gtFine_labelTrainIds.png" )
    # searchFine   = os.path.join( cityscapesPath , "gtFine", "*.png" )
    files = glob.glob( searchFine )
    print("Processing {} annotation files".format(len(files)))
    files.sort()
    print(files)
    # iterate through files
    progress = 0
    print("Progress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
    for f in files:
        # os.system("ls")
        label_f = f.replace("gtFine_es_vis", "gtFine_es")
        filename = os.path.basename(label_f)
        dir = label_f[0:-len(filename)]
        if not os.path.exists(dir):
            os.mkdir(dir)
        os.system("cp "+f+" "+label_f)
        
        # status
        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()