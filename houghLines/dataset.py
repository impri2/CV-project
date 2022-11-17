import json
import glob
import os
import cv2
def load_images():
    filenames=[]
    images=[]
    labels=[]
    files = glob.glob(os.path.join("train","train",'*.json'))
    for file in files:
     with open(file) as f:
                
        label = json.load(f)['corners']
       
        image = file[:-5]+'.png'
        filenames.append(file[:-5])
        images.append(cv2.imread(image))
        labels.append(label)
        if int(filenames[len(filenames)-1][-4:])>=100:
            break
       
    return filenames,images,labels