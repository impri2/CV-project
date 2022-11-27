import json
import glob
import os
import cv2

def load_rendered_images(path):
    filenames=[]
    images=[]
    labels=[]

    default_path = os.path.join("train","train")
    file_path = os.path.join(default_path if path is None else path, "*.json")

    files = glob.glob(file_path)

    for file in files:
        with open(file) as f:
            image = file[:-5]+'.png'
            images.append(cv2.imread(image))

            label = json.load(f)['corners']
            labels.append(label)

            filenames.append(file[:-5])
            if int(filenames[len(filenames)-1][-4:])>=100:
                break
       
    return filenames,images,labels
