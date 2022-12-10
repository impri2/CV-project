import json
import glob
import os
import cv2
import pathlib
import random
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
def load_images(path):
    files = glob.glob(os.path.join(path,"*.png"))
    files.sort()
    images=[]
    filenames=[]
    labels=[]
    for i,file in enumerate(files):
        images.append(cv2.imread(file))
        filenames.append(pathlib.Path(file).name)
        if i> 200:
            break
    return filenames,images
def load_random_images(path,number):
    files = glob.glob(os.path.join(path,"*.jpg"))
    files.extend(glob.glob(os.path.join(path,"*.png")))
    sample = random.sample(files,number)
    images=[]
    filenames=[]
    for file in sample:
        images.append(cv2.imread(file))
        filenames.append(pathlib.Path(file).name)
    return filenames,images