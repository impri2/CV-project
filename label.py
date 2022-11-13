import json
import glob

import os
import cv2

with open('train/_annotations.createml.json') as f:
    labels = json.load(f)

for i,label in enumerate(labels):
    file = label["image"]
    fileName= os.path.join("train",file)
    image=cv2.imread(fileName)
    for annotation in label["annotations"]:
        coordinates=annotation["coordinates"]
        
        x = int(coordinates["x"])
        y = int(coordinates["y"])
        width = coordinates["width"]
        height = coordinates["height"]
        x1,x2,y1,y2=int(x-width/2),int(x+width/2),int(y-width/2),int(y+width/2)
       # cv2.circle(image,(x,y),50,(0,1,255),3)
      #  cv2.rectangle(image,(x1,y1),(x2,y2),(255,128,0),2)
    fileName = os.path.join('labeled',str(i)+".png")
    
    cv2.imwrite(fileName,image)