import json
import glob
import os
import cv2
import numpy as np
import copy
import sys
corners=[[ 355, 112 ],[1576, 118 ], [  1868, 1206 ], [184, 1228]]
with open('train/_annotations.createml.json') as f:
    labels = json.load(f)

newLabels=[]
template=[]
for i,label in enumerate(labels):
    
    file = label["image"]
    fileName= os.path.join("train",file)
    image=cv2.imread(fileName)
    for j in range(4):
        cv2.imwrite(os.path.join('corners',"corner"+str(j+1)+".png"),image[corners[j][1]-8:corners[j][1]+8,corners[j][0]-8:corners[j][0]+8])
    break
for i,label in enumerate(labels):
    
    file = label["image"]
    fileName= os.path.join("train",file)
    image=cv2.imread(fileName)
    for j in range(4):
        template = cv2.imread(os.path.join('corners',"corner"+str(j+1)+".png"))
        res = cv2.matchTemplate(image,template,cv2.TM_SQDIFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        x,y = minLoc
        cv2.rectangle(image,(x,y),(x+10,y+10),(0,255,255),2)
    cv2.imwrite(os.path.join('corners',str(i)+".png"),image)

