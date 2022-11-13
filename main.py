import cv2
import glob
import os
import  numpy as np
import json
mouseclicks=[]
dest=[[0,0],[640,0],[640,640],[0,640]]
def click(event,x,y,flags,param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        if len(mouseclicks)<4:
            mouseclicks.append([x,y])
            print([x,y])


files = glob.glob('labeled/*.png')
with open('train/_annotations.createml.json') as f:
    labels = json.load(f)
newLabels=[]
for i,file in enumerate(files):
    mouseclicks=[]
    image=cv2.imread(file)
    fileName=os.path.join('results',str(i)+".png")
    cv2.imwrite(fileName,image)
    height, width, channel = image.shape
    cv2.namedWindow('board', cv2.WINDOW_NORMAL)
    
    cv2.imshow("board",image)
    cv2.setMouseCallback("board",click)
    cv2.waitKey()
    matrix = cv2.getPerspectiveTransform(np.array(mouseclicks, dtype=np.float32),np.array(dest, dtype=np.float32))
    print(matrix)
    fileName = os.path.join('result',str(i)+".png")
    
    cv2.imwrite(fileName,cv2.warpPerspective(image,matrix,(width,height)))
    
