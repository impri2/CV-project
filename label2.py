import json
import glob
import os
import cv2
import numpy as np
dest=[[0,0],[640,0],[640,640],[0,640]]
def getChessBoardCoordinates(image):
    mouseclicks=[]
    def click(event,x,y,flags,param):
        if event == cv2.EVENT_FLAG_LBUTTON:
            if len(mouseclicks)<4:
                mouseclicks.append([x,y])
                
    cv2.namedWindow('board', cv2.WINDOW_NORMAL)
    
    cv2.imshow("board",image)
    cv2.setMouseCallback("board",click)
    cv2.waitKey()
    return mouseclicks
with open('train/_annotations.createml.json') as f:
    labels = json.load(f)

newLabels=[]
for i,label in enumerate(labels):
    file = label["image"]
    fileName= os.path.join("train",file)
    image=cv2.imread(fileName)
    boardCoordinates = getChessBoardCoordinates(image)
    matrix = np.array( cv2.getPerspectiveTransform(np.array(boardCoordinates, dtype=np.float32),np.array(dest, dtype=np.float32)))
    warpedImage = cv2.warpPerspective(image,matrix,(image.shape[1],image.shape[0]))
    newLabel = {"boardCoordinates":boardCoordinates,"board":[['']*8 for i in range(8)]}
    for annotation in label["annotations"]:
        coordinates=annotation["coordinates"]
        
        x = int(coordinates["x"])
        y = int(coordinates["y"])
        width = coordinates["width"]
        height = coordinates["height"]
        pieceCoordinates = np.array([x,y+height/2,1])
        
        warpedPieceCoordinates = matrix @ pieceCoordinates
        warpedPieceCoordinates /= warpedPieceCoordinates[2]
        
        cv2.circle(warpedImage,(int(warpedPieceCoordinates[0]),int(warpedPieceCoordinates[1])),10,(0,1,255),3)
        fileName = os.path.join('result',str(i)+".png")
       
        newLabel['board'][int(warpedPieceCoordinates[1]/80)][int(warpedPieceCoordinates[0]/80)]=annotation['label']
    newLabels.append(newLabel)
    with open(os.path.join('results','label.json'),'w') as f:
        json.dump(newLabels,f,indent=2)
    