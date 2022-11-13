import json
import glob
import os
import cv2
import numpy as np
import copy
import sys
import chess
import chess.svg
dest=[[0,0],[640,0],[640,640],[0,640]]
FENNotation={
    "white-king":"K",
    "black-king":"k",
    "white-queen":"Q",
    "black-queen":"q",
    "white-rook":"R",
    "black-rook":"r",
    "white-bishop":"B",
    "black-bishop":"b",
    "white-knight":"N",
    "black-knight":"n",
    "white-pawn":"P",
    "black-pawn":"p"
    
}

def generateChessBoardImage(board,name):
    chessBoard = chess.Board(fen=None)
    for i in range(8):
        for j in range(8):
            if board[i][j]=='':
                continue
            chessBoard.set_piece_at(square=chess.square(j,7-i),piece=chess.Piece.from_symbol(FENNotation[board[i][j]]))
    chessBoardImage=chess.svg.board(board=chessBoard)
    with open(os.path.join('boards',name+'.svg'),'w') as f:
        f.write(chessBoardImage)
def getChessBoardCoordinates(image):
    originalImage=copy.deepcopy(image)
    mouseclicks=[[ 355, 112 ],[1576, 118 ], [  1868, 1206 ], [184, 1228]]
    moving = -1
    def click(event,x,y,flags,param):
        nonlocal moving
        if event == cv2.EVENT_FLAG_LBUTTON:
            for i in range(4):
                if abs(mouseclicks[i][0]-x)<5 and abs(mouseclicks[i][1]-y)<5:
                    moving = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE:
            if moving != -1:
                mouseclicks[moving]=[x,y]
                drawCircles()
        else:
            moving = -1
    def drawCircles():
        nonlocal originalImage
        image=originalImage
        originalImage=copy.deepcopy(image)
        
        for i in range(4):
            cv2.circle(image,mouseclicks[i],10,(0,255,25),3)    
        cv2.imshow('board',image)
    cv2.namedWindow('board', cv2.WINDOW_NORMAL)
    
    drawCircles()
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
    #boardCoordinates = getChessBoardCoordinates(image)
    boardCoordinates=[[ 355, 112 ],[1576, 118 ], [  1868, 1206 ], [184, 1228]]
    matrix = np.array( cv2.getPerspectiveTransform(np.array(boardCoordinates, dtype=np.float32),np.array(dest, dtype=np.float32)))
    warpedImage = cv2.warpPerspective(image,matrix,(image.shape[1],image.shape[0]))
    
    newLabel = {"boardCoordinates":boardCoordinates,"board":[['']*8 for i in range(8)]}
    for annotation in label["annotations"]:
        coordinates=annotation["coordinates"]
        
        x = int(coordinates["x"])
        y = int(coordinates["y"])
        width = coordinates["width"]
        height = coordinates["height"]
        pieceCoordinates = np.array([x,y+height*0.9/2,1])
        
        warpedPieceCoordinates = matrix @ pieceCoordinates
        warpedPieceCoordinates /= warpedPieceCoordinates[2]
        
        cv2.circle(warpedImage,(int(warpedPieceCoordinates[0]),int(warpedPieceCoordinates[1])),10,(0,1,255),3)
        fileName = os.path.join('result',str(i)+".png")
        
        newLabel['board'][int(warpedPieceCoordinates[1]/80)][int(warpedPieceCoordinates[0]/80)]=annotation['label']
        cv2.putText(warpedImage,annotation['label'],(int(warpedPieceCoordinates[0]/80)*80+40,int(warpedPieceCoordinates[1]/80)*80+40),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,255,255))
    cv2.imwrite(os.path.join('rectified',str(i)+".png"),warpedImage[0:650,0:650])
    newLabels.append(newLabel)

    with open(os.path.join('label','label.json'),'w') as f:
        json.dump(newLabels,f,indent=2)
    generateChessBoardImage(newLabel['board'],str(i))
    