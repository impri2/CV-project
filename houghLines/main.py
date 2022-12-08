from board import detect_board,detect_board_by_dl
from dataset import load_rendered_images,load_images
import cv2
import os
import sys
from tqdm import tqdm
import glob
from skimage import filters
import numpy as np
from skimage import img_as_ubyte
import torch
import model as model
import torchvision.transforms
from PIL import Image
'''
dataset.py: 데이터셋 불러오는 코드
board.py: 이미지 처리하는 코드
board.py, dataset.py,utils.py, main.py가 있는 폴더에 train폴더가 있고, 그 안에 있는 train폴더에 xxxx.png, xxxx.json데이터가 있다.
같은 폴더 내에 homography 폴더, board 폴더를 만들어 두고 실행해야 함.
homography matrix 찾을 때 반복 횟수는 board.py 49번 줄의 수를 바꾸면 된다.


필요 라이브러리: cv2, scikit-learn, numpy, sympy
'''
# mps is for M1/M2 mac devices. May be changed to cpu or cuda
# .pth file is for pre-trained parameters
DEVICE = 'mps'
def main(argv):
    filenames, images,labels = load_images(argv[0] if len(argv) > 0 else None)
    trained_model = model.ChessModel().to(DEVICE)
    
    trained_model.load_state_dict(torch.load('ChessModel5.pth',map_location=torch.device(DEVICE)))
    trained_model.eval()
    t = tqdm(zip(filenames, images), total=len(filenames))

    for filename, image in t:
        t.set_description("Processing: " + filename)
        '''if not os.path.exists(os.path.join('yes',filename[-4:])):
            os.makedirs(os.path.join('yes',filename[-4:]))
        if not os.path.exists(os.path.join('no',filename[-4:])):
            os.makedirs(os.path.join('no',filename[-4:]))'''
        warped_image, H, corners = detect_board_by_dl(image,trained_model,DEVICE)
        '''x = 1919
        y = 1919
        for i in range(4):
         corner = H@np.array([[label[i][0]],[label[i][1]],[1]])
         corner/=corner[2]
         corner[0]/=80
         corner[0,0]=round(corner[0,0])
         corner[0]*=80
         corner[1]/=80
         corner[1,0]=round(corner[1,0])
         corner[1]*=80
         x=min(x,corner[0,0])
         y=min(y,corner[1,0])
        print(corner)'''
        cv2.imwrite(os.path.join('homography', filename[-4:]+'.png'), warped_image)
        
        
        x1, y1, x2, y2 = corners
        x1, y1, x2, y2 = corners
        cv2.line(warped_image, (x1, y1), (x2, y1), (255, 0, 255), 2)
        cv2.line(warped_image, (x1, y1), (x1, y2), (255, 0, 255), 2)
        cv2.line(warped_image, (x2, y1), (x2, y2), (255, 0, 255), 2)
        cv2.line(warped_image, (x1, y2), (x2, y2), (255, 0, 255), 2)
        
        
        cv2.imwrite(os.path.join('board2', filename),warped_image)       
if __name__ == "__main__":
    main(sys.argv[1:])
    #print(torch.cuda.is_available())
