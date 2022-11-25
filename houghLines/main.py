from board import draw_lines,get_homography_from_image,get_board
from dataset import load_rendered_images
import cv2
import os
import glob
from skimage import filters
import numpy as np
from skimage import img_as_ubyte


  
if __name__ == '__main__':
    filenames,images,labels = load_rendered_images()
    
    for filename,image,label in zip(filenames,images,labels):
      image,xmax,ymax = get_homography_from_image(image)
      
      cv2.imwrite(os.path.join('homography',filename[-4:]+'.png'),image)
      
      xmin,xmax,ymin,ymax=get_board(image,xmax,ymax)
      cv2.line(image,(80*ymin+640,80*xmin+640),(80*ymax+640,80*xmin+640),(255,0,255),2)
      cv2.line(image,(80*ymin+640,80*xmax+640),(80*ymax+640,80*xmax+640),(255,0,255),2)
      cv2.line(image,(80*ymax+640,80*xmax+640),(80*ymax+640,80*xmin+640),(255,0,255),2)
      cv2.line(image,(80*ymin+640,80*xmin+640),(80*ymin+640,80*xmax+640),(255,0,255),2)
      cv2.imwrite( os.path.join('board',filename[-4:]+'.png'),image)

'''
dataset.py: 데이터셋 불러오는 코드
board.py: 이미지 처리하는 코드
board.py, dataset.py,utils.py, main.py가 있는 폴더에 train폴더가 있고, 그 안에 있는 train폴더에 xxxx.png, xxxx.json데이터가 있다.
같은 폴더 내에 homography 폴더, board 폴더를 만들어 두고 실행해야 함.
homography matrix 찾을 때 반복 횟수는 board.py 49번 줄의 수를 바꾸면 된다.


필요 라이브러리: cv2, scikit-learn, numpy, sympy
'''
