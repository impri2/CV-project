from board import draw_lines,get_homography_from_image
from dataset import load_rendered_images
import cv2
import os
if __name__ == '__main__':
    filenames,images,labels = load_rendered_images()
    for filename,image,label in zip(filenames,images,labels):
      image = get_homography_from_image(image)
     #image = draw_lines(image)
      cv2.imwrite(os.path.join('homography',filename[-4:]+'.png'),image)
'''

dataset.py: 데이터셋 불러오는 코드
board.py: 이미지 처리하는 코드
board.py, dataset.py,utils.py, main.py가 있는 폴더에 train폴더가 있고, 그 안에 있는 train폴더에 xxxx.png, xxxx.json데이터가 있다.
필요 라이브러리: cv2, scikit-learn, numpy, sympy
'''
