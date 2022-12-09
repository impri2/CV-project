from board import detect_board
from dataset import load_rendered_images
from utils import draw_corners
import cv2
import os
import sys
from tqdm import tqdm
import glob
from skimage import filters
import numpy as np
from skimage import img_as_ubyte

'''
dataset.py: 데이터셋 불러오는 코드
board.py: 이미지 처리하는 코드
board.py, dataset.py,utils.py, main.py가 있는 폴더에 train폴더가 있고, 그 안에 있는 train폴더에 xxxx.png, xxxx.json데이터가 있다.
같은 폴더 내에 homography 폴더, board 폴더를 만들어 두고 실행해야 함.
homography matrix 찾을 때 반복 횟수는 board.py 49번 줄의 수를 바꾸면 된다.


필요 라이브러리: cv2, scikit-learn, numpy, sympy
'''

def main(argv):
    filenames, images, labels = load_rendered_images(argv[0] if len(argv) > 0 else None)

    t = tqdm(zip(filenames, images, labels), total=len(filenames))

    for filename, image, label in t:
        t.set_description("Processing: " + filename)

        warped_image, H, corners = detect_board(image)

        cv2.imwrite(os.path.join('homography', filename[-4:]+'.png'), warped_image)

        warped_image = draw_corners(warped_image, corners)

        cv2.imwrite(os.path.join('board', filename[-4:]+'.png'), warped_image)

if __name__ == "__main__":
    main(sys.argv[1:])