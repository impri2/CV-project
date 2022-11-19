from board import draw_lines,get_homography_from_image,get_board
from dataset import load_rendered_images
import cv2
import os
import glob
for file in glob.glob('homography/*.png'):
  image = cv2.imread(file)
  xmin,xmax,ymin,ymax = get_board(image=image,xmax=2,ymax=2)
  print(xmin,xmax,ymin,ymax)
  greyImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  edge = cv2.Canny(greyImage,140,150)
  cv2.imwrite( os.path.join('canny',file[-8:]),edge)
  cv2.circle(image,(640,640),5,(255,0,255),2,)
  
  cv2.line(image,(80*ymin+640,80*xmin+640),(80*ymax+640,80*xmin+640),(255,0,255),2)
  cv2.line(image,(80*ymin+640,80*xmax+640),(80*ymax+640,80*xmax+640),(255,0,255),2)
  cv2.line(image,(80*ymax+640,80*xmax+640),(80*ymax+640,80*xmin+640),(255,0,255),2)
  cv2.line(image,(80*ymin+640,80*xmin+640),(80*ymin+640,80*xmax+640),(255,0,255),2)
  cv2.imwrite( os.path.join('board',file[-8:]),image)
  
'''if __name__ == '__main__':
    filenames,images,labels = load_rendered_images()
    for filename,image,label in zip(filenames,images,labels):
      #image = get_homography_from_image(image)
      #image = draw_lines(image)
      cv2.imwrite(os.path.join('homography',filename[-4:]+'.png'),image)
'''
'''
dataset.py: 데이터셋 불러오는 코드
board.py: 이미지 처리하는 코드
board.py, dataset.py,utils.py, main.py가 있는 폴더에 train폴더가 있고, 그 안에 있는 train폴더에 xxxx.png, xxxx.json데이터가 있다.
필요 라이브러리: cv2, scikit-learn, numpy, sympy
'''
