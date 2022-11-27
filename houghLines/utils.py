from sympy import Line
from sympy import abc
import numpy as np
from cv2 import getPerspectiveTransform
import math
import cv2
from skimage import filters
def angle_distance(a,b):
    return min(abs(a-b),math.pi-abs(a-b))

def get_intersection(rho1,theta1,rho2,theta2): #두 선의 교점
    # assume two lines are not parallel
    # intersection between a1x + b1y + c1 = 0 and a2x + b2y + c2 = 0
    a1, b1, c1 = math.cos(theta1), math.sin(theta1), -rho1
    a2, b2, c2 = math.cos(theta2), math.sin(theta2), -rho2

    x = (b1*c2 - b2*c1) / (a1*b2 - a2*b1)
    y = (a2*c1 - a1*c2) / (a1*b2 - a2*b1)

    return [x, y]

def get_homography_from_four_coordinates(coordinates,sx,sy):#coordinates:4 x 2 list
    sorted_coordinates=sorted(coordinates)
    dest_coordinates=[]
    if sorted_coordinates[0][1]<sorted_coordinates[1][1]:
        dest_coordinates.append([[0,0],[0,sy]])
    else:
        dest_coordinates.append([[0,sy],[0,0]])
    if sorted_coordinates[2][1]<sorted_coordinates[3][1]:
        dest_coordinates.append([[sx,0],[sx,sy]])
    else:
        dest_coordinates.append([[sx,sy],[sx,0]])
    return getPerspectiveTransform(np.array(sorted_coordinates,dtype=np.float32).reshape(4,1,2),np.array(dest_coordinates,dtype=np.float32).reshape(4,1,2),)
def get_mean_line(lines):
    rho=0
    theta=0
    for line in lines:
        rho+=line[0][0]
        theta+=line[0][1]
    return rho/len(lines),theta/len(lines)
def canny_h(image):
  
  image = filters.gaussian(image,sigma=2)
  edge = filters.sobel_h(image)

  edge = np.abs(edge)
       
  for i in range(1,edge.shape[0]-1):#Non max suppression
    for j in range(1,edge.shape[1]-1):
      
      if edge[i+1,j]>edge[i,j] or edge[i-1,j]>edge[i,j]:
        edge[i,j]=0

  edge = filters.apply_hysteresis_threshold(edge,0.01,0.03)
  return edge
def canny_v(image):
  
  image = filters.gaussian(image,sigma=2)
  edge = filters.sobel_v(image)

  edge = np.abs(edge)

  for i in range(1,edge.shape[0]-1):
    for j in range(1,edge.shape[1]-1):
     
      if edge[i,j+1]>edge[i,j] or edge[i,j-1]>edge[i,j]:
        edge[i,j]=0

  edge = filters.apply_hysteresis_threshold(edge,0.01,0.03)
  return edge
