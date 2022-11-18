from sympy import Line
from sympy import abc
import numpy as np
from cv2 import getPerspectiveTransform
import math
def angle_distance(a,b):
    return min(abs(a-b),math.pi-abs(a-b))
def get_intersection(rho1,theta1,rho2,theta2): #두 선의 교점
    line1 = Line(math.cos(theta1)*abc.x + math.sin(theta1)*abc.y -rho1)
    line2 = Line(math.cos(theta2)*abc.x + math.sin(theta2)*abc.y -rho2)
    intersection = line1.intersection(line2)
    return intersection[0].coordinates
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
