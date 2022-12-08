import os
import cv2
import dataset
import numpy as np
import math
import random
from functools import cmp_to_key
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from utils import *

def get_intersections(lines1,lines2):
    coordinates = []
    for line1 in lines1:
        for line2 in lines2:  
            coordinates.append(get_intersection(line1[0][0],line1[0][1],line2[0][0],line2[0][1]))
    return coordinates

# Using lines1, cluster similar lines in lines2 into one
def merge_lines(lines1,lines2):
    intersections=[]
    mergedLines=[]

    for line in lines2: #lines1의 임의의 선을 골라 그 선과의 교점을 구함(논문에서는 lines1의 평균을 냄)
        intersections.append(get_intersection(lines1[0][0][0],lines1[0][0][1],line[0][0],line[0][1]))

    cluster = DBSCAN(eps=20,min_samples=2)
    cluster.fit(intersections)#교점과 가까운 것기리 merge됨
    label_cnt = {}
    rho_sum = {}
    angle_sum = {}

    # collect lines with same labels and average them
    for i in range(len(lines2)):
        if cluster.labels_[i] == -1: # not labeled
            mergedLines.append(lines2[i])
        else:
            label = cluster.labels_[i]

            if not label in label_cnt:
                label_cnt[label] = 0
                rho_sum[label] = 0
                angle_sum[label] = 0

            label_cnt[label] += 1
            if lines2[i][0][0] < 0: # always consider rho to be positive
                rho_sum[label] += -lines2[i][0][0]
                angle_sum[label] += lines2[i][0][1] + math.pi
            else:
                rho_sum[label] += lines2[i][0][0]
                angle_sum[label] += lines2[i][0][1]

    # calculate average line
    for i in label_cnt:
        rho = rho_sum[i] / label_cnt[i]
        theta = (angle_sum[i] / label_cnt[i]) % (2 * math.pi)

        if theta > math.pi:
            theta -= math.pi
            rho = -rho

        new_line = np.array([[rho, theta]])  # (1, 2) dimension for some reason
        mergedLines.append(new_line)

    # print(cluster.labels_)
    # print(intersections)
    return mergedLines

def get_homography(lines1, lines2, gamma=0.02):
    intersections = get_intersections(lines1, lines2)
    N = len(intersections)
    max_inlier_set=[]
    max_inlier_warped=[]

    max_cell_size = 4

    for X in range(len(lines1)-1):# 정렬순으로 선택하기
        for Y in range(len(lines2)-1):
            chosenLines1 = lines1[X:X+2]
            chosenLines2 = lines2[Y:Y+2]
            line_intersection = []
            for line1 in chosenLines1:# 선택한 선의 교점
                for line2 in chosenLines2:
                    line_intersection.append(get_intersection(line1[0][0],line1[0][1],line2[0][0],line2[0][1]))

            # 구한 교점이 이루는 사각형이 1x1, 1x2 ... max_cell_size x max_cell_size
            # 크기라고 가정하고 homography를 구함
            for sx in range(1, max_cell_size + 1):
                for sy in range(1, max_cell_size + 1):
                    inliers=[]
                    inliers_warped=[]

                    homography = get_homography_from_four_coordinates(line_intersection,sx,sy)

                    for i in range(N): #모든 교점에 대해 warp
                        warped= homography @ np.array([[intersections[i][0]],[intersections[i][1]],[1]])
                        warped/=warped[2]

                        dist = math.hypot(warped[0][0]-round(warped[0][0]),warped[1][0]-round(warped[1][0]))#한 칸이 1x1크기가 되도록 warp 했으므로 inlier는 좌표가 정수에 가까워야 함
                        if dist<gamma: #가장 가까운 정수좌표로부터 거리가 gamma 이하라면 inlier에 추가
                            inliers.append(intersections[i])
                            inliers_warped.append((round(warped[0][0]),round(warped[1][0])))

                    if len(inliers)>len(max_inlier_set):
                        max_inlier_set = inliers
                        max_inlier_warped = inliers_warped
                    #최대 inlier수가 N/2에 도달할 때까지 반복
                    if len(max_inlier_set)>N//2:
                         break
                if len(max_inlier_set)>N//2:
                    break
            if len(max_inlier_set)>N//2:
                break

    iter = 300
    for x in range(iter):
        if len(max_inlier_set) > N//2:
            break

        #임의로 가로선 2개, 세로선 2개 선택
        chosenLines1 = random.sample(lines1,2)
        chosenLines2 = random.sample(lines2,2)
        line_intersection = []

        for line1 in chosenLines1:# 선택한 선의 교점
            for line2 in chosenLines2:
                line_intersection.append(get_intersection(line1[0][0],line1[0][1],line2[0][0],line2[0][1]))

        for sx in range(1, max_cell_size + 1):
            for sy in range(1, max_cell_size + 1): #구한 교점이 이루는 사각형이 1x1, 1x2 ... 8x8크기라고 가정하고 homography를 구함
                inliers=[]
                inliers_warped=[]
                
                homography = get_homography_from_four_coordinates(line_intersection,sx,sy)
                
                for i in range(N): #모든 교점에 대해 warp
                  
                  warped= homography @ np.array([[intersections[i][0]],[intersections[i][1]],[1]])
                  warped/=warped[2]
                 
                  dist = math.hypot(warped[0][0]-round(warped[0][0]),warped[1][0]-round(warped[1][0]))#한 칸이 1x1크기가 되도록 warp 했으므로 inlier는 좌표가 정수에 가까워야 함
                  if dist<gamma: #가장 가까운 정수좌표로부터 거리가 gamma 이하라면 inlier에 추가
                      inliers.append(intersections[i])
                      inliers_warped.append((round(warped[0][0]),round(warped[1][0])))
                
                if len(inliers)>len(max_inlier_set):
                    max_inlier_set = inliers 
                    max_inlier_warped = inliers_warped
                #최대 inlier수가 N/2에 도달할 때까지 반복
                if len(max_inlier_set)>N//2:
                     break
            if len(max_inlier_set)>N//2:
                break
        if len(max_inlier_set)>N//2:
            break

    xmin, xmax, ymin, ymax = min([i[0] for i in max_inlier_warped]),\
                             max([i[0] for i in max_inlier_warped]),\
                             min([i[1] for i in max_inlier_warped]),\
                             max([i[1] for i in max_inlier_warped])

    for i in range(len(max_inlier_warped)):
        max_inlier_warped[i]=( max_inlier_warped[i][0]-(xmin),  max_inlier_warped[i][1]-(ymin)) # xmin, ymin을 0으로 함

    xmax -= xmin
    ymax -= ymin
    # print(xmax,ymax)

    homography = cv2.findHomography(np.array(max_inlier_set,dtype=np.float32),80*np.array(max_inlier_warped,dtype=np.float32)+640)[0]
   
    return homography, xmax, ymax

def get_lines(image, debug=False):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(img_gray, (3, 3))
    canny = cv2.Canny(img_blur, threshold1=120, threshold2=150)

    # if debug:
    #     open_wait_cv2_window("canny", canny)

    lines = cv2.HoughLines(canny, 1, np.pi / 180, threshold=100)

    # ideally, 8x8 board consists of 9+9 lines
    # but consider cluster of similar lines which are to be merged
    max_lines = 30

    return lines[:min(max_lines,len(lines))]

#선을 가로선 세로선으로 clustering
def cluster_lines(lines):
    thetas = []
    dist = []
    for i in range(len(lines)):
        for rho,theta in lines[i]:
            thetas.append(theta)
    for i in range(len(lines)): #모든 직선 쌍에 대해 각도 거리를 구함
        dist.append([])
        for j in range(len(lines)):
            dist[i].append(angle_distance(thetas[i],thetas[j]))
    
    cluster = AgglomerativeClustering(n_clusters=2,affinity='precomputed',linkage='average')
    cluster.fit(dist)
    lines1=[]
    lines2=[]
    for i in range(len(lines)):
        if cluster.labels_[i]==0:
            lines1.append(lines[i])
        else:
             lines2.append(lines[i])
    return lines1,lines2

# rectified 이미지로 보드 위치 구하기
# returns real corner coordinate (x1, y1, x2, y2) on 1920x1920 image
def get_board(image, xmax, ymax):
    greyImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edgeH = canny_h(greyImage)
    edgeV = canny_v(greyImage)
    
    xmin=0
    ymin=0
    while xmax-xmin<8:
        xmax_edge=0
        xmin_edge=0
        for i in range(1920):
            for j in range(-3,4):
                if 0<=(xmax+1)*80+640+j and (xmax+1)*80+640+j<1920:
                 xmax_edge+=edgeH[(xmax+1)*80+640+j,i]
                if 0<=(xmin-1)*80+640+j and (xmin-1)*80+640+j<1920:
                 xmin_edge+=edgeH[(xmin-1)*80+640+j,i]
        if xmax_edge>xmin_edge:
            xmax+=1
        else:
            xmin-=1
        # print(xmax,xmin)
    while ymax-ymin<8:
        ymax_edge=0
        ymin_edge=0
        for i in range(1920):
            for j in range(-3,4):
             if 0<=(ymax+1)*80+640+j and (ymax+1)*80+640+j<1920:
              ymax_edge+=edgeV[i,(ymax+1)*80+640+j]
             if 0<=(ymin-1)*80+640+j and (ymin-1)*80+640+j<1920:
              ymin_edge+=edgeV[i,(ymin-1)*80+640+j]
        if ymax_edge>ymin_edge:
            ymax+=1
        else:
            ymin-=1

    print((xmin, ymin, xmax, ymax))
    # convert to real coordinate on 1920 x 1920 image
    return np.array((xmin, ymin, xmax, ymax)) * 80 + 640

#구한 선과 교점을 그리기
def draw_lines(image, lines1, lines2):

    for seq in (lines1, lines2):
        for i in range(len(seq)):
            rho, theta = seq[i][0]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 - 1000*b), int(y0 + 1000*a))
            pt2 = (int(x0 + 1000*b), int(y0 - 1000*a))
            cv2.line(image, pt1, pt2, (0, 20 * i % 256, (255 - 20 * i) % 256), 1, cv2.LINE_AA)

    return image

# input: image
# output: homography, xmax, ymax
# xmax and ymax is right-bottom corner coordinate for naively detected board
def get_homography_from_image(image, debug=False):
    lines = get_lines(image, debug=debug)

    # cluster into vertical and horizontal and merge similar ones
    lines1, lines2 = cluster_lines(lines)

    if debug:
        open_wait_cv2_window("lines", draw_lines(image.copy(), lines1, lines2))

    lines2 = merge_lines(lines1,lines2)
    lines1 = merge_lines(lines2,lines1)

    # sort lines so that neighboring lines are adjacent in the list
    cmp = lambda line: line[0][0] if line[0][1] <= math.pi/2 else -line[0][0]
    lines1.sort(key=cmp)
    lines2.sort(key=cmp)

    homography, xmax, ymax = get_homography(lines1, lines2)

    if debug:
        open_wait_cv2_window("lines_merged", draw_lines(image.copy(), lines1, lines2))
        if xmax > 15 or ymax > 15: # this is very unlikely to happen
            print("Something has gone wrong with lines")
            print(lines1)
            print(lines2)

    return homography, xmax, ymax

# rescale image so that width * height is fixed
def resize_img(image):
    res = (1200 * 800) * 0.9

    h, w, _ = image.shape

    scale = math.sqrt(res / (h * w))
    new_dim = (int(scale * w), int(scale * h)) # be cautious: format is (width, height)

    return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

# output: warp_image, homography, corner_coordinates
# warped image: warped image
# homography: 3x3 homography matrix from input image to warped image
# corner_coordinates: four corners on the warped image (x1, y1, x2, y2)
#                     where (x1, y1) is top-left, (x2, y2) is bottom-right corner
def detect_board(image, debug=False):
    image = resize_img(image)

    # this xmax, ymax is prematurely computed board boarder (right-bottom lines)
    # to convert to real coordinate: xmax * 80 + 640
    homography, xmax, ymax = get_homography_from_image(image, debug=debug)

    if debug:
        print("premature xmax, ymax = %d, %d" % (int(xmax), int(ymax)))

    warped_image = cv2.warpPerspective(image, homography, (1920, 1920))

    corners = get_board(warped_image, int(xmax), int(ymax))
    print(corners)

    return warped_image, homography, corners



