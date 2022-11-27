import os
import cv2
import dataset
import numpy as np
import math
import random
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

from utils import *
def get_intersections(lines1,lines2):
    coordinates = []
    for line1 in lines1:
        for line2 in lines2:  
            coordinates.append(get_intersection(line1[0][0],line1[0][1],line2[0][0],line2[0][1]))
    return coordinates
def merge_lines(lines1,lines2):
    #lines2를 merge하는 함수
    intersections=[]
    mergedLines=[]
    for line in lines2: #lines1의 임의의 선을 골라 그 선과의 교점을 구함(논문에서는 lines1의 평균을 냄)
        intersections.append(get_intersection(lines1[0][0][0],lines1[0][0][1],line[0][0],line[0][1]))
    cluster = DBSCAN(eps=20,min_samples=2)
    cluster.fit(intersections)#교점과 가까운 것기리 merge됨
    check = set() 
    for i in range(len(lines2)):
        if cluster.labels_[i]==-1:
            mergedLines.append(lines2[i])
        else:
            if cluster.labels_[i] in check: #각 cluster의 첫 선만 포함 (-1을 분류되지 않은 것)
                continue
            else:
                check.add(cluster.labels_[i])
                mergedLines.append(lines2[i])

    print(cluster.labels_)
    print(intersections)
    return mergedLines
def get_homography(lines1,lines2,gamma=0.02):
    lines2 = merge_lines(lines1,lines2)
    lines1 = merge_lines(lines2,lines1)
    intersections = get_intersections(lines1,lines2)
    N = len(intersections)
    max_inlier_set=[]
    max_inlier_warped=[]

    for x in range(20):#may be changed
        #임의로 가로선 2개, 세로선 2개 선택
        chosenLines1 = random.sample(lines1,2)
        chosenLines2 = random.sample(lines2,2)
        line_intersection = []
        for line1 in chosenLines1:# 선택한 선의 교점
            for line2 in chosenLines2:
                line_intersection.append(get_intersection(line1[0][0],line1[0][1],line2[0][0],line2[0][1]))
        for sx in range(1,9):
            for sy in range(1,9): #구한 교점이 이루는 사각형이 1x1, 1x2 ... 8x8크기라고 가정하고 homography를 구함
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

    xmin,xmax,ymin,ymax = min([i[0] for i in max_inlier_warped]),max([i[0] for i in max_inlier_warped]),min([i[1] for i in max_inlier_warped]),max([i[1] for i in max_inlier_warped])
    for i in range(len(max_inlier_warped)):
       max_inlier_warped[i]=( max_inlier_warped[i][0]-(xmin),  max_inlier_warped[i][1]-(ymin))#xmin, ymin을 0으로 함
    xmax-=xmin
    ymax-=ymin
    print(xmax,ymax)

    homography = cv2.findHomography(np.array(max_inlier_set,dtype=np.float32),80*np.array(max_inlier_warped,dtype=np.float32)+640)[0]
   
    return homography,xmax,ymax
def get_lines(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,150,170)
    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    
    return lines
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
#rectified 이미지로 보드 위치 구하기
def get_board(image,xmax,ymax):
    greyImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edgeH =  canny_h(greyImage)
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
        print(xmax,xmin)
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
    return xmin,xmax,ymin,ymax
#구한 선과 교점을 그리기
def draw_lines(image):
    lines = get_lines(image)
    lines1,lines2 = cluster_lines(lines)
    lines2 = merge_lines(lines1,lines2)
    lines1 = merge_lines(lines2,lines1)
    
    for i in range(len(lines1)):
            rho, theta = lines1[i][0]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(image, pt1, pt2, (0,0,255) , 1, cv2.LINE_AA)
    
    cv2.line(image, pt1, pt2, (0,0,255) , 1, cv2.LINE_AA)
    for i in range(len(lines2)):
        rho, theta = lines2[i][0]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(image, pt1, pt2, (0,255,0) , 1, cv2.LINE_AA)
    intersections = get_intersections(lines1,lines2)
    for intersection in intersections:
        
        cv2.circle(image,(int(intersection[0]),int(intersection[1])),3,(255,0,255),1)
    return image
def get_homography_from_image(image):
    lines = get_lines(image)
    lines1,lines2 = cluster_lines(lines)
    homography,xmax,ymax = get_homography(lines1,lines2)
    image = cv2.warpPerspective(image,homography,(1920,1920))

    xmax=int(xmax)
    ymax=int(ymax)
    xmin=0
    ymin=0
    xmin,xmax,ymin,ymax = get_board(image,xmax,ymax)
  
    #cv2.line(image,(80*xmin+640,80*ymin+640),(80*xmax+640,80*ymin+640),(255,0,255),2)
   # cv2.line(image,(80*xmin+640,80*ymax+640),(80*xmax+640,80*ymax+640),(255,0,255),2)
   # cv2.line(image,(80*xmax+640,80*ymax+640),(80*xmax+640,80*ymin+640),(255,0,255),2)
   # cv2.line(image,(80*xmin+640,80*ymin+640),(80*xmin+640,80*ymax+640),(255,0,255),2)
    return image,xmax,ymax
    '''

filenames,images,labels = dataset.load_images()
for filename,image,label in zip(filenames,images,labels):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,150,170)
    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    thetas = []
    dist = []
    for i in range(len(lines)):
        for rho,theta in lines[i]:
            thetas.append(theta)
    for i in range(len(lines)):
        dist.append([])
        for j in range(len(lines)):
            dist[i].append(angle_distance(thetas[i],thetas[j]))
    print(thetas)
    cluster = AgglomerativeClustering(n_clusters=2,affinity='precomputed',linkage='average')
    cluster.fit(dist)
    print(cluster.labels_)
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(image, pt1, pt2, (0,0,255) if cluster.labels_[i] else (255,255,0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join('hough',filename[-4:]+'.png'),image)'''
