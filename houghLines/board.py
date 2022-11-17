import os
import cv2
import dataset
import numpy as np
import math
from sklearn.cluster import AgglomerativeClustering
def angle_distance(a,b):
    return min(abs(a-b),math.pi-abs(a-b))
filenames,images,labels = dataset.load_images()
for filename,image,label in zip(filenames,images,labels):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,150,170)
    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    thetas = []
    dist = []
    for i in range(len(lines)):
        for rho,theta in lines[i]:
            thetas.append(theta) # Hough transform으로 구한 각도 List
    for i in range(len(lines)):
        dist.append([])
        for j in range(len(lines)):
            dist[i].append(angle_distance(thetas[i],thetas[j])) #각도 List의 모든 pairwise distance를 구함
    print(thetas)
    cluster = AgglomerativeClustering(n_clusters=2,affinity='precomputed',linkage='average') 
    cluster.fit(dist)
    print(cluster.labels_) #cluster에 따라 0/1 이 저장됨
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(image, pt1, pt2, (0,0,255) if cluster.labels_[i] else (255,255,0), 1, cv2.LINE_AA) #cluster label의 값 0/1에 따라 색을 다르게 함
    cv2.imwrite(os.path.join('hough',filename[-4:]+'.png'),image)
