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

canny_hysteresis_threshold1 = 5
canny_hysteresis_threshold2 = 10

# detect vertical lines
def canny_v(image, debug=False):
    edge = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # x sobel
    edge /= (1 + 2 + 0 + 2 + 1) * (1 + 4 + 6 + 4 + 1)  # normalize by sobel filter size
    edge = np.abs(edge)

    # Non max suppression
    edge_nms = np.copy(edge)
    for i in range(1, edge.shape[0]-1):
        for j in range(1, edge.shape[1]-1):
            if edge[i, j + 1] > edge[i, j] or edge[i, j - 1] > edge[i, j]:
                edge_nms[i, j] = 0

    edge = edge_nms

    edge = filters.apply_hysteresis_threshold(
        edge,
        canny_hysteresis_threshold1,
        canny_hysteresis_threshold2)

    if debug:
        open_wait_cv2_window("canny_vertical", cv2.resize(edge.astype('uint8') * 255, (0,0), fx=0.5, fy=0.5))

    return edge

# detect horizontal lines
def canny_h(image, debug=False):
    edge = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # y sobel
    edge /= (1 + 2 + 0 + 2 + 1) * (1 + 4 + 6 + 4 + 1)  # normalize by sobel filter size
    edge = np.abs(edge)

    # Non max suppression
    edge_nms = np.copy(edge)
    for i in range(1, edge.shape[0] - 1):
        for j in range(1, edge.shape[1] - 1):
            if edge[i + 1, j] > edge[i, j] or edge[i - 1, j] > edge[i, j]:
                edge_nms[i, j] = 0

    edge = edge_nms

    edge = filters.apply_hysteresis_threshold(
        edge,
        canny_hysteresis_threshold1,
        canny_hysteresis_threshold2)

    if debug:
        open_wait_cv2_window("canny_horizontal", cv2.resize(edge.astype('uint8') * 255, (0, 0), fx=0.5, fy=0.5))

    return edge

# open cv2 image then wait until key q is pressed or window is closed
# taken from: https://medium.com/@mh_yip/opencv-detect-whether-a-window-is-closed-or-close-by-press-x-button-ee51616f7088
def open_wait_cv2_window(window_name, image):
    cv2.imshow(window_name, image)

    wait_time = 10000
    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
        keyCode = cv2.waitKey(wait_time)
        if (keyCode & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break

def draw_corners(image, corners):
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    x1, y1, x2, y2 = corners
    cv2.line(image, (x1, y1), (x2, y1), (255, 0, 255), 2)
    cv2.line(image, (x1, y1), (x1, y2), (255, 0, 255), 2)
    cv2.line(image, (x2, y1), (x2, y2), (255, 0, 255), 2)
    cv2.line(image, (x1, y2), (x2, y2), (255, 0, 255), 2)

    return image

def draw_lines(image, lines1, lines2):
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for seq in (lines1, lines2):
        for i in range(len(seq)):
            rho, theta = seq[i][0]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 - 1000 * b), int(y0 + 1000 * a))
            pt2 = (int(x0 + 1000 * b), int(y0 - 1000 * a))
            cv2.line(image, pt1, pt2, (0, 20 * i % 256, (255 - 20 * i) % 256), 1, cv2.LINE_AA)

    return image

# given warped canny edge, isolate vertical ones and horizontal ones
# https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html
def isolate_edge(canny, debug=False):
    hor = np.copy(canny)
    ver = np.copy(canny)

    cell_size = 80
    line_len = 5
    # want to detect lines that are at line_len
    hor_size = line_len
    ver_size = line_len

    hor_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_size, 1))
    ver_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_size))

    # apply morphology operations
    hor = cv2.erode(hor, hor_structure)
    hor = cv2.dilate(hor, hor_structure)

    ver = cv2.erode(ver, ver_structure)
    ver = cv2.dilate(ver, ver_structure)

    if debug:
        open_wait_cv2_window("horizontal edges",
                             cv2.resize(hor, (0, 0), fx=0.3, fy=0.3))
        open_wait_cv2_window("vertical edges",
                             cv2.resize(ver, (0, 0), fx=0.3, fy=0.3))

    return hor, ver