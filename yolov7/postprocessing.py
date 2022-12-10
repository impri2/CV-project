import argparse
import os
import numpy
import torch
import cv2

from houghLines.board import detect_board
from yolov7.predict import detect

def detect_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    #check_requirements(exclude=('pycocotools', 'thop'))

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_image")
    weights = r"runs\train\yolov7_multi_res10\weights\best.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        a = detect(weights, source, device)
    return a


def overlap():
    print("todo") # todo


tensor = detect_predict()[0]

image = cv2.imread('./test_image/IMG_0159_JPG.rf.f0d34122f8817d538e396b04f2b70d33.jpg')
warped_image, homography, corners = detect_board(image)

entire_x = corners[2] - corners[0]
entire_y = corners[3] - corners[1]

warped_tensor = numpy.hstack([numpy.zeros(tensor.shape), numpy.zeros((len(tensor), 2))]) # 8 rows

count_labels = numpy.zeros(13)
limit_labels = numpy.array([0, 2, 1, 2, 8, 2, 2, 2, 1, 2, 8, 2, 2])
chess_labels = ["bishop", "black-bishop", "black-king", "black-knight", "black-pawn", "black-queen", "black-rook", \
                "white-bishop", "white-king", "white-knight", "white-pawn", "white-queen", "white-rook"]

for i in range(len(tensor)):
    if count_labels[int(tensor[i][5])] > limit_labels[int(tensor[i][5])]:
        continue

    x1, y1, z1 = homography @ numpy.array((tensor[i][0], tensor[i][1], 1))
    x2, y2, z2 = homography @ numpy.array((tensor[i][2], tensor[i][3], 1))

    x = (x1 + x2) / 2
    y = (y1 + 2*y2) / 3 # 2:1 내분

    frac_x = (x - corners[0]) / entire_x
    frac_y = (y - corners[1]) / entire_y
    cell_x = round(8 * frac_x)
    cell_y = round(8 * frac_y)

    warped_tensor[i] = numpy.array([x1, y1, x2, y2, tensor[i][4], tensor[i][5], cell_x, cell_y])

    count_labels[int(tensor[i][5])] = (count_labels[int(tensor[i][5])] + 1)

chessboard = numpy.zeros((8, 8))
for i in range(len(warped_tensor)):
    x = warped_tensor[i][6]
    y = warped_tensor[i][7]


