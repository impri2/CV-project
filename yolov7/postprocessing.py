# piece & board

import argparse
import numpy
import torch

from houghLines.board import detect_board
from yolov7.predict import detect

# visual representation

import os
import cv2
import chess
import chess.svg

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


tensor = detect_predict()[0]

image_path = './test_image/IMG_0159_JPG.rf.f0d34122f8817d538e396b04f2b70d33.jpg'
image = cv2.imread(image_path)
warped_image, homography, corners = detect_board(image)

entire_x = corners[2] - corners[0]
entire_y = corners[3] - corners[1]

warped_tensor = numpy.hstack([numpy.zeros(tensor.shape), numpy.zeros((len(tensor), 2))]) # 8 rows

count_labels = numpy.zeros(13)
limit_labels = numpy.array([0, 2, 1, 2, 8, 2, 2, 2, 1, 2, 8, 2, 2])
label_to_FENNotation = ['x', 'b', 'k', 'n', 'p', 'q', 'r', 'B', 'K', 'N', 'P', 'Q', 'R']

# chess_labels = ["bishop", "black-bishop", "black-king", "black-knight", "black-pawn", "black-queen", "black-rook", \
#                 "white-bishop", "white-king", "white-knight", "white-pawn", "white-queen", "white-rook"]

# FENNotation = {
#     "white-king": "K",
#     "black-king": "k",
#     "white-queen": "Q",
#     "black-queen": "q",
#     "white-rook": "R",
#     "black-rook": "r",
#     "white-bishop": "B",
#     "black-bishop": "b",
#     "white-knight": "N",
#     "black-knight": "n",
#     "white-pawn": "P",
#     "black-pawn": "p"
# }

for i in range(len(tensor)):
    if count_labels[int(tensor[i][5])] >= limit_labels[int(tensor[i][5])]:
        continue



    x1 = tensor[i][0]
    y1 = tensor[i][1]
    x2 = tensor[i][2]
    y2 = tensor[i][3]

    x = (x1 + x2) / 2
    y = (y1 + y2) / 2

    x, y, z = homography @ numpy.array((x, y, 1))
    x = x / z
    y = y / z

    frac_x = (x - corners[0]) / entire_x
    frac_y = (y - corners[1]) / entire_y
    cell_x = round(8 * frac_x - 0.5)
    cell_y = round(8 * frac_y - 0.5)

    if cell_x < 0 or cell_y < 0 or cell_x >= 8 or cell_y >= 8:
        continue

    temp = numpy.array([x1, y1, x2, y2, tensor[i][4], tensor[i][5], cell_x, cell_y])

    this_piece_already_fixed = False
    for j in range(i):
        if warped_tensor[j][-1] == temp[-1] and warped_tensor[j][-2] == temp[-2]:
            this_piece_already_fixed = True
            break
    if this_piece_already_fixed:
        continue

    warped_tensor[i] = temp

    count_labels[int(tensor[i][5])] = (count_labels[int(tensor[i][5])] + 1)

chessBoard = chess.Board(fen=None)

for i in range(len(warped_tensor)):
    cell_x = int(warped_tensor[i][-2])
    cell_y = int(warped_tensor[i][-1])
    label = int(warped_tensor[i][-3])

    if label == 0:
        continue

    chessBoard.set_piece_at(square=chess.square(cell_x, 7 - cell_y),
                            piece=chess.Piece.from_symbol(label_to_FENNotation[label]))

chessBoardImage = chess.svg.board(board=chessBoard)
filename = image_path.split("/")[-1].split(".jpg")[0].split(".png")[0].split(".jpeg")[0]
with open(os.path.join(('chessboard_results/' + filename + '.svg')), 'w') as f:
    f.write(chessBoardImage)
