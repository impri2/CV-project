# piece & board

import numpy as np
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from houghLines.board import detect_board
#from yolov7.predict import do_detect
from yolov7 import predict
# visual representation

import cv2
import chess
import chess.svg

def main(opt):
    file_path = opt.image
    file_path = os.path.join(file_path[0])
    print(file_path)
    tensor = predict.do_detect(file_path, opt)[0]

    image_path = file_path
    image = cv2.imread(image_path)
    warped_image, homography, corners = detect_board(image)

    entire_x = corners[2] - corners[0]
    entire_y = corners[3] - corners[1]
    cell_size = entire_x / 8

    warped_tensor = np.hstack([np.zeros(tensor.shape), np.zeros((len(tensor), 2))]) # 8 rows

    count_labels = np.zeros(13)
    limit_labels = np.array([0, 2, 1, 2, 8, 2, 2, 2, 1, 2, 8, 2, 2])
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

    def homo(H, x, y):
        x, y, z = homography @ np.array((x, y, 1))
        return (x/z, y/z)

    for i in range(len(tensor)):
        if count_labels[int(tensor[i][5])] >= limit_labels[int(tensor[i][5])]:
            continue



        x1 = tensor[i][0]
        y1 = tensor[i][1]
        x2 = tensor[i][2]
        y2 = tensor[i][3]
        
        
        distance = np.zeros((8,8))
        piece_lower_left = homo(homography, x1, y2)
        piece_lower_right = homo(homography, x2, y2)
        for x_grid in range(8):
            for y_grid in range(8):
                cell_lower_left = np.array([corners[0] + (cell_size * x_grid), corners[1] + (y_grid + 1) * cell_size])
                cell_lower_right = np.array([cell_lower_left[0] + cell_size, cell_lower_left[1]])
                cell_distance = np.sqrt(np.sum(np.square(cell_lower_left - piece_lower_left))) + np.sqrt(np.sum(np.square(cell_lower_right - piece_lower_right)))
                distance[y_grid, x_grid] = cell_distance
                
        k = np.argmin(distance)
        cell_x = k % 8
        cell_y = k // 8
        #print(distance)
        #print(distance[cell_y, cell_x])

        if distance[cell_y, cell_x] >= cell_size * (2 **0.5)* 2:
            continue
        



        temp = np.array([x1, y1, x2, y2, tensor[i][4], tensor[i][5], cell_x, cell_y])

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

parser = argparse.ArgumentParser()
parser.add_argument("--image", nargs='+', type=str)
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
main(opt)