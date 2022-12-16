# piece & board

import argparse
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from houghLines.board import detect_board
from yolov7.predict import do_detect

# visual representation

import cv2
import chess
import chess.svg

tensor = do_detect()[0]

image_path = './test_image/28.png'
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
