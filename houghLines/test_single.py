from board import detect_board
from utils import draw_corners
import cv2
import os
import sys

# Program to test a single image

def main(argv):
    result_dir = 'test_single_results'
    os.makedirs(result_dir, exist_ok=True)

    # load file
    if len(argv) < 1:
        print("No file name passed")
        return

    img_path = argv[0]
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)

    debug = True
    warped_image, H, corners = detect_board(img, debug=debug)

    warped_image = draw_corners(warped_image, corners)

    cv2.imwrite(os.path.join(result_dir, img_name + '.png'), warped_image)

if __name__ == "__main__":
    main(sys.argv[1:])