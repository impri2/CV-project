from board import detect_board
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

    warped_image, H, corners = detect_board(img, debug=True)

    x1, y1, x2, y2 = corners
    cv2.line(warped_image, (x1, y1), (x2, y1), (255, 0, 255), 2)
    cv2.line(warped_image, (x1, y1), (x1, y2), (255, 0, 255), 2)
    cv2.line(warped_image, (x2, y1), (x2, y2), (255, 0, 255), 2)
    cv2.line(warped_image, (x1, y2), (x2, y2), (255, 0, 255), 2)

    cv2.imwrite(os.path.join(result_dir, img_name + '.png'), warped_image)

if __name__ == "__main__":
    main(sys.argv[1:])