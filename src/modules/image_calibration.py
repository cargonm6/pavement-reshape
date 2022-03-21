import os

import cv2 as cv
import numpy as np


def main(project_root):
    """
    Main function of image calibration
    :param project_root: root path of the project
    :return: calibration parameters
    """
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    board_corners = (9, 6)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((np.prod(board_corners), 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_corners[0], 0:board_corners[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    obj_points = []  # 3d point in real world space
    img_points = []  # 2d points in image plane.
    gray = None

    f_list = []

    for root, dirs, files in os.walk(project_root + "/res/calibration/"):
        for f in files:
            f_list.append((os.path.join(root, f)).replace("\\", "/")) if not f.startswith(".") else 0

    if len(f_list) == 0:
        return None

    for image in f_list:
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, board_corners, None)

        # If found, add object points, image points (after refining them)
        if ret is True:
            obj_points.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)

            # Draw and display the corners
            cv.drawChessboardCorners(img, board_corners, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(250)

    cv.destroyAllWindows()

    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs
