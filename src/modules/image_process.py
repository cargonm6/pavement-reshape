import os

import cv2
import numpy as np


def image_read(p_file):
    p_image = cv2.imread(p_file)
    return p_image


def image_distortion(p_image, p_calibration=None):
    if p_calibration is not None:
        ret, mtx, dist, rvecs, tvecs = p_calibration
        p_height, p_width = p_image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (p_width, p_height), 1, (p_width, p_height))

        dst = cv2.undistort(p_image, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        return dst
    else:

        dist_coef = np.zeros((4, 1), np.float64)

        # negative to remove barrel distortion
        k1 = -1.0e-5
        k2 = 0.0
        p1 = 0.0
        p2 = 0.0

        dist_coef[0, 0] = k1
        dist_coef[1, 0] = k2
        dist_coef[2, 0] = p1
        dist_coef[3, 0] = p2

        # assume unit matrix for camera
        cam = np.eye(3, dtype=np.float32)

        p_height, p_width = p_image.shape[:2]
        cam[0, 2] = p_width / 2.0  # define center x
        cam[1, 2] = p_height / 2.0  # define center y
        cam[0, 0] = 3.1  # define focal length x
        cam[1, 1] = 4.  # define focal length y

    # here the undistortion will be computed
    return cv2.undistort(p_image, cam, dist_coef)


def image_slice(p_image):
    # Top, bottom, left, right
    # margin = [0, 0, p_image.shape[1] - p_image.shape[1] // 8, 0]
    margin = [0, p_image.shape[0] - p_image.shape[0] // 4, 0, 0]

    return p_image[(0 + margin[0]):(p_image.shape[0] - margin[1]), (0 + margin[2]):(p_image.shape[1] - margin[3])]


def image_save(p_path, p_name, p_image):
    cv2.imwrite(p_path + p_name, p_image)


def main(project_root, calibration):
    path_origin = project_root + "/res/0_source/"
    path_distor = project_root + "/res/1_distor/"
    path_joined = project_root + "/res/2_joined/"
    path_sliced = project_root + "/res/3_sliced/"
    path_concat = project_root + "/res/4_concat/"
    f_list = []

    for root, dirs, files in os.walk(path_origin):
        for f in files:
            f_list.append((os.path.join(root, f)).replace("\\", "/")) if not f.startswith(".") else 0

    list_sliced = []
    count = 0

    for n in range(0, len(f_list)):
        count += 1

        image_origin = image_read(f_list[n])
        image_distor = image_distortion(image_origin, calibration)
        image_sliced = image_slice(image_distor)

        image_save(path_distor, os.path.basename(f_list[n]), image_distor)
        image_save(path_joined, os.path.basename(f_list[n]), np.concatenate(
            (image_origin, cv2.resize(image_distor, dsize=(image_origin.shape[1], image_origin.shape[0]),
                                      interpolation=cv2.INTER_CUBIC)), axis=1))
        image_save(path_sliced, os.path.basename(f_list[n]), image_sliced)

        list_sliced.append(image_sliced)

        if count % 3 == 0:
            p_size = 3
            p_axis = 0  # 0/1: horizontal/vertical
            image_save(path_concat, os.path.basename(f_list[n]), np.concatenate(list_sliced[-p_size:], axis=p_axis))
