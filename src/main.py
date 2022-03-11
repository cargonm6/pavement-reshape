import os

import numpy as np
import cv2


def image_read(p_file):
    p_image = cv2.imread(p_file)
    return p_image


def image_distortion(p_image):
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

    p_height, p_width = p_image.shape[0:2]
    cam[0, 2] = p_width / 2.0  # define center x
    cam[1, 2] = p_height / 2.0  # define center y
    cam[0, 0] = 3.1  # define focal length x
    cam[1, 1] = 4.  # define focal length y

    # here the undistortion will be computed
    return cv2.undistort(p_image, cam, dist_coef)


def image_slice(p_image):
    return p_image[:p_image.shape[0] // 4, :]


def image_save(p_path, p_name, p_image):
    cv2.imwrite(p_path + p_name, p_image)


def main(project_root):
    path_origin = project_root + "/res/origin/"
    path_joined = project_root + "/res/joined/"
    path_concat = project_root + "/res/concat/"
    f_list = []

    for root, dirs, files in os.walk(path_origin):
        for f in files:
            f_list.append((os.path.join(root, f)).replace("\\", "/")) if not f.startswith(".") else 0

    list_sliced = []
    count = 0

    for n in range(0, len(f_list)):
        count += 1

        image_origin = image_read(f_list[n])
        image_distor = image_distortion(image_origin)

        list_sliced.append(image_slice(image_distor))

        image_save(path_joined, os.path.basename(f_list[n]), np.concatenate((image_origin, image_distor), axis=1))

        if count % 3 == 0:
            image_save(path_concat, os.path.basename(f_list[n]), np.concatenate(list_sliced[-3:], axis=0))
