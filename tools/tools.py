import json
import glob
import cv2
import numpy as np


def load_config(config_path):
    f = open(config_path, encoding='utf-8')
    config = json.load(f)
    f.close()
    return config


def get_paths(root, format):
    paths = glob.glob(r"{}/*{}".format(root, format))
    paths = sorted(paths)

    return [p.replace("\\", "/") for p in paths]


def remap_image(intrinsics, img):
    f_x, s, c_x, f_y, c_y = intrinsics[0][0], intrinsics[0][1], intrinsics[0][2], intrinsics[1][1], intrinsics[1][2]

    height, width = img.shape
    x_list = np.arange(0, width)
    y_list = np.arange(0, height)
    x, y = np.meshgrid(x_list, y_list)

    x_src = f_x / f_y * x + s / f_y * y
    y_src = y

    x_src = x_src.astype(np.float32)
    y_src = y_src.astype(np.float32)

    new_img = cv2.remap(img, x_src, y_src, cv2.INTER_LINEAR)
    new_intrinsics = np.array([[f_y, 0, (c_x * f_y - s * c_y) / f_x],
                               [0, f_y, c_y],
                              [0, 0, 1]])

    return new_intrinsics, new_img
