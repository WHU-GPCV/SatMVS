"""
data io
"""

from __future__ import print_function
import re
import os
import numpy as np
import sys
from PIL import Image
from osgeo import gdal

import matplotlib.pyplot as plt


# PFM file
def load_pfm(fname):
    file = open(fname, 'rb')
    header = str(file.readline().decode('latin-1')).rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('latin-1'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    scale = float((file.readline().decode('latin-1')).rstrip())
    if scale < 0:  # little-endian
        data_type = '<f'
    else:
        data_type = '>f'  # big-endian

    data = np.fromfile(file, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = np.flip(data, 0)

    return data


def save_pfm(file, image, scale=1):
    file = open(file, mode='wb')

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(bytes('PF\n' if color else 'Pf\n', encoding='utf8'))
    file.write(bytes('%d %d\n' % (image.shape[1], image.shape[0]), encoding='utf8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(bytes('%f\n' % scale, encoding='utf8'))

    image_string = image.tostring()
    file.write(image_string)

    file.close()


# rpc file
def load_rpc_as_array(filepath):
    if os.path.exists(filepath) is False:
        raise Exception("RPC not found! Can not find " + filepath + " in the file system!")

    with open(filepath, 'r') as f:
        all_the_text = f.read().splitlines()

    data = [text.split(' ')[1] for text in all_the_text]
    # print(data)
    data = np.array(data, dtype=np.float64)

    h_min = data[4] - data[9]
    h_max = data[4] + data[9]

    return data, h_max, h_min


def to_tensor(data):
    assert data.shape[0] == 20 and len(data.shape) == 1
    coeff_tensor = np.array(
        [[[data[0], data[1] / 3.0, data[2] / 3.0, data[3] / 3.0],
          [data[1] / 3.0, data[7] / 3.0, data[4] / 6.0, data[5] / 6.0],
          [data[2] / 3.0, data[4] / 6.0, data[8] / 3.0, data[6] / 6.0],
          [data[3] / 3.0, data[5] / 6.0, data[6] / 6.0, data[9] / 3.0]],

         [[data[1] / 3.0, data[7] / 3.0, data[4] / 6.0, data[5] / 6.0],
          [data[7] / 3.0, data[11], data[14] / 3.0, data[17] / 3.0],
          [data[4] / 6.0, data[14] / 3.0, data[12] / 3.0, data[10] / 6.0],
          [data[5] / 6.0, data[17] / 3.0, data[10] / 6.0, data[13] / 3.0]],

         [[data[2] / 3.0, data[4] / 6.0, data[8] / 3.0, data[6] / 6.0],
          [data[4] / 6.0, data[14] / 3.0, data[12] / 3.0, data[10] / 6.0],
          [data[8] / 3.0, data[12] / 3.0, data[15], data[18] / 3.0],
          [data[6] / 6.0, data[10] / 6.0, data[18] / 3.0, data[16] / 3.0]],

         [[data[3] / 3.0, data[5] / 6.0, data[6] / 6.0, data[9] / 3.0],
          [data[5] / 6.0, data[17] / 3.0, data[10] / 6.0, data[13] / 3.0],
          [data[6] / 6.0, data[10] / 6.0, data[18] / 3.0, data[16] / 3.0],
          [data[9] / 3.0, data[13] / 3.0, data[16] / 3.0, data[19]]]
         ]
    )

    return coeff_tensor


def load_rpc_as_qc_tensor(filepath: str):
    if os.path.exists(filepath) is False:
        raise Exception("RPC not found! Can not find " + filepath + " in the file system!")

    with open(filepath, 'r') as f:
        all_the_text = f.read().splitlines()

    data = [text.split(' ')[1] for text in all_the_text]
    # print(data)
    data = np.array(data, dtype=np.float64)

    rpc = dict()

    rpc["line_off"], rpc["samp_off"], rpc[
        "lat_off"], rpc["lon_off"], rpc["height_off"
    ], rpc["line_scale"], rpc["samp_scale"], rpc[
        "lat_scale"], rpc["lon_scale"], rpc["height_scale"
    ] = data[0:10]
    rpc["line_num_tensor"] = to_tensor(data[10: 30])
    rpc["line_den_tensor"] = to_tensor(data[30: 50])
    rpc["samp_num_tensor"] = to_tensor(data[50: 70])
    rpc["samp_den_tensor"] = to_tensor(data[70: 90])
    rpc["lat_num_tensor"] = to_tensor(data[90: 110])
    rpc["lat_den_tensor"] = to_tensor(data[110: 130])
    rpc["lon_num_tensor"] = to_tensor(data[130: 150])
    rpc["lon_den_tensor"] = to_tensor(data[150: 170])

    return rpc


# image
def read_img(filename):
    org = Image.open(filename)
    imgs = org.split()

    if len(imgs) == 3:
        img = org
    elif len(imgs) == 1:
        g = imgs[0]
        img = Image.merge("RGB", (g, g, g))
    else:
        raise Exception("Images must have 3 channels or 1.")

    return img


def gdal_get_size(path):
    dataset = gdal.Open(path)
    if dataset is None:
        raise Exception("GDAL RasterIO Error: Opening" + path + " failed!")

    width = dataset.RasterXSize
    height = dataset.RasterYSize

    del dataset
    return width, height


def gdal_read_img_tone(path, x_lu=None, y_lu=None, x_size=None, y_size=None):
    dataset = gdal.Open(path)
    if dataset is None:
        raise Exception("GDAL RasterIO Error: Opening" + path + " failed!")

    if x_lu is None:
        x_lu = 0
    if y_lu is None:
        y_lu = 0
    if x_size is None:
        x_size = dataset.RasterXSize - x_lu
    if y_size is None:
        y_size = dataset.RasterYSize - y_lu

    data = dataset.ReadAsArray(x_lu, y_lu, x_size, y_size)

    if data.ndim > 2:
        data = np.sum(data.astype(float), axis=0) / data.shape[0]
    else:
        data = data

    im = np.power(data, 1.0 / 2.2)  # gamma correction

    # cut off the small values
    below_thres = np.percentile(im.reshape((-1, 1)), 0.5)
    im[im < below_thres] = below_thres
    # cut off the big values
    above_thres = np.percentile(im.reshape((-1, 1)), 99.5)
    im[im > above_thres] = above_thres
    img = 255 * (im - below_thres) / (above_thres - below_thres)

    del dataset

    return img


def gdal_read_img(path, x_lu, y_lu, xsize, ysize):
    dataset = gdal.Open(path)
    if dataset is None:
        raise Exception("GDAL RasterIO Error: Opening" + path + " failed!")

    if x_lu is None:
        x_lu = 0
    if y_lu is None:
        y_lu = 0
    if xsize is None:
        xsize = dataset.RasterXSize - x_lu
    if ysize is None:
        ysize = dataset.RasterYSize - y_lu

    data = dataset.ReadAsArray(x_lu, y_lu, xsize, ysize)

    del dataset

    return data


def gdal_read_img_pipeline(path, x_lu, y_lu, xsize, ysize):
    dataset = gdal.Open(path)
    if dataset is None:
        raise Exception("GDAL RasterIO Error: Opening" + path + " failed!")

    data = dataset.ReadAsArray(x_lu, y_lu, xsize, ysize)

    if len(data.shape) == 2:
        im = np.power(data, 1.0 / 2.2)  # gamma correction

        # cut off the small values
        below_thres = np.percentile(im.reshape((-1, 1)), 0.5)
        im[im < below_thres] = below_thres
        # cut off the big values
        above_thres = np.percentile(im.reshape((-1, 1)), 99.5)
        im[im > above_thres] = above_thres
        img = 255 * (im - below_thres) / (above_thres - below_thres)

        img = img.astype(np.uint8)
        data = np.stack([img, img, img], axis=0)

    del dataset

    return data


def read_tfw(path):
    """
    TFW files are the ASCII files containing information
    for geocoding image data so TIFF images can be used
    directly by GIS and CAD applications.
    """
    file_object = open(path)
    try:
        all_the_text = file_object.read().splitlines()
    finally:
        file_object.close()

    tfw = np.array(all_the_text, dtype=np.float)

    if tfw.shape[0] != 6:
        raise Exception("6 parameters excepted in the tfw file, but got {}.".format(tfw.shape[0]))

    return tfw


def cv_save_image(filepath, img):
    import cv2
    cv2.imwrite(filepath, img)


def gdal_create_dsm_file(out_path, e_ul, n_ul, xuint, yuint, width, height):
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    driver.Create(out_path, width, height, 1, gdal.GDT_Float32)

    text_ = str(xuint) + "\n0\n0\n" + str(-yuint) + "\n" + str(e_ul) + "\n" + str(n_ul)
    tfw_path = out_path.replace(".tif", ".tfw")
    with open(tfw_path, "w") as f:
        f.write(text_)


def gdal_write_to_tif(out_path, xlu, ylu, data):
    dataset = gdal.Open(out_path, gdal.GF_Write)
    if dataset is None:
        raise Exception("GDAL RasterIO Error: Opening" + out_path + " failed!")

    if data is None:
        return

    if len(data.shape) == 3:
        im_bands = data.shape[0]
    else:
        im_bands = 1

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(data, xlu, ylu)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(data[i], xlu, ylu)
    del dataset


def init_dsm(path, invalid):
    width, height = gdal_get_size(path)
    data = gdal_read_img(path, 0, 0, width, height)
    data += invalid

    gdal_write_to_tif(path, 0, 0, data)


def read_camera(path):
    with open(path, "r") as f:
        all_text = f.read().splitlines()

    E = np.array([[float(at) for at in all_text[0].split(" ")],
                 [float(at) for at in all_text[1].split(" ")],
                 [float(at) for at in all_text[2].split(" ")],
                 [float(at) for at in all_text[3].split(" ")]])

    K_text = [float(at) for at in all_text[5].split(" ")]
    K = np.array([[K_text[0], 0, K_text[1]],
                 [0, K_text[0], K_text[2]],
                 [0, 0, 1]])

    d_min, d_max, d_inter = [float(at) for at in all_text[7].split(" ")]

    return K, E, d_min, d_max, d_inter


def save_camera(filepath, k, r, t, d_min, d_max, d_interval, img_index, width, height):
    """
    0 E00 E01 E02 E03
    1 E10 E11 E12 E13
    2 E20 E21 E22 E23
    3 E30 E31 E32 E33
    4
    5 f(pixel)  x0(pixel)  y0(pixel)
    6
    7 DEPTH_MIN   DEPTH_MAX   DEPTH_INTERVAL
    8 IMAGE_INDEX 0 0 0 0 WIDTH HEIGHT
    """
    text_ = ""

    E = np.concatenate((r, t), axis=-1)

    text_ += str(E[0][0]) + " " + str(E[0][1]) + " " + str(E[0][2]) + " " + str(E[0][3]) + "\n"
    text_ += str(E[1][0]) + " " + str(E[1][1]) + " " + str(E[1][2]) + " " + str(E[1][3]) + "\n"
    text_ += str(E[2][0]) + " " + str(E[2][1]) + " " + str(E[2][2]) + " " + str(E[2][3]) + "\n"
    text_ += "0 0 0 1\n\n"

    text_ += str(k[0, 0]) + " " + str(k[0, 2]) + " " + str(k[1, 2]) + "\n\n"

    text_ += str(d_min) + " " + str(d_max) + " " + str(d_interval) + "\n"
    text_ += str(img_index) + " 0 0 0 0 " + str(width) + " " + str(height) + "\n"

    with open(filepath, "w") as f:
        f.write(text_)


def read_vir_camera_in_nn(path):

    K, E, d_min, d_max, d_inter = read_camera(path)

    # read camera txt file
    cam = np.zeros((2, 4, 4), dtype=np.float64)
    cam[0] = E.astype(np.float64)
    cam[1, 0:3, 0:3] = K.astype(np.float64)

    # depth range
    cam[1][3][0] = np.float64(d_min)  # start
    cam[1][3][1] = np.float64(d_inter)  # interval
    cam[1][3][3] = np.float64(d_max)  # end

    return cam


def save_errors(filepath, proj_err, inv_proj_err):
    text_ = 'mean_proj_err (pixels), median_proj_err (pixels), max_proj_err (pixels), mean_inv_proj_err (meters), median_inv_proj_err (meters), max_inv_proj_err (meters)\n'
    text_ += str(np.mean(proj_err)) + ", " + str(np.median(proj_err)) + ", " + str(np.max(proj_err)) + ", "
    text_ += str(np.mean(inv_proj_err)) + ", " + str(np.median(inv_proj_err)) + ", " + str(np.max(inv_proj_err))

    with open(filepath, "w") as f:
        f.write(text_)


if __name__ == "__main__":
    pass
