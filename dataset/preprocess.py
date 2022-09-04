"""
data preprocesses.
"""

import numpy as np
import cv2
import math
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import random


def scale_rpc(rpc, scale=1):
    new_rpc = np.copy(rpc)
    # sample
    new_rpc[0] = rpc[0] * scale
    new_rpc[5] = rpc[5] * scale
    # line
    new_rpc[1] = rpc[1] * scale
    new_rpc[6] = rpc[6] * scale

    return new_rpc


def scale_camera(cam, scale=1):
    """ resize input in order to produce sampled depth map """
    new_cam = np.copy(cam)
    # focal:
    new_cam[1][0][0] = cam[1][0][0] * scale
    new_cam[1][1][1] = cam[1][1][1] * scale
    # principle point:
    new_cam[1][0][2] = cam[1][0][2] * scale
    new_cam[1][1][2] = cam[1][1][2] * scale

    return new_cam


def scale_mvs_camera(cams, view_num, scale=1):
    """ resize input in order to produce sampled depth map """
    for view in range(view_num):
        cams[view] = scale_camera(cams[view], scale=scale)
    return cams


def scale_image(image, scale=1, interpolation='linear'):
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    """ resize image using cv2 """
    # h, w = image.shape[0:2]
    # new_w = int(w * scale)
    # new_h = int(h * scale)
    if interpolation == 'linear':
        # return image.resize((new_h, new_w), Image.BILINEAR)
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if interpolation == 'biculic':
        # return image.resize((new_h, new_w), Image.BICUBIC)
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


# rpc
def scale_input_rpc(image, rpc, depth_image=None, scale=1):
    """ resize input to fit into the memory """
    image = scale_image(image, scale=scale)
    rpc = scale_rpc(rpc, scale=scale)
    if depth_image is None:
        return image, rpc, None
    else:
        depth_image = scale_image(depth_image, scale=scale, interpolation='linear')
        return image, rpc, depth_image


def crop_input_rpc(image, rpc, depth_image=None, max_h=384, max_w=768, resize_scale=1, base_image_size=32):
    """ resize images and cameras to fit the network (can be divided by base image size) """
    # crop images and cameras
    max_h = int(max_h * resize_scale)
    max_w = int(max_w * resize_scale)
    h, w = image.shape[0:2]
    new_h = h
    new_w = w
    if new_h > max_h:
        new_h = max_h
    else:
        new_h = int(math.ceil(h / base_image_size) * base_image_size)
    if new_w > max_w:
        new_w = max_w
    else:
        new_w = int(math.ceil(w / base_image_size) * base_image_size)
    start_h = int(math.ceil((h - new_h) / 2))
    start_w = int(math.ceil((w - new_w) / 2))
    finish_h = start_h + new_h
    finish_w = start_w + new_w
    image = image[start_h:finish_h, start_w:finish_w]
    # rpc 像方归一化参数需要更改
    rpc[0] -= start_w
    rpc[1] -= start_h

    # crop depth image
    if depth_image is not None:
        depth_image = depth_image[start_h:finish_h, start_w:finish_w]
        return image, rpc, depth_image
    else:
        return image, rpc, None


# camera
def scale_input_cam(image, cam, depth_image=None, scale=1):
    """ resize input to fit into the memory """
    image = scale_image(image, scale=scale)
    cam = scale_camera(cam, scale=scale)
    if depth_image is None:
        return image, cam, None
    else:
        depth_image = scale_image(depth_image, scale=scale, interpolation='linear')
        return image, cam, depth_image


def crop_input_cam(image, cam, depth_image=None, max_h=384, max_w=768, resize_scale=1, base_image_size=32):
    """ resize images and cameras to fit the network (can be divided by base image size) """
    # crop images and cameras
    max_h = int(max_h * resize_scale)
    max_w = int(max_w * resize_scale)
    h, w = image.shape[0:2]
    new_h = h
    new_w = w
    if new_h > max_h:
        new_h = max_h
    else:
        new_h = int(math.ceil(h / base_image_size) * base_image_size)
    if new_w > max_w:
        new_w = max_w
    else:
        new_w = int(math.ceil(w / base_image_size) * base_image_size)
    start_h = int(math.ceil((h - new_h) / 2))
    start_w = int(math.ceil((w - new_w) / 2))
    finish_h = start_h + new_h
    finish_w = start_w + new_w
    image = image[start_h:finish_h, start_w:finish_w]
    # rpc 像方归一化参数需要更改
    cam[1][0][2] = cam[1][0][2] - start_w
    cam[1][1][2] = cam[1][1][2] - start_h

    # crop depth image
    if depth_image is not None:
        depth_image = depth_image[start_h:finish_h, start_w:finish_w]
        return image, cam, depth_image
    else:
        return image, cam, None


def center_image(img):
    # scale 0~255 to 0~1
    # np_img = np.array(img, dtype=np.float32) / 255.
    # return np_img
    # normalize image input
    img_array = np.array(img)
    img = img_array.astype(np.float32)
    # img = img.astype(np.float32)
    var = np.var(img, axis=(0, 1), keepdims=True)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return (img - mean) / (np.sqrt(var) + 0.00000001)


def image_augment(image):
    image = random_color(image)
    # image = randomGaussian(image, mean=0.2, sigma=0.3)

    return image


def random_color(image):
    random_factor = np.random.randint(1, 301) / 100.
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # Image Color
    random_factor = np.random.randint(10, 201) / 100.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # Image Brightness
    random_factor = np.random.randint(10, 201) / 100.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # Image Contrast
    random_factor = np.random.randint(0, 301) / 100.
    sharpness_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # Image Sharpness

    return sharpness_image


def random_gaussian(image, mean=0.02, sigma=0.03):

    def gaussian_noisy(im, mean=0.02, sigma=0.03):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    img.flags.writeable = True
    width, height = img.shape[:2]
    img_r = gaussian_noisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussian_noisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussian_noisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])

    return Image.fromarray(np.uint8(img))


