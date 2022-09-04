import numpy as np
import cv2
# from tools.RPCCore import RPCModelParameter
from tools.rpc_tensor import RPCModelParameter
import time


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, rpc_ref, depth_src, rpc_src):

    rpc_model_ref = RPCModelParameter(rpc_ref)
    rpc_model_src = RPCModelParameter(rpc_src)
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])

    # t1 = time.time()
    lat, lon = rpc_model_ref.RPC_PHOTO2OBJ(x_ref.astype(np.float), y_ref.astype(np.float), depth_ref.reshape([-1]))
    # t2 = time.time()
    # source view x, y
    x_src, y_src = rpc_model_src.RPC_OBJ2PHOTO(lat, lon, depth_ref.reshape([-1]))
    # t3 = time.time()

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = x_src.reshape([height, width])
    y_src = y_src.reshape([height, width])
    sampled_depth_src = cv2.remap(depth_src, x_src.astype(np.float32), y_src.astype(np.float32),
                                  interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-999)

    """import matplotlib.pyplot as plt

    plt.imshow(sampled_depth_src)
    plt.show()"""

    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    lat, lon = rpc_model_src.RPC_PHOTO2OBJ(x_src.astype(np.float).reshape(-1), y_src.astype(np.float).reshape(-1),
                             sampled_depth_src.reshape(-1))
    # reference 3D space
    x_reprojected, y_reprojected = rpc_model_ref.RPC_OBJ2PHOTO(lat, lon, sampled_depth_src.reshape(-1))
    # source view x, y, depth

    return sampled_depth_src, x_reprojected.reshape(height, width), y_reprojected.reshape(height, width), x_src, y_src


def check_geometric_consistency(depth_ref, rpc_ref, depth_src, rpc_src, p_ratio, d_ratio):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))

    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, rpc_ref,
                                                     depth_src, rpc_src)

    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| < 2.5m
    depth_diff = np.abs(depth_reprojected - depth_ref)

    mask = np.logical_and(dist < p_ratio, depth_diff < d_ratio)
    # mask = np.logical_and(dist < p_ratio, dist < p_ratio)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(depths, rpcs, p_ratio, d_ratio, geo_consist_num, prob=None, confidence_ratio=0.0):
    # for each reference view and the corresponding source views

    ref_depth = depths[0]
    ref_rpc = rpcs[0]
    vnum = depths.shape[0]

    # photometric mask of the reference view
    if prob is not None:
        ref_prob = prob
        photo_mask = ref_prob > confidence_ratio
    else:
        photo_mask = np.ones_like(ref_depth, np.bool)

    all_srcview_depth_ests = []
    all_srcview_x = []
    all_srcview_y = []
    all_srcview_geomask = []

    # compute the geometric mask
    geo_mask_sum = 0

    for v in range(1, vnum):
        src_depth = depths[v]
        src_rpc = rpcs[v]

        geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth, ref_rpc, src_depth,
                                                                                    src_rpc, p_ratio, d_ratio)
        geo_mask_sum += geo_mask.astype(np.int32)
        all_srcview_depth_ests.append(depth_reprojected)
        all_srcview_x.append(x2d_src)
        all_srcview_y.append(y2d_src)
        all_srcview_geomask.append(geo_mask)


    depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth) / (geo_mask_sum + 1)
    # at least N source views matched

    geo_mask = geo_mask_sum >= geo_consist_num
    final_mask = np.logical_and(photo_mask, geo_mask)

    return final_mask, depth_est_averaged
