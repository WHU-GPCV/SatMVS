
import numpy as np
import cv2


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    ## step1. project reference pixels to the source view
    height, width = depth_ref.shape
    row, col = np.meshgrid(range(height), range(width), indexing='ij')
    col = col.reshape((1, -1))
    row = row.reshape((1, -1))
    depth_ref = depth_ref.reshape((1, -1))

    tmp = np.vstack((depth_ref * col, depth_ref * row, depth_ref, np.ones((1, width * height))))

    P_ref = np.matmul(intrinsics_ref, extrinsics_ref[:3])
    temp = np.array([[0, 0, 0, 1]])
    P_ref = np.concatenate((P_ref, temp), axis=0)
    inv_p_ref = np.linalg.inv(P_ref)

    P_src = np.matmul(intrinsics_src, extrinsics_src[:3])
    P_src = np.concatenate((P_src, temp), axis=0)
    inv_p_src = np.linalg.inv(P_src)

    xy_src = np.matmul(inv_p_ref, tmp)
    xy_src = np.matmul(P_src, xy_src)

    xy_src = xy_src[:2]/xy_src[2]
    x_src = xy_src[0].reshape(height, width).astype(np.float32)
    y_src = xy_src[1].reshape(height, width).astype(np.float32)

    depth_reprojected = cv2.remap(depth_src, x_src, y_src, cv2.INTER_LINEAR)
    depth_reprojected_vec = depth_reprojected.reshape(1, -1)

    tmp = np.vstack((depth_reprojected_vec * xy_src[0], depth_reprojected_vec * xy_src[1],
                                depth_reprojected_vec, np.ones((1, width * height))))

    xy_reprojected = np.matmul(inv_p_src, tmp)
    xy_reprojected = np.matmul(P_ref, xy_reprojected)
    xy_reprojected = xy_reprojected[:2] / xy_reprojected[2]

    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src,
                                p_thre=1, relative_d_thre=0.01):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1|  < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < p_thre, relative_depth_diff < relative_d_thre)

    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src

