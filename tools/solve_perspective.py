#  ===============================================================================================================
#  Copyright (c) 2019, Cornell University. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that
#  the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright otice, this list of conditions and
#        the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#        the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#      * Neither the name of Cornell University nor the names of its contributors may be used to endorse or
#        promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
#  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
#  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
#  OF SUCH DAMAGE.
#
#  Author: Kai Zhang (kz298@cornell.edu)
#
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes.
#  ===============================================================================================================

import numpy as np
from scipy import linalg


def factorize(matrix):
    # QR factorize the submatrix
    r, q = linalg.rq(matrix[:, :3])
    # compute the translation
    t = linalg.lstsq(r, matrix[:, 3:4])[0]

    # fix the intrinsic and rotation matrix
    # intrinsic matrix's diagonal entries must be all positive
    # rotation matrix's determinant must be 1; otherwise there's an reflection component
    #logging.info('before fixing, diag of r: {}, {}, {}'.format(r[0, 0], r[1, 1], r[2, 2]))

    neg_sign_cnt = int(r[0, 0] < 0) + int(r[1, 1] < 0) + int(r[2, 2] < 0)
    if neg_sign_cnt == 1 or neg_sign_cnt == 3:
        r = -r

    new_neg_sign_cnt = int(r[0, 0] < 0) + int(r[1, 1] < 0) + int(r[2, 2] < 0)
    assert (new_neg_sign_cnt == 0 or new_neg_sign_cnt == 2)

    fix = np.diag((1, 1, 1))
    if r[0, 0] < 0 and r[1, 1] < 0:
        fix = np.diag((-1, -1, 1))
    elif r[0, 0] < 0 and r[2, 2] < 0:
        fix = np.diag((-1, 1, -1))
    elif r[1, 1] < 0 and r[2, 2] < 0:
        fix = np.diag((1, -1, -1))
    r = np.dot(r, fix)
    q = np.dot(fix, q)
    t = np.dot(fix, t)

    assert (linalg.det(q) > 0)
    #logging.info('after fixing, diag of r: {}, {}, {}'.format(r[0, 0], r[1, 1], r[2, 2]))

    # check correctness
    # ratio = np.dot(r, np.hstack((q, t))) / matrix
    # assert (np.all(ratio > 0) or np.all(ratio < 0))
    # tmp = np.max(np.abs(np.abs(ratio) - np.ones((3, 4))))
    # logging.info('factorization, max relative error: {}'.format(tmp))
    # assert (np.max(tmp) < 1e-9)

    # normalize the r matrix
    r /= r[2, 2]

    return r, q, t


# colmap convention for pixel indices: (col, row)
def solve_perspective(xx, yy, zz, col, row, keep_mask=None):
    diff_size = np.array([yy.size - xx.size, zz.size - xx.size, col.size - xx.size, row.size - xx.size])
    assert (np.all(diff_size == 0))

    if keep_mask is not None:
        # logging.info('discarding {} % outliers'.format((1. - np.sum(keep_mask) / keep_mask.size) * 100.))
        xx = xx[keep_mask].reshape((-1, 1))
        yy = yy[keep_mask].reshape((-1, 1))
        zz = zz[keep_mask].reshape((-1, 1))
        row = row[keep_mask].reshape((-1, 1))
        col = col[keep_mask].reshape((-1, 1))

    # logging.info('solving perspective, xx: [{}, {}], yy: [{}, {}], zz: [{}, {}]'.format(np.min(xx), np.max(xx),
    #                                                                                     np.min(yy), np.max(yy),
    #                                                                                     np.min(zz), np.max(zz)))
    #
    # scatter3d(xx, yy, zz, '/data2/temp.jpg')

    point_cnt = xx.size
    all_ones = np.ones((point_cnt, 1))
    all_zeros = np.zeros((point_cnt, 4))
    # construct the least square problem
    A1 = np.hstack((xx, yy, zz, all_ones,
                    all_zeros,
                    -col * xx, -col * yy, -col * zz, -col * all_ones))
    A2 = np.hstack((all_zeros,
                    xx, yy, zz, all_ones,
                    -row * xx, -row * yy, -row * zz, -row * all_ones))

    A = np.vstack((A1, A2))
    u, s, vh = linalg.svd(A, full_matrices=False)

    # logging.info('smallest singular value: {}'.format(s[11]))

    P = np.real(vh[11, :]).reshape((3, 4))

    singular_values = ''
    for i in range(11, -1, -1):
        singular_values += ' {}'.format(np.real(s[i]))
    # logging.info('singular values: {}'.format(singular_values))

    # factorize into standard form
    r, q, t = factorize(P)

    return r, q, t


def check_perspective_error(xx, yy, zz, col, row, r, q, t, keep_mask=None):
    if keep_mask is not None:
        # logging.info('discarding {} % outliers'.format((1. - np.sum(keep_mask) / keep_mask.size) * 100.))
        xx = xx[keep_mask].reshape((-1, 1))
        yy = yy[keep_mask].reshape((-1, 1))
        zz = zz[keep_mask].reshape((-1, 1))
        row = row[keep_mask].reshape((-1, 1))
        col = col[keep_mask].reshape((-1, 1))

    point_cnt = xx.size
    all_ones = np.ones((point_cnt, 1))

    # # check the order of the quantities
    # logging.info('\n')
    # logging.info('fx: {}, fy: {}, cx: {} cy: {}, skew: {}'.format(r[0, 0], r[1, 1], r[0, 2], r[1, 2], r[0, 1]))

    translation = np.tile(t.T, (point_cnt, 1))
    result = np.dot(np.hstack((xx, yy, zz)), q.T) + translation
    cam_xx = result[:, 0:1]
    cam_yy = result[:, 1:2]
    cam_zz = result[:, 2:3]

    # logging.info("cam_xx: {}, {}".format(np.min(cam_xx), np.max(cam_xx)))
    # logging.info("cam_yy: {}, {}".format(np.min(cam_yy), np.max(cam_yy)))
    # logging.info("cam_zz: {}, {}".format(np.min(cam_zz), np.max(cam_zz)))

    # # drift caused by skew
    # drift = r[0, 1] * cam_yy / cam_zz
    # min_drift = np.min(drift)
    # max_drift = np.max(drift)
    # logging.info("drift caused by skew (pixel): {}, {}, {}".format(min_drift, max_drift, max_drift - min_drift))

    # decompose the intrinsic
    # logging.info("normalized skew: {}, fx: {}, fy: {}, cx: {}, cy: {}".format(
    #     r[0, 1] / r[1, 1], r[0, 0], r[1, 1], r[0, 2] - r[0, 1] * r[1, 2] / r[1, 1], r[1, 2]))

    # check projection accuracy
    P_hat = np.dot(r, np.hstack((q, t)))
    result = np.dot(np.hstack((xx, yy, zz, all_ones)), P_hat.T)
    esti_col = result[:, 0:1] / result[:, 2:3]
    esti_row = result[:, 1:2] / result[:, 2:3]

    # max_row_err = np.max(np.abs(esti_row - row))
    # max_col_err = np.max(np.abs(esti_col - col))
    # logging.info('projection accuracy, max_row_err: {}, max_col_err: {}'.format(max_row_err, max_col_err))

    # pixel error
    proj_err = np.sqrt((esti_row - row) ** 2 + (esti_col - col) ** 2)
    #logging.info('proj_err (pixels), min, mean, median, max: {}, {}'.format(np.min(proj_err), np.mean(proj_err), np.median(proj_err), np.max(proj_err)))

    # check inverse projection accuracy
    # assume the matching are correct
    result = np.dot(np.hstack((col, row, all_ones)), np.linalg.inv(r.T))
    esti_cam_xx = result[:, 0:1]  # in camera coordinate
    esti_cam_yy = result[:, 1:2]
    esti_cam_zz = result[:, 2:3]

    # compute scale
    scale = (cam_xx * esti_cam_xx + cam_yy * esti_cam_yy + cam_zz * esti_cam_zz) / (
            esti_cam_xx * esti_cam_xx + esti_cam_yy * esti_cam_yy + esti_cam_zz * esti_cam_zz)
    # assert (np.all(scale > 0))
    esti_cam_xx = esti_cam_xx * scale
    esti_cam_yy = esti_cam_yy * scale
    esti_cam_zz = esti_cam_zz * scale

    # check accuarcy in camera coordinate frame
    # logging.info('inverse projection accuracy, cam xx: {}'.format(np.max(np.abs(esti_cam_xx - cam_xx))))
    # logging.info('inverse projection accuracy, cam yy: {}'.format(np.max(np.abs(esti_cam_yy - cam_yy))))
    # logging.info('inverse projection accuracy, cam zz: {}'.format(np.max(np.abs(esti_cam_zz - cam_zz))))

    # inv proj. err
    inv_proj_err = np.sqrt((esti_cam_xx - cam_xx) ** 2 + (esti_cam_yy - cam_yy) ** 2 + (esti_cam_zz - cam_zz) ** 2)
    #logging.info('inv_proj_err (meters), min, mean, median, max: {}, {}'.format(np.min(inv_proj_err), np.mean(inv_proj_err), np.median(inv_proj_err), np.max(inv_proj_err)))

    # check accuracy in object coordinate frame
    # result = np.dot(np.hstack((esti_cam_xx, esti_cam_yy, esti_cam_zz)) - translation, linalg.inv(q.T))
    # esti_xx = result[:, 0:1]
    # esti_yy = result[:, 1:2]
    # esti_zz = result[:, 2:3]

    # logging.info('inverse projection accuracy, xx: {}'.format(np.max(np.abs(esti_xx - xx))))
    # logging.info('inverse projection accuracy, yy: {}'.format(np.max(np.abs(esti_yy - yy))))
    # logging.info('inverse projection accuracy, zz: {}'.format(np.max(np.abs(esti_zz - zz))))

    return proj_err, inv_proj_err

