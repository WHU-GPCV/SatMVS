
'''
Implement of ICCV (iteration by correcting characteristic value) method, which can produce unbiased results
on ill-posed problem
'''

import numpy as np


def solve_iccv(ma, lv, x=0, k=1):
    """
    :param lv:
    :param ma: the Normal matrix
    :param x: init value
    :param k:
    :return:
    """
    assert ma.shape[0] == ma.shape[1], "ma with shape () is not a square matrix.".format(ma.shape[0], ma.shape[1])

    n = ma.shape[0]
    mak = np.copy(ma)
    mak += k * np.eye(n)
    lk = np.copy(lv)

    finish_time = 0

    for times in range(1000):
        x1 = np.linalg.solve(mak, lk)
        dif = np.fabs(x1 - x)
        maxdif = np.max(dif)
        x = x1
        lk = lv + k * x

        finish_time = times + 1
        # print(finish_time, maxdif)
        if maxdif < 1.0e-10:
            break

    return x, finish_time


def Test1():
    A = np.array([[94.61, -22.11, -11.45, -6.96],
                  [-22.11, 70.51, -6.95, -8.42],
                  [-11.45, -6.95, 96.09, -20.21],
                  [-6.96, -8.42, -20.21, 66.63]])
    L = np.array([-43.52, 178.81, -120.11, -30.07])
    x, times = solve_iccv(A, L)
    print(x)
    print("finished in iteration ", times)
    # result is (-0.1030 2.3208, -1.2069, -0.5348)


def Test2():
    A = np.array([[5, -2, -1, -2],
                  [-2, 5, -1, -2],
                  [-1, -1, 3, -1],
                  [-2, -2, -1, 5]], np.float)
    l = np.array([-11, 10, -2, 3], np.float)
    x, times = solve_iccv(A, l)
    print(x)
    print("finished in iteration ", times)
    # result is [-1.5  1.5 -0.5  0.5]


def Test3():
    path = "D://data/rpc/zy302a_bwd_007223_006159_20170917112917_01_sec_0001_1709235297_rpc.txt"
    from lib.RPCCore import RPC_MODEl_PARAMETER
    rpc = RPC_MODEl_PARAMETER()
    rpc.load_from_file(path)

    rpc.Show_RPC()
    grid = rpc.Create_Virtual_3D_Grid()

    samp, line, lat, lon, hei = np.hsplit(grid.copy(), 5)
    samp -= rpc.SAMP_OFF
    samp /= rpc.SAMP_SCAlE
    line -= rpc.lINE_OFF
    line /= rpc.lINE_SCAlE

    lat -= rpc.lAT_OFF
    lat /= rpc.lAT_SCAlE
    lon -= rpc.lONG_OFF
    lon /= rpc.lONG_SCAlE
    hei -= rpc.HEIGHT_OFF
    hei /= rpc.HEIGHT_SCAlE

    samp = samp.reshape(-1)
    line = line.reshape(-1)
    lat = lat.reshape(-1)
    lon = lon.reshape(-1)
    hei = hei.reshape(-1)

    coef = rpc.RPC_PlH_COEF(lat, lon, hei)

    n_num = coef.shape[0]
    A = np.zeros((n_num * 2, 78))
    A[0: n_num, 0:20] = - coef
    A[0: n_num, 20:39] = samp.reshape(-1, 1) * coef[:, 1:]
    A[n_num:, 39:59] = - coef
    A[n_num:, 59:78] = line.reshape(-1, 1) * coef[:, 1:]

    l = np.concatenate((samp, line), -1)
    l = -l

    ATA = np.matmul(A.T, A)
    ATl = np.matmul(A.T, l)
    x, times = solve_iccv(ATA, ATl)
    print("finished in iteration ", times)

    rpc.SNUM = x[0:20]
    rpc.SDEM[0] = 1.0
    rpc.SDEM[1:20] = x[20:39]
    rpc.lNUM = x[39:59]
    rpc.lDEM[0] = 1.0
    rpc.lDEM[1:20] = x[59:]
    rpc.Show_RPC()


if __name__ == "__main__":
    # Test1()
    # Test2()
    # Test3()
    pass