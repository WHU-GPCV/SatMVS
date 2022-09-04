
import os
import cupy as cp
import numpy as np


class RPCModelParameter:
    def __init__(self, data=np.zeros(170, dtype=np.float64)):
        data = cp.asarray(data)

        self.LINE_OFF, self.SAMP_OFF, self.LAT_OFF, self.LONG_OFF, self.HEIGHT_OFF = data[0:5]
        self.LINE_SCALE, self.SAMP_SCALE, self.LAT_SCALE, self.LONG_SCALE, self.HEIGHT_SCALE = data[5:10]

        self.LNUM = self.to_T(data[10:30])
        self.LDEM = self.to_T(data[30:50])
        self.SNUM = self.to_T(data[50:70])
        self.SDEM = self.to_T(data[70:90])

        self.LATNUM = self.to_T(data[90:110])
        self.LATDEM = self.to_T(data[110:130])
        self.LONNUM = self.to_T(data[130:150])
        self.LONDEM = self.to_T(data[150:170])

    @staticmethod
    def to_T(data):
        assert data.shape[0] == 20 and len(data.shape) == 1
        coeff_tensor = cp.array(
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

    @staticmethod
    def QC_cal(x, T):
        assert x.shape[0] == 4 and T.shape == (4, 4, 4)

        # x (i, n) (j, n) (k, n)
        # T (i, j, k)
        # print(x.shape, T.shape)
        xT = cp.tensordot(x, T, axes=[1, 0])
        print(xT.shape)
        xxT = cp.tensordot(x, xT, axes=[1, 1])
        print(xxT.shape)
        xxxT = cp.tensordot(x, xxT, axes=[1, 2])
        print(xxxT.shape)

        # Tx = cp.tensordot(x, T, axes=[0, 1]) # (100, 4, 4)
        # Txx = cp.tensordot(x, Tx, axes=[(0, 1), (1, 0)]) # (4)
        # y = cp.tensordot(x, Txx, axes=[0, 0])

        return y

    @staticmethod
    def QC_cal_en(x, T):
        assert x.shape[0] == 4 and T.shape == (4, 4, 4)
        y = cp.einsum('ijk, in, jn, kn->n', T, x, x, x)

        return y

    def load_dirpc_from_file(self, filepath):
        """
        Read the direct and inverse rpc from a file
        :param filepath:
        :return:
        """
        if os.path.exists(filepath) is False:
            print("Error#001: cann't find " + filepath + " in the file system!")
            return

        with open(filepath, 'r') as f:
            all_the_text = f.read().splitlines()

        data = [text.split(' ')[1] for text in all_the_text]
        # print(data)
        data = cp.array(data, dtype=cp.float64)

        self.LINE_OFF, self.SAMP_OFF, self.LAT_OFF, self.LONG_OFF, self.HEIGHT_OFF = data[0:5]
        self.LINE_SCALE, self.SAMP_SCALE, self.LAT_SCALE, self.LONG_SCALE, self.HEIGHT_SCALE = data[5:10]

        self.LNUM = self.to_T(data[10:30])
        self.LDEM = self.to_T(data[30:50])
        self.SNUM = self.to_T(data[50:70])
        self.SDEM = self.to_T(data[70:90])

        self.LATNUM = self.to_T(data[90:110])
        self.LATDEM = self.to_T(data[110:130])
        self.LONNUM = self.to_T(data[130:150])
        self.LONDEM = self.to_T(data[150:170])

    def RPC_OBJ2PHOTO(self, inlat, inlon, inhei):
        assert inlat.shape == inlon.shape and inlon.shape == inhei.shape
        lat = cp.asarray(inlat)
        lon = cp.asarray(inlon)
        hei = cp.asarray(inhei)

        tmp = cp.ones_like(lat)
        x = cp.stack((tmp, lon, lat, hei), axis=0)

        x[1] -= self.LONG_OFF
        x[1] /= self.LONG_SCALE

        x[2] -= self.LAT_OFF
        x[2] /= self.LAT_SCALE

        x[3] -= self.HEIGHT_OFF
        x[3] /= self.HEIGHT_SCALE

        samp = self.QC_cal_en(x, self.SNUM) / self.QC_cal_en(x, self.SDEM)
        line = self.QC_cal_en(x, self.LNUM) / self.QC_cal_en(x, self.LDEM)

        samp *= self.SAMP_SCALE
        samp += self.SAMP_OFF

        line *= self.LINE_SCALE
        line += self.LINE_OFF

        return cp.asnumpy(samp), cp.asnumpy(line)

    def RPC_PHOTO2OBJ(self, insamp, inline, inhei):
        assert insamp.shape == inline.shape and inline.shape == inhei.shape
        samp = cp.asarray(insamp)
        line = cp.asarray(inline)
        hei = cp.asarray(inhei)

        tmp = cp.ones_like(samp)
        x = cp.stack((tmp, line, samp, hei), axis=0)

        x[1] -= self.LINE_OFF
        x[1] /= self.LINE_SCALE

        x[2] -= self.SAMP_OFF
        x[2] /= self.SAMP_SCALE

        x[3] -= self.HEIGHT_OFF
        x[3] /= self.HEIGHT_SCALE

        lat = self.QC_cal_en(x, self.LATNUM) / self.QC_cal_en(x, self.LATDEM)
        lon = self.QC_cal_en(x, self.LONNUM) / self.QC_cal_en(x, self.LONDEM)

        lat *= self.LAT_SCALE
        lat += self.LAT_OFF

        lon *= self.LONG_SCALE
        lon += self.LONG_OFF

        return cp.asnumpy(lat), cp.asnumpy(lon)


if __name__ == "__main__":
    rpcs = []
    heights = []
    for i in range(3):
        from dataset.data_io import load_rpc_as_array, load_pfm

        rpc_path = "D:/pipeline_result/index7_idx1/rpc/{}/block0000.rpc".format(i)

        rpc, _, _ = load_rpc_as_array(rpc_path)
        rpcs.append(rpc)

        height_map_path = "D:/pipeline_result/index7_idx1/mvs_results/{}/init/block0000.pfm".format(i)
        height_map = load_pfm(height_map_path)
        heights.append(height_map)

    ref_rpc = RPCModelParameter(rpcs[2])
    src_rpc = RPCModelParameter(rpcs[0])

    import time

    height, width = heights[0].shape
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    hei_ref = heights[0].reshape([-1])

    start = time.time()
    lat, lon = ref_rpc.RPC_PHOTO2OBJ(x_ref.astype(np.float), y_ref.astype(np.float), hei_ref)
    print(lat, lon)

    samp, line = src_rpc.RPC_OBJ2PHOTO(lat, lon, hei_ref)
    print(samp, line)
    end = time.time()
    print(end - start)

