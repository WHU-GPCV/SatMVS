
import numpy as np


class Ellipsoid:
    '''
    based on (Coordinate Conversions and Transformation including Formulas, pdf, page 15)

    For WGS84:
        --ellipsoid_a = 6378137.000
        --inverse_flattening = 298.257223563

    '''
    a = 0 # semi-major axis
    b = 0 # semi-minor axis
    inv_f = 0 # inverse flattening 1
    f = 0 # flattening
    e = 0 # eccentricity
    sec_e = 0 # second eccentricity

    def __init__(self, ellipsoid_a=6378137.000, inverse_flattening=298.257223563):
        self.a = ellipsoid_a
        self.inv_f = inverse_flattening
        self.f = 1.0/self.inv_f
        self.b = self.a * (1 - self.f)
        self.e = np.sqrt(2*self.f - self.f * self.f)
        self.sec_e = np.sqrt((self.e * self.e)/(1 - self.e * self.e))

    def show_All_Info(self):
        print("========Info. for the Ellipsoid Defined==========")
        print("--semi-major axis: ", self.a)
        print("--semi-minor axis: ", self.b)
        print("--inverse flattening: ", self.inv_f)
        print("--flattening: ", self.f)
        print("--eccentricity: ", self.e)
        print("--second eccentricity: ", self.sec_e)
        print("=================================================")


class TransverseMercator:
    """
    Transverse_Mercator Projection
    based on (Coordinate Conversions and Transformation including Formulas, pdf, page 43, USGS formula)
    """

    def __init__(self, ellipsoid, latitude_origin=0.0, longitude_origin=0.0,
                 scale_factor=1.0, False_Easting=500000.0, False_Northing=0.0):
        self.M_PI = 3.14159265358979323846
        self.a = ellipsoid.a
        self.b = ellipsoid.b
        self.f = ellipsoid.f
        self.e = ellipsoid.e
        self.sec_e = ellipsoid.sec_e
        self.lat0_org = latitude_origin
        self.lon0_org = longitude_origin
        self.lat0 = latitude_origin / 180 * self.M_PI
        self.lon0 = longitude_origin / 180 * self.M_PI
        self.k0 = scale_factor
        self.FE = False_Easting
        self.FN = False_Northing

    def Show_Info(self):
        print("========Info. for the Projection Defined==========")
        print("------------------- Ellipsoid: -------------------")
        print("---- semi-major axis: ", self.a)
        print("---- semi-minor axis: ", self.b)
        print("---- flattening: ", self.f)
        print("---- eccentricity: ", self.e)
        print("---- second eccentricity: ", self.sec_e)
        print("--------------------------------------------------")
        print("--------------- Projection Para.: ----------------")
        print("---- latitude origin: ", self.lat0_org)
        print("---- longitude origin: ", self.lon0_org)
        print("---- scale factor: ", self.k0)
        print("---- False Easting: ", self.FE)
        print("---- False Northing: ", self.FN)
        print("--------------------------------------------------")
        print("=================================================")

    def proj(self, pts, reverse=False):
        """
        :param pts:
        :param reverse: True, EastNorth2latlon; False, latlon2EastNorth
        :return:
        """
        shape = pts.shape
        reshaped_pts = pts.reshape(-1, 2)
        if reverse:
            output = self.EastNorth2latlon(reshaped_pts)
        else:
            output = self.latlon2EastNorth(reshaped_pts)
        return output.reshape(shape)

    def latlon2EastNorth(self, pts):
        """
        For the calculation of easting and northing from latitude and longitude
        pts:(N, 2)
        """
        lat = pts[:, 0]
        lon = pts[:, 1]

        lat = lat / 180 * self.M_PI
        lon = lon / 180 * self.M_PI

        # Calculate Then the meridional arc distance from equator to the projection origin (M0)
        e_2 = self.e * self.e
        e_4 = e_2 * e_2
        e_6 = e_2 * e_2 * e_2
        M0 = self.a * ((1 - e_2 / 4 - 3 * e_4 / 64 - 5 * e_6 / 256) * self.lat0 -
                       (3 * e_2 / 8 + 3 * e_4 / 32 + 45 * e_6 / 1024) * np.sin(2 * self.lat0) +
                       (15 * e_4 / 256 + 45*e_6/1024)*np.sin(4*self.lat0) -
                       (35*e_6/3072)*np.sin(6*self.lat0))

        # calculate T C A v M
        cos_lat = np.cos(lat)
        sin_lat = np.sin(lat)
        tan_lat = np.tan(lat)

        T = tan_lat * tan_lat
        C = e_2 * cos_lat * cos_lat / (1 - e_2)
        A = (lon - self.lon0) * cos_lat
        v = self.a / np.sqrt(1 - e_2*sin_lat*sin_lat)
        M = self.a * ((1 - e_2 / 4 - 3 * e_4 / 64 - 5 * e_6 / 256) * lat -
                       (3 * e_2 / 8 + 3 * e_4 / 32 + 45 * e_6 / 1024) * np.sin(2 * lat) +
                       (15 * e_4 / 256 + 45*e_6/1024)*np.sin(4*lat) -
                       (35*e_6/3072)*np.sin(6*lat))

        # print("A = ", A, A.dtype)
        # print("T = ", T, T.dtype)
        # print("v = ", v, v.dtype)
        # print("C = ", C, C.dtype)
        # print("M = ", M, M.dtype)
        # print("M0 = ", M0, M0.dtype)

        A2 = A * A
        A3 = A * A * A
        E = self.FE + self.k0 * v * (A + (1 - T + C) * A3 / 6 + (
                    5 - 18 * T + T * T + 72 * C - 58 * self.sec_e * self.sec_e) * A2 * A3 / 120)
        N = self.FN + self.k0 * (M - M0 + v * tan_lat * (
                A2 / 2 + (5 - T + 9 * C + 4 * C * C) * A2 * A2 / 24 + (
                61 - 58 * T + T * T + 600 * C - 330 * self.sec_e * self.sec_e) * A3 * A3 / 720))

        return np.stack((E, N), axis=-1)

    def EastNorth2latlon(self, pts):
        """
        The reverse formulas to convert Easting and Northing projected coordinates to latitude and longitude
        pts:(N, 2)
        """
        E = pts[:, 0]
        N = pts[:, 1]

        e_2 = self.e * self.e
        e_4 = e_2 * e_2
        e_6 = e_2 * e_2 * e_2
        M0 = self.a * ((1 - e_2 / 4 - 3 * e_4 / 64 - 5 * e_6 / 256) * self.lat0 -
                       (3 * e_2 / 8 + 3 * e_4 / 32 + 45 * e_6 / 1024) * np.sin(2 * self.lat0) +
                       (15 * e_4 / 256 + 45 * e_6 / 1024) * np.sin(4 * self.lat0) -
                       (35 * e_6 / 3072) * np.sin(6 * self.lat0))

        # calculate e1 u1 M1
        temp_e = np.sqrt(1 - self.e * self.e)
        e1 = (1 - temp_e) / (1 + temp_e)
        M1 = M0 + (N - self.FN) / self.k0
        u1 = M1 / (self.a * (1 - e_2 / 4 - 3 * e_4 / 64 - 5 * e_6 / 256))

        # calculate lat1
        e1_2 = e1*e1
        lat1 = u1 + (3 * e1 / 2 - 27 * e1_2 * e1 / 32) * np.sin(2 * u1) + (
                21 * e1_2 / 16 - 55 * e1_2 * e1_2 / 32) * np.sin(4 * u1) + (
                151 * e1_2 * e1 / 96) * np.sin(6 * u1) + (
                1097 * e1_2 * e1_2 / 512) * np.sin(8 * u1)

        temp = np.sqrt(1 - e_2 * np.sin(lat1) * np.sin(lat1))
        v1 = self.a / temp
        p1 = self.a * (1 - e_2) / (temp * temp * temp)
        T1 = np.tan(lat1) * np.tan(lat1)

        C1 = self.sec_e * np.cos(lat1)
        C1 = C1 * C1

        D = (E - self.FE) / (v1 * self.k0)

        # calculate lat, lon
        D2 = D * D
        D3 = D2 * D
        sece_2 = self.sec_e * self.sec_e

        # print("e1 = ", e1)
        # print("M0 = ", M0)
        # print("M1 = ", M1)
        # print("lat1 = ", lat1)
        # print("p1 = ", p1)
        # print("T1 = ", T1)
        # print("u1 = ", u1)
        # print("v1 = ", v1)
        # print("D = ", D)
        # print("C1 = ", C1)

        lat = lat1 - (v1 * np.tan(lat1) / p1) * (
                    D2 / 2 - (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * sece_2) * D2 * D2 / 24 + (
                        61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * sece_2 - 3 * C1 * C1) * D3 * D3 / 720)
        lon = self.lon0 + (D - (1 + 2 * T1 + C1) * D3 / 6 + (
                    5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * sece_2 + 24 * T1 * T1) * D2 * D3 / 120) / np.cos(lat1)

        lat = lat * 180 / self.M_PI
        lon = lon * 180 / self.M_PI

        return np.stack((lat, lon), axis=-1)


def Test():
    wgs84 = Ellipsoid()
    proj = TransverseMercator(wgs84, 0, 123)
    # proj.Show_Info()

    pts = np.zeros((100, 192, 96, 2), dtype=np.float64)
    pts[:, :, :, 0] = 29.267563
    pts[:, :, :, 1] = 120.653181

    import time

    start = time.time()
    out = proj.proj(pts)
    back = proj.proj(out, reverse=True)
    end = time.time()

    print(out)
    print(back)
    print("finished in ", end - start, " s")


if __name__ == "__main__":
    wgs84 = Ellipsoid()
    proj = TransverseMercator(wgs84, 0, 123)

    pts = np.zeros((2, 2), dtype=np.float64)
    pts[0, 0] = 29.267563
    pts[0, 1] = 120.653181
    pts[1, 0] = 29.26756264
    pts[1, 1] = 120.65318143

    proj_pts = proj.proj(pts, False)
    distance = (proj_pts[0][0] - proj_pts[1][0]) * (proj_pts[0][0] - proj_pts[1][0]) + (
                proj_pts[0][1] - proj_pts[1][1]) * (proj_pts[0][1] - proj_pts[1][1])

    print(np.sqrt(distance))
