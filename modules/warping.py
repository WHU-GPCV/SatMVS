
import torch
import torch.nn.functional as F


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]

    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    # depth_values = -depth_values ???
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj)) # Tcw
        #proj = torch.matmul(torch.inverse(src_proj), ref_proj)   # Twc
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x))).double()  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(
            batch, 1, num_depth, -1).double()  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy.float()

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


# RPC warping, Using Quaternary Cubic Form.
#  It consumes a lot of memory.
def cal_qc_opt(x, T):
    # optimal contaction path(To be continued)
    return torch.einsum("abcde, abcdf, abcdg, aefg->abcd", [x, x, x, T])


def cal_qc(x, T):
    # quaternary cubic form
    # this contaction path can finish rpc warping in less time but requires more memory.
    return torch.einsum("abcde, abcdf, abcdg, aefg->abcd", [x, x, x, T])


def RPC_Obj2Photo_enisum(inlat, inlon, inhei, rpc):
    # inlat: (B, ndepth, H, W)
    # inlon:  (B, ndepth, H, W)
    # inhei:  (B, ndepth, H, W)
    # rpc: (B, 170)

    with torch.no_grad():
        lat = inlat.clone()
        lon = inlon.clone()
        hei = inhei.clone()

        lat -= rpc["lat_off"].view(-1, 1, 1, 1)
        lat /= rpc["lat_scale"].view(-1, 1, 1, 1)

        lon -= rpc["lon_off"].view(-1, 1, 1, 1)
        lon /= rpc["lon_scale"].view(-1, 1, 1, 1)

        hei -= rpc["height_off"].view(-1, 1, 1, 1)
        hei /= rpc["height_scale"].view(-1, 1, 1, 1)

        ones = torch.ones_like(hei, dtype=torch.double, device=hei.device)
        x = torch.stack((ones, lon, lat, hei), dim=-1)  # B, D, H, W, 4

        samp = cal_qc(x, rpc["samp_num_tensor"])
        samp /= cal_qc(x, rpc["samp_den_tensor"])
        line = cal_qc(x, rpc["line_num_tensor"])
        line /= cal_qc(x, rpc["line_den_tensor"])

        samp *= rpc["samp_scale"].view(-1, 1, 1, 1)
        samp += rpc["samp_off"].view(-1, 1, 1, 1)
        line *= rpc["line_scale"].view(-1, 1, 1, 1)
        line += rpc["line_off"].view(-1, 1, 1, 1)

    return samp, line


def RPC_Photo2Obj_enisum(insamp, inline, inhei, rpc):
    # insamp: (B, ndepth*H* W)
    # inline:  (B, ndepth*H* W)
    # inhei:  (B, ndepth*H* W)
    # rpc: (B, 170)

    # import time

    with torch.no_grad():
        # torch.cuda.synchronize()
        # t0 = time.time()
        samp = insamp.clone()
        line = inline.clone()
        hei = inhei.clone()

        samp -= rpc["samp_off"].view(-1, 1, 1, 1)

        samp /= rpc["samp_scale"].view(-1, 1, 1, 1)

        line -= rpc["line_off"].view(-1, 1, 1, 1)
        line /= rpc["line_scale"].view(-1, 1, 1, 1)

        hei -= rpc["height_off"].view(-1, 1, 1, 1)
        hei /= rpc["height_scale"].view(-1, 1, 1, 1)
        ones = torch.ones_like(hei, dtype=torch.double, device=hei.device)
        x = torch.stack((ones, line, samp, hei), dim=-1)  # B, D, H, W, 4

        # torch.einsum("ija, ijb, ijc, abc->ij", [x, x, x, coef])
        # print(x.shape, rpc["lat_num_tensor"].shape)
        lat = cal_qc(x, rpc["lat_num_tensor"])
        lat /= cal_qc(x, rpc["lat_den_tensor"])

        lon = cal_qc(x, rpc["lon_num_tensor"])
        lon /= cal_qc(x, rpc["lon_den_tensor"])

        lat *= rpc["lat_scale"].view(-1, 1, 1, 1)
        lat += rpc["lat_off"].view(-1, 1, 1, 1)
        lon *= rpc["lon_scale"].view(-1, 1, 1, 1)
        lon += rpc["lon_off"].view(-1, 1, 1, 1)

    return lat, lon


def rpc_warping_enisum(src_fea, src_rpc, ref_rpc, depth_values):
    # src_fea: [B, C, H, W]
    # src_rpc: dict
    # ref_rpc: dict
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]

    # import time
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.double, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.double, device=src_fea.device)])

        x = x.repeat(batch, num_depth, 1, 1)
        y = y.repeat(batch, num_depth, 1, 1)

        depth_values = depth_values.double()

        lat, lon = RPC_Photo2Obj_enisum(x, y, depth_values, ref_rpc)
        samp, line = RPC_Obj2Photo_enisum(lat, lon, depth_values, src_rpc)  # (B, ndepth, H, W)

        samp = samp.float()
        line = line.float()

        proj_x_normalized = samp / ((width - 1) / 2) - 1
        proj_y_normalized = line / ((height - 1) / 2) - 1
        proj_x_normalized = proj_x_normalized.view(batch, num_depth, height * width)
        proj_y_normalized = proj_y_normalized.view(batch, num_depth, height * width)

        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


# RPC warping
# To reduce memory consumption
def RPC_PLH_COEF(P, L, H, coef):
    # P: (batch, n_num)

    # import time
    # start = time.time()
    with torch.no_grad():
        coef[:, :, 1] = L
        coef[:, :, 2] = P
        coef[:, :, 3] = H
        coef[:, :, 4] = L * P
        coef[:, :, 5] = L * H
        coef[:, :, 6] = P * H
        coef[:, :, 7] = L * L
        coef[:, :, 8] = P * P
        coef[:, :, 9] = H * H
        coef[:, :, 10] = P * coef[:, :, 5]
        coef[:, :, 11] = L * coef[:, :, 7]
        coef[:, :, 12] = L * coef[:, :, 8]
        coef[:, :, 13] = L * coef[:, :, 9]
        coef[:, :, 14] = L * coef[:, :, 4]
        coef[:, :, 15] = P * coef[:, :, 8]
        coef[:, :, 16] = P * coef[:, :, 9]
        coef[:, :, 17] = L * coef[:, :, 5]
        coef[:, :, 18] = P * coef[:, :, 6]
        coef[:, :, 19] = H * coef[:, :, 9]
        # torch.cuda.synchronize()
        # end = time.time()

        # print(P.shape, L.shape, H.shape)
        # print((H*H*H).shape)
    # if P.shape[1] == 7426048:
        # print(P.shape, end-start, "s")
    # return coef


def RPC_Obj2Photo(inlat, inlon, inhei, rpc, coef):
    # inlat: (B, ndepth*H* W)
    # inlon:  (B, ndepth*H* W)
    # inhei:  (B, ndepth*H*W)
    # rpc: (B, 170)

    with torch.no_grad():
        lat = inlat.clone()
        lon = inlon.clone()
        hei = inhei.clone()

        lat -= rpc[:, 2].view(-1, 1) # self.LAT_OFF
        lat /= rpc[:, 7].view(-1, 1) # self.LAT_SCALE

        lon -= rpc[:, 3].view(-1, 1) # self.LONG_OFF
        lon /= rpc[:, 8].view(-1, 1) # self.LONG_SCALE

        hei -= rpc[:, 4].view(-1, 1) # self.HEIGHT_OFF
        hei /= rpc[:, 9].view(-1, 1) # self.HEIGHT_SCALE

        RPC_PLH_COEF(lat, lon, hei, coef)

        # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
        samp = torch.sum(coef * rpc[:, 50: 70].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[:, 70:90].view(-1, 1, 20), dim=-1)
        line = torch.sum(coef * rpc[:, 10: 30].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[:, 30:50].view(-1, 1, 20), dim=-1)

        samp *= rpc[:, 6].view(-1, 1) # self.SAMP_SCALE
        samp += rpc[:, 1].view(-1, 1) # self.SAMP_OFF

        line *= rpc[:, 5].view(-1, 1) # self.LINE_SCALE
        line += rpc[:, 0].view(-1, 1) # self.LINE_OFF

    return samp, line


def RPC_Photo2Obj(insamp, inline, inhei, rpc, coef):
    # insamp: (B, ndepth*H* W)
    # inline:  (B, ndepth*H* W)
    # inhei:  (B, ndepth*H* W)
    # rpc: (B, 170)

    # import time

    with torch.no_grad():
        # torch.cuda.synchronize()
        # t0 = time.time()
        samp = insamp.clone()
        line = inline.clone()
        hei = inhei.clone()

        samp -= rpc[:, 1].view(-1, 1) # self.SAMP_OFF

        samp /= rpc[:, 6].view(-1, 1) # self.SAMP_SCALE

        line -= rpc[:, 0].view(-1, 1) # self.LINE_OFF
        line /= rpc[:, 5].view(-1, 1) # self.LINE_SCALE

        hei -= rpc[:, 4].view(-1, 1) # self.HEIGHT_OFF
        hei /= rpc[:, 9].view(-1, 1) # self.HEIGHT_SCALE
        # t1 = time.time()
        RPC_PLH_COEF(samp, line, hei, coef)
        # torch.cuda.synchronize()
        # t2 = time.time()

        # coef: (B, ndepth*H*W, 20) rpc[:, 90:110] (B, 20)
        lat = torch.sum(coef * rpc[:, 90:110].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[:, 110:130].view(-1, 1, 20), dim=-1)
        lon = torch.sum(coef * rpc[:, 130:150].view(-1, 1, 20), dim=-1) / torch.sum(
            coef * rpc[:, 150:170].view(-1, 1, 20), dim=-1)

        # torch.cuda.synchronize()
        # t3 = time.time()

        lat *= rpc[:, 7].view(-1, 1)
        lat += rpc[:, 2].view(-1, 1)

        lon *= rpc[:, 8].view(-1, 1)
        lon += rpc[:, 3].view(-1, 1)

        # torch.cuda.synchronize()
        # t4 = time.time()
    # if (insamp.shape[1]==7426048):
        # print(t1 - t0, "s")
        # print(t2 - t1, "s")
        # print(t3 - t2, "s")
        # print(t4 - t3, "s")
        # print()
    return lat, lon


def rpc_warping(src_fea, src_rpc, ref_rpc, depth_values, coef):
    # src_fea: [B, C, H, W]
    # src_rpc: [B, 170]
    # ref_rpc: [B, 170]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]

    # import time
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.double, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.double, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y = y.view(1, 1, height, width).repeat(batch, num_depth, 1, 1) # (B, ndepth, H, W)
        x = x.view(1, 1, height, width).repeat(batch, num_depth, 1, 1)

        if len(depth_values.shape) == 2:
            h = depth_values.view(batch, num_depth, 1, 1).double().repeat(1, 1, height, width) # (B, ndepth, H, W)
        else:
            h = depth_values # (B, ndepth, H, W)

        x = x.view(batch, -1)
        y = y.view(batch, -1)
        h = h.view(batch, -1)
        h = h.double()

        # start = time.time()
        lat, lon = RPC_Photo2Obj(x, y, h, ref_rpc, coef)
        samp, line = RPC_Obj2Photo(lat, lon, h, src_rpc, coef) # (B, ndepth*H*W)
        # end = time.time()

        # print(torch.mean(samp - x), torch.var(samp - x))
        # print(torch.mean(line - y), torch.var(line - y))

        samp = samp.float()
        line = line.float()

        proj_x_normalized = samp / ((width - 1) / 2) - 1
        proj_y_normalized = line / ((height - 1) / 2) - 1
        proj_x_normalized = proj_x_normalized.view(batch, num_depth, height * width)
        proj_y_normalized = proj_y_normalized.view(batch, num_depth, height * width)

        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    # if height == 592*4:
        # print(end - start, "s")

    return warped_src_fea


