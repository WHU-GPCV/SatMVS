import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.module import *
from modules.warping import *
from modules.depth_range import *


def compute_depth(feats, proj_mats, depth_samps, cost_reg, lamb, geo_model, is_training=False, use_qc=False):
    '''

    :param feats: [(B, C, H, W), ] * num_views
    :param proj_mats: [()]
    :param depth_samps:
    :param cost_reg:
    :param lamb:
    :return:
    '''
    if not use_qc:
        proj_mats = torch.unbind(proj_mats, 1)

    num_views = len(feats)
    num_depth = depth_samps.shape[1]

    assert len(proj_mats) == num_views, "Different number of images and projection matrices"

    ref_feat, src_feats = feats[0], feats[1:]
    ref_proj, src_projs = proj_mats[0], proj_mats[1:]

    ref_volume = ref_feat.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
    volume_sum = ref_volume
    volume_sq_sum = ref_volume ** 2
    del ref_volume

    if geo_model == "rpc" and not use_qc:
        # Create tensor in advance to save time
        b_num, f_num, img_h, img_w = ref_feat.shape
        coef = torch.ones((b_num, img_h * img_w * num_depth, 20), dtype=torch.double).cuda()
    else:
        coef = None

    #todo optimize impl
    for src_fea, src_proj in zip(src_feats, src_projs):
        if geo_model == "rpc" and not use_qc:
            warped_volume = rpc_warping(src_fea, src_proj, ref_proj, depth_samps, coef)
        elif geo_model == "rpc" and use_qc:
            warped_volume = rpc_warping_enisum(src_fea, src_proj, ref_proj, depth_samps)
        else:
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_samps)

        if is_training:
            volume_sum = volume_sum + warped_volume
            volume_sq_sum = volume_sq_sum + warped_volume ** 2
        else:
            volume_sum += warped_volume
            volume_sq_sum += warped_volume.pow_(2) #in_place method
        del warped_volume
    volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

    prob_volume_pre = cost_reg(volume_variance).squeeze(1)
    prob_volume = F.softmax(prob_volume_pre, dim=1)
    # print(depth_samps.dtype)
    depth = depth_regression(prob_volume, depth_values=depth_samps)

    with torch.no_grad():
        prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1),
                                            stride=1, padding=0).squeeze(1)
        depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device,
                                                                              dtype=torch.float)).long()
        depth_index = depth_index.clamp(min=0, max=num_depth - 1)
        prob_conf = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

    samp_variance = (depth_samps - depth.unsqueeze(1)) ** 2
    exp_variance = lamb * torch.sum(samp_variance * prob_volume, dim=1, keepdim=False) ** 0.5

    return {"depth": depth, "photometric_confidence": prob_conf, 'variance': exp_variance}


class UCSNet(nn.Module):
    def __init__(self, geo_model, lamb=1.5, stage_configs=[64, 32, 8], grad_method="detach", base_chs=[8, 8, 8],
                 feat_ext_ch=8, use_qc=False):
        super(UCSNet, self).__init__()
        self.geo_model = geo_model
        assert self.geo_model in ["rpc", "pinhole"]
        self.stage_configs = stage_configs
        self.grad_method = grad_method
        self.base_chs = base_chs
        self.lamb = lamb
        self.num_stage = len(stage_configs)
        self.use_qc = use_qc
        self.ds_ratio = {"stage1": 4.0,
                         "stage2": 2.0,
                         "stage3": 1.0
                         }

        self.feature_extraction = FeatureNet(base_channels=feat_ext_ch, num_stage=self.num_stage)

        self.cost_regularization = nn.ModuleList([CostRegNet(
            in_channels=self.feature_extraction.out_channels[i], base_channels=self.base_chs[i])
            for i in range(self.num_stage)])

    def forward(self, imgs, proj_matrices, depth_values):
        features = []
        for nview_idx in range(imgs.shape[1]):
            img = imgs[:, nview_idx]
            features.append(self.feature_extraction(img))

        outputs = {}
        depth, cur_depth, exp_var = None, None, None
        depth_min = depth_values[:, 0]
        depth_max = depth_values[:, -1]

        for stage_idx in range(self.num_stage):
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.ds_ratio["stage{}".format(stage_idx + 1)]
            cur_h = img.shape[2] // int(stage_scale)
            cur_w = img.shape[3] // int(stage_scale)

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                    exp_var = exp_var.detach()
                else:
                    cur_depth = depth

                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                                                [cur_h, cur_w], mode='bilinear',align_corners=False)
                exp_var = F.interpolate(exp_var.unsqueeze(1), [cur_h, cur_w], mode='bilinear',align_corners=False)

            else:
                cur_depth = depth_values

            depth_range_samples = uncertainty_aware_samples(cur_depth=cur_depth,
                                                            depth_min=depth_min,
                                                            depth_max=depth_max,
                                                            exp_var=exp_var,
                                                            ndepth=self.stage_configs[stage_idx],
                                                            dtype=img[0].dtype,
                                                            device=img[0].device,
                                                            shape=[img.shape[0], cur_h, cur_w])

            outputs_stage = compute_depth(features_stage, proj_matrices_stage,
                                          depth_samps=depth_range_samples,
                                          cost_reg=self.cost_regularization[stage_idx],
                                          lamb=self.lamb,
                                          geo_model=self.geo_model,
                                          is_training=self.training,
                                          use_qc=self.use_qc)

            depth = outputs_stage['depth']
            exp_var = outputs_stage['variance']

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage

        return outputs

