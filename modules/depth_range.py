import torch


def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
    # cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel).clamp(min=0.0)   #(B, H, W)
    # cur_depth_max = (cur_depth_min + (ndepth - 1) * depth_inteval_pixel).clamp(max=max_depth)

    assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (
            torch.arange(0, ndepth, device=cur_depth.device, dtype=cur_depth.dtype,
                         requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1))

    return depth_range_samples


def get_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, device, dtype, shape):
    #shape: (B, H, W)
    #cur_depth: (B, H, W) or (B, D)
    #return depth_range_samples: (B, D, H, W)
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )

        depth_range_samples = cur_depth_min.unsqueeze(1) + (
                torch.arange(0, ndepth, device=device, dtype=dtype,
                             requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)

        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, shape[1], shape[2]) #(B, D, H, W)

    else:
        depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape)

    return depth_range_samples


def uncertainty_aware_samples(cur_depth, depth_min, depth_max, exp_var, ndepth, device, dtype, shape):
    eps = 1e-12
    if cur_depth.dim() == 2:
        #must be the first stage
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )
        depth_range_samples = cur_depth_min.unsqueeze(1) + (
                torch.arange(0, ndepth, device=device, dtype=dtype,
                             requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) # (B, D)
        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, shape[1], shape[2]) # (B, D, H, W)
    else:
        #
        batch_num, d_num, w_num, h_num = cur_depth.shape
        low_bound = cur_depth - exp_var
        high_bound = cur_depth + exp_var

        tensor_depth_min = depth_min.view(batch_num, 1, 1, 1).repeat(1, 1, w_num, h_num)
        tensor_depth_max = depth_max.view(batch_num, 1, 1, 1).repeat(1, 1, w_num, h_num)

        # print(low_bound.shape, tensor_depth_min.shape)
        # print(torch.max(high_bound), torch.min(low_bound))
        lower_than_min = (low_bound - tensor_depth_min) < 0
        higher_than_max = (high_bound - tensor_depth_max) > 0
        low_bound[lower_than_min] = tensor_depth_min[lower_than_min]
        high_bound[higher_than_max] = tensor_depth_max[higher_than_max]
        # print(torch.max(depth_max), torch.min(depth_min))
        # print(torch.max(high_bound), torch.min(low_bound))

        # assert exp_var.min() >= 0, exp_var.min()
        assert ndepth > 1

        step = (high_bound - low_bound) / (float(ndepth) - 1)
        new_samps = []
        for i in range(int(ndepth)):
            new_samps.append(low_bound + step * i + eps)

        depth_range_samples = torch.cat(new_samps, 1)
        # assert depth_range_samples.min() >= 0, depth_range_samples.min()

    return depth_range_samples
