import argparse
import torch
import os
import datetime
from tensorboardX import SummaryWriter
import sys
from dataset import find_dataset_def
import torch.backends.cudnn as cudnn
from networks.casmvs import CascadeMVSNet
from networks.ucs import UCSNet
from networks.casred import CascadeREDNet, Infer_CascadeREDNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
from tools.utils import *
from dataset.data_io import save_pfm
import matplotlib.pyplot as plt
from networks.loss import cas_mvsnet_loss

cudnn.benchmark = True


parser = argparse.ArgumentParser(description='A PyTorch Implementation')
parser.add_argument('--mode', default='test', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default="casmvs", help='select model', choices=['red', "casmvs", "ucs"])
parser.add_argument('--geo_model', default="pinhole", help='select dataset', choices=["rpc", "pinhole"])
parser.add_argument('--use_qc', default=False, help="whether to use Quaternary Cubic Form for RPC warping.")
parser.add_argument('--dataset_root', default='/mnt/gj/WHU_TLC', help='dataset root')

parser.add_argument('--loadckpt', default="checkpoints/casmvs/rpc/model.ckpt", help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', default=False, help='continue to train the model')
# input parameters
parser.add_argument('--view_num', type=int, default=3, help='Number of images.')
parser.add_argument('--ref_view', type=int, default=2)

parser.add_argument('--batch_size', type=int, default=1, help='train batch size')

# Cascade parameters
parser.add_argument('--ndepths', type=str, default="64,32,8", help='ndepths')
parser.add_argument('--min_interval', type=float, default=2.5, help='min_interval in the bottom stage')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--lamb', type=float, default=1.5, help="lamb in ucs-net")

parser.add_argument('--dlossw', type=str, default="0.5,1.0,2.0", help='depth loss weight for different stage')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
# network architecture
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2",
                    help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--gpu_id', type=str, default="0")

# parse arguments and check
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

trainpath = "{}/open_dataset_{}/train".format(args.dataset_root, args.geo_model)
testpath = "{}/open_dataset_{}/test".format(args.dataset_root, args.geo_model)

if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if testpath is None:
    testpath = trainpath
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

cur_log_dir = os.path.join(args.logdir, "{}/{}".format(args.model, args.geo_model)).replace("\\", "/")

ck_dir = os.path.join(cur_log_dir, "train").replace("\\", "/")
if not os.path.exists(ck_dir):
    os.makedirs(ck_dir)

# create logger for mode "train" and "testall"
if args.mode == "train":
    if not os.path.isdir(cur_log_dir):
        os.makedirs(cur_log_dir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(cur_log_dir)

print("argv:", sys.argv[1:])
print_args(args)

# dataset, dataloader
MVSDataset = find_dataset_def(args.geo_model)
train_dataset = MVSDataset(trainpath, "train", args.view_num, ref_view=args.ref_view, use_qc=args.use_qc)
test_dataset = MVSDataset(testpath, "test", args.view_num, ref_view=args.ref_view, use_qc=args.use_qc)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=0, drop_last=False)

# model
model = None
if args.model == "casmvs":
    model = CascadeMVSNet(min_interval=args.min_interval,
                          ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          geo_model=args.geo_model, use_qc=args.use_qc)
    print("===============> Model: Cascade MVS Net ===========>")
elif args.model == "ucs":
    model = UCSNet(lamb=args.lamb, stage_configs=[int(nd) for nd in args.ndepths.split(",") if nd],
                   base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                   geo_model=args.geo_model, use_qc=args.use_qc)
    print("===============> Model: UCS-Net ===========>")
elif args.model == "red":
    model = CascadeREDNet(min_interval=args.min_interval,
                          ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                          geo_model=args.geo_model, use_qc=args.use_qc)
    print("===============> Model: Cascade RED Net ===========>")
else:
    raise Exception("{}? Not implemented yet!".format(args.model))


if args.mode in ["train", "test"]:
    model = nn.DataParallel(model)
model.cuda()

# model_loss = mvsnet_loss  # MVSNet and RMVSNet
model_loss = cas_mvsnet_loss    # CascadeRMVSNet
# optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
optimizer = optim.RMSprop([{'params': model.parameters(), 'initial_lr': args.lr}],
                          lr=args.lr, alpha=0.9, weight_decay=args.wd)

# load parameters
start_epoch = 1
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(cur_log_dir) if fn.endswith(".ckpt") and len(fn.split("_")) == 3]
    # print(saved_models)
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[1].split(".")[0]))
    # use the latest checkpoint file
    # print(saved_models)
    loadckpt = os.path.join(cur_log_dir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = int(saved_models[-1].split("_")[1].split(".")[0]) + 1
    # print(saved_models)
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))

        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, lr_scheduler, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            print(
                'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}, train_result = {}'.format(
                    epoch_idx, args.epochs, batch_idx, len(TrainImgLoader), loss,
                    time.time() - start_time, scalar_outputs))
            del scalar_outputs, image_outputs

        # testing
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)

            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}, {}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time, scalar_outputs))

            del scalar_outputs, image_outputs
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())
        abs_depth_error = avg_test_scalars.mean()["abs_depth_acc"]

        train_record = open(cur_log_dir + '/train_record.txt', "a+")
        train_record.write(str(epoch_idx) + ' ' + str(avg_test_scalars.mean()) + '\n')
        train_record.close()

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(cur_log_dir, epoch_idx, abs_depth_error))
        # gc.collect()alars:", avg_test_scalars.mean())
        # gc.collect()


def test():
    # create output folder
    output_folder = os.path.join(testpath, 'height_result')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    avg_test_scalars = DictAverageMeter()

    total_time = 0
    for batch_idx, sample in enumerate(TestImgLoader):

        bview = sample['out_view'][0]
        bname = sample['out_name'][0]

        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        scalar_outputs = {k: float("{0:.6f}".format(v)) for k, v in scalar_outputs.items()}
        total_time += time.time() - start_time
        print("Iter {}/{}, {}, time = {:3f}, test results = {}".format(batch_idx, len(TestImgLoader),
                                                                       bname, time.time() - start_time, scalar_outputs))

        # save results
        depth_est = np.float32(np.squeeze(tensor2numpy(image_outputs["depth_est"])))
        prob = np.float32(np.squeeze(tensor2numpy(image_outputs["photometric_confidence"])))

        depth_gt = sample['depth']['stage3']
        mask = sample['mask']['stage3']

        depth_gt = np.float32(np.squeeze(tensor2numpy(depth_gt)))
        mask = (np.squeeze(tensor2numpy(mask))).astype(np.int)

        depth_gt[mask < 0.5] = -999.0

        plt.imshow(depth_est)
        plt.show()

        del scalar_outputs, image_outputs

    print("final, time = {:3f}, test results = {}".format(total_time, avg_test_scalars.mean()))


def train_sample(sample, lr_scheduler, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model(sample_cuda["imgs"], sample_cuda["cam_para"], sample_cuda["depth_values"])
    depth_est = outputs["stage3"]["depth"]

    loss, depth_loss = model_loss(outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e])

    loss.backward()
    optimizer.step()

    lr_scheduler.step()

    scalar_outputs = {"loss": loss, "depth_loss": depth_loss}
    image_outputs = {"depth_est": depth_est, "depth_gt": depth_gt,
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"]}

    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, 250.0)
        scalar_outputs["thres1.0m_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 1.0)
        scalar_outputs["thres2.5m_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2.5)
        scalar_outputs["thres7.5m_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 7.5)


    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    outputs = model(sample_cuda["imgs"], sample_cuda["cam_para"], sample_cuda["depth_values"])
    depth_est = outputs["stage3"]["depth"]
    photometric_confidence = outputs["stage3"]["photometric_confidence"]

    loss, depth_loss = model_loss(outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e])

    scalar_outputs = {"loss": loss, "depth_loss": depth_loss}
    image_outputs = {"depth_est": depth_est,
                     "photometric_confidence": photometric_confidence,
                     "depth_gt": sample["depth"]["stage1"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]["stage1"]}

    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask

    scalar_outputs["abs_depth_acc"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, 250.0)
    scalar_outputs["1.0m_acc"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 1.0)  #0.6
    scalar_outputs["2.5m_acc"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2.5)
    scalar_outputs["7.5m_acc"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 7.5)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()

