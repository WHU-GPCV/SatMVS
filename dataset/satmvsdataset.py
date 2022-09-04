
from torch.utils.data import Dataset
from dataset.data_io import *
from dataset.preprocess import *
from dataset.gen_list import *
import copy


class MVSDataset(Dataset):
    def __init__(self, data_folder, mode, view_num, ref_view=2, use_qc=False):
        super(MVSDataset, self).__init__()
        self.data_folder = data_folder
        self.mode = mode
        self.view_num = view_num
        self.ref_view = ref_view
        self.use_qc = use_qc
        assert self.mode in ["train", "val", "test", "pred"]
        self.sample_list = self.build_list()

        self.sample_num = len(self.sample_list)

    def build_list(self):
        # Prepare all training samples
        if self.mode == "pred":
            sample_list = gen_all_mvs_list_rpc(self.data_folder, self.view_num)
        elif self.ref_view < 0:
            sample_list = gen_all_mvs_list_rpc(self.data_folder, self.view_num)
        else:
            sample_list = gen_ref_list_rpc(self.data_folder, self.view_num, self.ref_view)

        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def get_sample(self, idx):
        data = self.sample_list[idx]
        ###### read input data ######

        centered_images = []
        rpc_paramters = []

        _, depth_max, depth_min = load_rpc_as_array(data[2 * self.ref_view + 1])

        # height map, but we name it as depth image to be consistent with that in homography warping
        depth_image = load_pfm(data[2 * self.view_num]).astype(np.float32)

        for view in range(self.view_num):
            # Images
            if self.mode == "train":
                image = image_augment(read_img(data[2 * view]))
            else:
                image = read_img(data[2 * view])
            image = np.asarray(image)

            # Cameras, We use d to denote the search direction
            rpc, _, _ = load_rpc_as_array(data[2 * view + 1])

            rpc_paramters.append(rpc)
            centered_images.append(center_image(image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        rpc_paramters = np.array(rpc_paramters)

        # Depth
        # print(new_ndepths)
        depth_values = np.array([depth_min, depth_max], dtype=np.float32)

        mask = np.float32((depth_image >= depth_min) * 1.0) * np.float32((depth_image <= depth_max) * 1.0)

        h, w = depth_image.shape
        depth_ms = {
            "stage1": cv2.resize(depth_image, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_image, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_image
        }
        mask_ms = {
            "stage1": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": mask
        }

        stage2_rpc = rpc_paramters.copy()
        stage2_rpc[:, 0] = stage2_rpc[:, 0] / 2
        stage2_rpc[:, 1] = stage2_rpc[:, 1] / 2
        stage2_rpc[:, 5] = stage2_rpc[:, 5] / 2
        stage2_rpc[:, 6] = stage2_rpc[:, 6] / 2

        stage3_rpc = rpc_paramters.copy()
        stage3_rpc[:, 0] = stage3_rpc[:, 0] / 4
        stage3_rpc[:, 1] = stage3_rpc[:, 1] / 4
        stage3_rpc[:, 5] = stage3_rpc[:, 5] / 4
        stage3_rpc[:, 6] = stage3_rpc[:, 6] / 4

        rpc_paramters_ms = {
            "stage1": stage3_rpc,
            "stage2": stage2_rpc,
            "stage3": rpc_paramters
        }

        out_view = data[0].split("/")[-2]
        out_name = os.path.splitext(data[0].split("/")[-1])[0]

        return {"imgs": centered_images,
                "cam_para": rpc_paramters_ms,
                "depth": depth_ms,
                "mask": mask_ms,
                "depth_values": depth_values,
                "out_view": out_view,
                "out_name": out_name
                }

    def get_pred_sample(self, idx):
        data = self.sample_list[idx]
        ###### read input data ######

        centered_images = []
        rpc_paramters = []

        _, depth_max, depth_min = load_rpc_as_array(data[1])
        for view in range(self.view_num):
            # Images
            image = read_img(data[2 * view])
            image = np.array(image)

            # Cameras
            rpc, _, _ = load_rpc_as_array(data[2 * view + 1])

            rpc_paramters.append(rpc)
            centered_images.append(center_image(image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        rpc_paramters = np.stack(rpc_paramters)

        # Depth
        depth_values = np.array([depth_min, depth_max], dtype=np.float32)

        stage2_rpc = rpc_paramters.copy()
        stage2_rpc[:, 0] = stage2_rpc[:, 0] / 2
        stage2_rpc[:, 1] = stage2_rpc[:, 1] / 2
        stage2_rpc[:, 5] = stage2_rpc[:, 5] / 2
        stage2_rpc[:, 6] = stage2_rpc[:, 6] / 2

        stage3_rpc = rpc_paramters.copy()
        stage3_rpc[:, 0] = stage3_rpc[:, 0] / 4
        stage3_rpc[:, 1] = stage3_rpc[:, 1] / 4
        stage3_rpc[:, 5] = stage3_rpc[:, 5] / 4
        stage3_rpc[:, 6] = stage3_rpc[:, 6] / 4

        rpc_paramters_ms = {
            "stage1": stage3_rpc,
            "stage2": stage2_rpc,
            "stage3": rpc_paramters
        }

        out_view = data[0].split("/")[-2]
        out_name = os.path.splitext(data[0].split("/")[-1])[0]

        return {"imgs": centered_images,
                "cam_para": rpc_paramters_ms,
                "depth_values": depth_values,
                "out_view": out_view,
                "out_name": out_name
                }

    def get_sample_qc(self, idx):
        data = self.sample_list[idx]

        centered_images = []
        rpc_paramters = []

        # depth
        depth_image = load_pfm(data[2 * self.view_num]).astype(np.float32)
        rpc = load_rpc_as_qc_tensor(data[2 * self.ref_view + 1])
        depth_max = rpc["height_off"] + rpc["height_scale"]
        depth_min = rpc["height_off"] - rpc["height_scale"]

        for view in range(self.view_num):
            # Images
            if self.mode == "train":
                image = image_augment(read_img(data[2 * view]))
            else:
                image = read_img(data[2 * view])
            image = np.array(image)

            # Cameras
            rpc = load_rpc_as_qc_tensor(data[2 * view + 1])

            rpc_paramters.append(rpc)
            centered_images.append(center_image(image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        depth_values = np.array([depth_min, depth_max], dtype=np.float32)

        mask = np.float32((depth_image >= depth_min) * 1.0) * np.float32((depth_image <= depth_max) * 1.0)

        h, w = depth_image.shape
        depth_ms = {
            "stage1": cv2.resize(depth_image, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_image, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_image
        }
        mask_ms = {
            "stage1": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": mask
        }

        stage2_rpc = copy.deepcopy(rpc_paramters)
        for v in range(len(rpc_paramters)):
            stage2_rpc[v]["line_off"] /= 2
            stage2_rpc[v]["samp_off"] /= 2
            stage2_rpc[v]["line_scale"] /= 2
            stage2_rpc[v]["samp_scale"] /= 2

        stage3_rpc = copy.deepcopy(rpc_paramters)
        for v in range(len(rpc_paramters)):
            stage3_rpc[v]["line_off"] /= 4
            stage3_rpc[v]["samp_off"] /= 4
            stage3_rpc[v]["line_scale"] /= 4
            stage3_rpc[v]["samp_scale"] /= 4

        rpc_paramters_ms = {
            "stage1": stage3_rpc,
            "stage2": stage2_rpc,
            "stage3": rpc_paramters
        }
        # print(type(rpc_paramters_ms["stage1"]))

        out_view = data[0].split("/")[-2]
        out_name = os.path.splitext(data[0].split("/")[-1])[0]

        return {"imgs": centered_images,
                "cam_para": rpc_paramters_ms,
                "depth": depth_ms,
                "mask": mask_ms,
                "depth_values": depth_values,
                "out_view": out_view,
                "out_name": out_name
                }

    def get_pred_sample_qc(self, idx):
        data = self.sample_list[idx]

        centered_images = []
        rpc_paramters = []

        rpc = load_rpc_as_qc_tensor(data[2 * self.ref_view + 1])
        depth_max = rpc["height_off"] + rpc["height_scale"]
        depth_min = rpc["height_off"] - rpc["height_scale"]

        for view in range(self.view_num):
            # Images
            image = read_img(data[2 * view])
            image = np.array(image)

            # Cameras
            rpc = load_rpc_as_qc_tensor(data[2 * view + 1])
            rpc_paramters.append(rpc)

            centered_images.append(center_image(image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        depth_values = np.array([depth_min, depth_max], dtype=np.float32)

        stage2_rpc = copy.deepcopy(rpc_paramters)
        for v in range(len(rpc_paramters)):
            stage2_rpc[v]["line_off"] = stage2_rpc[v]["line_off"] / 2
            stage2_rpc[v]["samp_off"] = stage2_rpc[v]["samp_off"] / 2
            stage2_rpc[v]["line_scale"] = stage2_rpc[v]["line_scale"] / 2
            stage2_rpc[v]["samp_scale"] = stage2_rpc[v]["samp_scale"] / 2

        stage3_rpc = copy.deepcopy(rpc_paramters)
        for v in range(len(rpc_paramters)):
            stage3_rpc[v]["line_off"] = stage3_rpc[v]["line_off"] / 4
            stage3_rpc[v]["samp_off"] = stage3_rpc[v]["samp_off"] / 4
            stage3_rpc[v]["line_scale"] = stage3_rpc[v]["line_scale"] / 4
            stage3_rpc[v]["samp_scale"] = stage3_rpc[v]["samp_scale"] / 4

        rpc_paramters_ms = {
            "stage1": stage3_rpc,
            "stage2": stage2_rpc,
            "stage3": rpc_paramters
        }

        out_view = data[0].split("/")[-2]
        out_name = os.path.splitext(data[0].split("/")[-1])[0]

        return {"imgs": centered_images,
                "cam_para": rpc_paramters_ms,
                "depth_values": depth_values,
                "out_view": out_view,
                "out_name": out_name
                }

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        if self.mode != "pred":
            if self.use_qc:
                return self.get_sample_qc(idx)
            else:
                return self.get_sample(idx)
        else:
            if self.use_qc:
                return self.get_pred_sample_qc(idx)
            else:
                return self.get_pred_sample(idx)
