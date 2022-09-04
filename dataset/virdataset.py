from torch.utils.data import Dataset
from dataset.data_io import *
from dataset.preprocess import *
from dataset.gen_list import *
from imageio import imread


class MVSDataset(Dataset):
    def __init__(self, data_folder, mode, view_num, ref_view=2, use_qc=False):
        super(MVSDataset, self).__init__()
        self.data_folder = data_folder
        self.mode = mode
        self.view_num = view_num
        self.ref_view = ref_view
        assert self.mode in ["train", "val", "test", "pred"]
        self.sample_list = self.build_list()
        self.sample_num = len(self.sample_list)

    def build_list(self):
        # Prepare all training samples

        if self.mode == "pred":
            sample_list = gen_all_mvs_list_cam(self.data_folder, self.view_num)
        elif self.ref_view < 0:
            sample_list = gen_all_mvs_list_cam(self.data_folder, self.view_num)
        else:
            sample_list = gen_ref_list_cam(self.data_folder, self.view_num, self.ref_view)

        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def read_depth(self, filename):
        # read pfm depth file
        depth_image = np.float32(load_pfm(filename))

        return np.array(depth_image)

    def get_sample(self, idx):
        data = self.sample_list[idx]
        ###### read input data ######

        centered_images = []
        proj_matrices = []
        depth_min = None
        depth_max = None

        # depth
        depth_image = self.read_depth(os.path.join(data[2 * self.view_num]))

        for view in range(self.view_num):
            # Images
            if self.mode == "train":
                image = image_augment(read_img(data[2 * view]))
            else:
                image = read_img(data[2 * view])
            image = np.array(image)

            # Cameras
            cam = read_vir_camera_in_nn(data[2 * view + 1])

            if view == 0:
                depth_min = cam[1][3][0]
                depth_max = cam[1][3][3]

            extrinsics = cam[0, :, :]
            intrinsics = cam[1, 0:3, 0:3]
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])

            proj_matrices.append(proj_mat)
            centered_images.append(center_image(image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

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

        # ms proj_mats
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, :2, :] = proj_matrices[:, :2, :] / 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, :2, :] = proj_matrices[:, :2, :] / 4

        proj_matrices_ms = {
            "stage1": stage3_pjmats,
            "stage2": stage2_pjmats,
            "stage3": proj_matrices
        }

        out_view = data[0].split("/")[-2]
        out_name = os.path.splitext(data[0].split("/")[-1])[0]

        return {"imgs": centered_images,
                "cam_para": proj_matrices_ms,
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
        proj_matrices = []
        depth_min = None
        depth_max = None

        for view in range(self.view_num):
            # Images
            image = read_img(data[2 * view])
            image = np.array(image)

            # Cameras
            cam = read_vir_camera_in_nn(data[2 * view + 1])

            if view == 0:
                depth_min = cam[1][3][0]
                depth_max = cam[1][3][3]

            extrinsics = cam[0, :, :]
            intrinsics = cam[1, 0:3, 0:3]
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])

            proj_matrices.append(proj_mat)
            centered_images.append(center_image(image))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        # Depth
        # print(new_ndepths)
        depth_values = np.array([depth_min, depth_max], dtype=np.float32)

        # ms proj_mats
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, :2, :] = proj_matrices[:, :2, :] / 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, :2, :] = proj_matrices[:, :2, :] / 4

        proj_matrices_ms = {
            "stage1": stage3_pjmats,
            "stage2": stage2_pjmats,
            "stage3": proj_matrices
        }

        out_view = data[0].split("/")[-2]
        out_name = os.path.splitext(data[0].split("/")[-1])[0]

        return {"imgs": centered_images,
                "cam_para": proj_matrices_ms,
                "depth_values": depth_values,
                "out_view": out_view,
                "out_name": out_name
                }

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        if self.mode != "pred":
            return self.get_sample(idx)
        else:
            return self.get_pred_sample(idx)
