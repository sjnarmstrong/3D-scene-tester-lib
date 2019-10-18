from segtester.types import Scene, Dataset
import open3d as o3d  # https://github.com/pytorch/pytorch/issues/19739 Thanks
from tqdm import tqdm
from segtester import logger
from PIL import Image
from segtester.algs.mv3d import util
import os
import math
from .projection import ProjectionHelper
from .enet import create_enet_for_3d
from .model import Model2d3d
import numpy as np
from segtester.util.create_image_viewpoint_grid_new import create_image_viewpoints_grid, visualise_viewpoints
from segtester.util.from_occ_to_pcd import create_pcd_from_occ
from torch.nn import functional as F

import torchvision.transforms as transforms
import torch
from math import ceil
import shutil
import sys

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from segtester.configs.alg.MV3DConfig import Execute3DMVConfig


ENET_TYPES = {'scannet': (41, [0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129])}


class Execute3DMV:
    def __init__(self, conf: 'Execute3DMVConfig'):
        self.conf = conf

        if conf.cpu_only:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Configure grid
        self.grid_dims = [self.conf.grid_dimX, self.conf.grid_dimY, self.conf.grid_dimZ]
        self.column_height = self.conf.grid_dimZ
        self.num_images = self.conf.num_nearest_images
        self.grid_padX = self.conf.grid_dimX // 2
        self.grid_padY = self.conf.grid_dimY // 2
        self.color_mean = ENET_TYPES[self.conf.model2d_type][1]
        self.color_std = ENET_TYPES[self.conf.model2d_type][2]
        self.input_image_dims = [328, 256]
        self.proj_image_dims = [41, 32]
        self.model, self.model2d_fixed, self.model2d_trainable = self.create_model()

    def create_model(self):
        # Create model
        num_classes = self.conf.num_classes
        model2d_fixed, model2d_trainable, model2d_classifier = create_enet_for_3d(
            ENET_TYPES[self.conf.model2d_type], self.conf.model2d_orig_path, num_classes
        )
        model2dt_path = self.conf.model_path.replace('model.pth', 'model2d.pth')
        fixedname = os.path.basename(self.conf.model_path).split('model.pth')[0] + 'model2dfixed.pth'
        model2dfixed_path = os.path.join(os.path.dirname(self.conf.model_path), fixedname)
        model2d_fixed.load_state_dict(torch.load(model2dfixed_path))
        model2d_trainable.load_state_dict(torch.load(model2dt_path))
        # if opt.test_2d_model:
        #    model2dc_path = opt.model_path.replace('model.pth', 'model2dc.pth')
        #    model2d_classifier.load_state_dict(torch.load(model2dc_path))
        model = Model2d3d(num_classes, self.num_images, None, self.proj_image_dims, self.grid_dims,
                          self.conf.depth_min, self.conf.depth_max, self.conf.voxel_size)
        model.load_state_dict(torch.load(self.conf.model_path))

        # move to gpu
        model = model.to(self.device)
        model.eval()
        model2d_fixed = model2d_fixed.to(self.device)
        model2d_fixed.eval()
        model2d_trainable = model2d_trainable.to(self.device)
        model2d_trainable.eval()
        # model2d_classifier = model2d_classifier.cuda()
        # model2d_classifier.eval()
        return model, model2d_fixed, model2d_trainable

    def init_for_scenes(self, scene: Scene):

        # Camera init
        intrinsic = torch.from_numpy(scene.get_intrinsic_depth().copy()[:3, :3])
        intrinsic_image_height, intrinsic_image_width = scene.get_depth_size() # Todo assert that is correct should be 640, 480
        intrinsic = util.adjust_intrinsic(intrinsic, [intrinsic_image_width, intrinsic_image_height],
                                          self.proj_image_dims)
        intrinsic = intrinsic.to(self.device)

        projection = ProjectionHelper(intrinsic, self.conf.depth_min, self.conf.depth_max, self.proj_image_dims,
                                      self.grid_dims, self.conf.voxel_size)

        self.model.intrinsic = intrinsic

        return projection

    def fetch_and_scale_images(self, scene: Scene):
        image_dims_d = scene.get_depth_size()
        resize_width = int(math.floor(self.proj_image_dims[1] * float(image_dims_d[0]) / float(image_dims_d[1])))
        depth_transforms = transforms.Compose([
            transforms.Lambda(lambda x: x.astype(np.float32)/scene.get_depth_scale()),
            transforms.ToPILImage(),
            transforms.Resize([self.proj_image_dims[1], resize_width], interpolation=Image.NEAREST),
            transforms.CenterCrop([self.proj_image_dims[1], self.proj_image_dims[0]]),
            transforms.ToTensor(),
        ])

        image_dims = scene.get_rgb_size()
        resize_width = int(math.floor(self.input_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        colour_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([self.input_image_dims[1], resize_width], interpolation=Image.ANTIALIAS),
            transforms.CenterCrop([self.input_image_dims[1], self.input_image_dims[0]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.color_mean, std=self.color_std)
        ])
        to_tensor = transforms.Compose([
            transforms.Lambda(lambda x: x.astype(np.float32)/scene.get_depth_scale()),
            transforms.ToTensor(),
        ])
        num_frames_to_process = ceil(scene.get_num_frames()/self.conf.process_nth_frame)
        depth_images = torch.empty(num_frames_to_process, self.proj_image_dims[1], self.proj_image_dims[0],
                                   dtype=torch.float)
        depth_images_hd = torch.empty(num_frames_to_process, image_dims_d[0], image_dims_d[1],
                                   dtype=torch.float)
        color_images = torch.empty(num_frames_to_process, 3, self.input_image_dims[1], self.input_image_dims[0],
                                   dtype=torch.float)
        poses = torch.empty(num_frames_to_process, 4, 4,
                            dtype=torch.float)

        j = 0
        for rgbdimage, i in tqdm(scene.get_rgbd_image_it(), desc="Pre-process images",
                                           total=scene.get_num_frames()):
            if i % self.conf.process_nth_frame != 0:
                continue
            c_img, d_img = rgbdimage.get_color_image(), rgbdimage.get_depth_image()
            c_t_w = rgbdimage.get_camera_to_world()
            depth_images[j] = depth_transforms(d_img)
            depth_images_hd[j] = to_tensor(d_img)
            color_images[j] = colour_transforms(c_img)
            poses[j] = torch.from_numpy(c_t_w)
            j+=1
        # assert j == len(depth_images), "Invalid number if images processed"
        return depth_images, depth_images_hd, color_images, poses, scene.get_intrinsic_depth()

    def __call__(self, base_result_path, dataset_conf, *_, **__):
        dataset: Dataset = dataset_conf.get_dataset()

        with torch.no_grad():
            for scene in tqdm(dataset.scenes, desc="scene"):
                try:
                    logger.info(f"Processing {self.conf.alg_name} on scene {scene.id}...")

                    save_path = self.conf.format_string_with_meta(f"{base_result_path}/{self.conf.save_path}", **{
                        "dataset_id": dataset_conf.id, "scene_id": scene.id,
                        "alg_name": self.conf.alg_name,
                    })

                    if self.conf.skip_existing and os.path.exists(f"{save_path}"):
                        logger.warn(f"When processing {dataset_conf.id}->{scene.id}->{self.conf.alg_name}, "
                                    f"found existing path {save_path}.\n Skipping this scene...")
                        continue

                    os.makedirs(f"{save_path}/torch", exist_ok=True)
                    os.makedirs(f"{save_path}/frames", exist_ok=True)

                    projection = self.init_for_scenes(scene)

                    # Prepopulate and augment images and poses:
                    #

                    scene_occ, occ_start = scene.get_tensor_occ(voxel_size=self.conf.voxel_size,
                                                                padding_x=self.grid_padX,
                                                                padding_y=self.grid_padY,
                                                                device=self.device)
                    pcd = create_pcd_from_occ(scene_occ, occ_start, self.conf.voxel_size, self.device).cpu().numpy().T
                    # pred_pcd = o3d.geometry.PointCloud()
                    # pred_pcd.points = o3d.utility.Vector3dVector(pcd)
                    # o3d.visualization.draw_geometries([pred_pcd])
                    # o3d.io.write_point_cloud(f"{save_path}/pcd.ply", pred_pcd)

                    depth_images, depth_images_hd, color_images, poses, intrinsic_depth_hd = \
                        self.fetch_and_scale_images(scene)

                    world_to_grids = scene.get_world_to_grids(scene_occ.shape, occ_start,
                                                              self.conf.voxel_size, self.grid_padX,  self.grid_padY,
                                                              self.device)
                    # image_viewpoint_grid = scene.get_image_viewpoints_grid(
                    #     scene_occ.shape, occ_start, self.conf.voxel_size, self.conf.process_nth_frame, self.device)

                    image_viewpoint_grid = create_image_viewpoints_grid(depth_images_hd, poses, scene_occ.shape, occ_start,
                                                                        self.conf.voxel_size, intrinsic_depth_hd,
                                                                        self.device)
                    del depth_images_hd
                    # image_viewpoint_grid = create_image_viewpoints_grid(self.conf.depth_min, self.conf.depth_max,
                    #                                                     depth_images[0].shape, poses, scene_occ.shape,
                    #                                                     occ_start, self.conf.voxel_size,
                    #                                                     projection.intrinsic, self.device)

                    # visualise_viewpoints(image_viewpoint_grid, scene_occ, occ_start, color_images)

                    scene_occ = scene_occ.permute(2, 0, 1)
                    scene_occ = torch.stack([scene_occ, scene_occ])

                    if scene_occ.shape[1] > self.column_height:
                        scene_occ = scene_occ[:, :self.column_height, :, :]
                    scene_occ_sz = scene_occ.shape[1:]

                    output_probs = torch.zeros(self.conf.num_classes, scene_occ_sz[0], scene_occ_sz[1], scene_occ_sz[2],
                                               dtype=torch.float, device=self.device)

                    input_occ = torch.empty(1, 2, self.grid_dims[2], self.grid_dims[1], self.grid_dims[0],
                                            dtype=torch.float,
                                            device=self.device)
                    # go thru all columns
                    # note, voxels are split into grid of 31x31x62. In order to ensure that convolution is proper,
                    # a total of 31//2 zero voxels are used for padding
                    yx_iter = ((y, x) for y in range(self.grid_padY, scene_occ_sz[1] - self.grid_padY)
                               for x in range(self.grid_padX, scene_occ_sz[2] - self.grid_padX))
                    for (y, x) in tqdm(yx_iter, desc="Processing scene",
                                       total=(scene_occ_sz[1] - 2*self.grid_padY)*(scene_occ_sz[2] - 2*self.grid_padX)):
                            input_occ.fill_(0)
                            input_occ[0, :, :scene_occ_sz[0], :, :] = scene_occ[:, :,
                                                                                y-self.grid_padY:y+self.grid_padY+1,
                                                                                x-self.grid_padX:x+self.grid_padX+1]
                            cur_frame_ids = image_viewpoint_grid[y][x]
                            if len(cur_frame_ids) < self.num_images or \
                                    torch.sum(input_occ[0, 0, :, self.grid_padY, self.grid_padX]) == 0:
                                continue


                            # get first few images that cover location
                            depth_image = depth_images[cur_frame_ids[:self.num_images]].to(self.device)
                            color_image = color_images[cur_frame_ids[:self.num_images]].to(self.device)
                            pose = poses[cur_frame_ids[:self.num_images]].to(self.device)
                            world_to_grid = torch.stack(self.num_images * [world_to_grids[y, x]])

                            proj_mapping = [projection.compute_projection(d, c, t, pcd, (x,y), c_img) for d, c, t, c_img in
                                            zip(depth_image, pose, world_to_grid, color_image)]
                            if None in proj_mapping:  # invalid sample
                                continue

                            proj_mapping = list(zip(*proj_mapping))
                            # stack the orelation between 3D coords wrt grid and the 2D pixel values
                            proj_ind_3d = torch.stack(proj_mapping[0])
                            proj_ind_2d = torch.stack(proj_mapping[1])

                            imageft_fixed = self.model2d_fixed(torch.autograd.Variable(color_image))
                            imageft = self.model2d_trainable(imageft_fixed)
                            if self.conf.save_frames:
                                for ft_img, img_nr in zip(imageft, cur_frame_ids):
                                    np.savez_compressed(f"{save_path}/frames/frame_{img_nr*self.conf.process_nth_frame}",
                                                        likelihoods=F.softmax(ft_img, dim=2).cpu().numpy())

                            out = self.model(torch.autograd.Variable(input_occ), imageft,
                                             torch.autograd.Variable(proj_ind_3d),
                                             torch.autograd.Variable(proj_ind_2d), self.grid_dims)
                            output = out.data[0].permute(1, 0)
                            output_probs[:, :, y, x] = output.cpu()[:, :scene_occ_sz[0]]

                    torch.save(occ_start, f"{save_path}/torch/occ_start.torch")
                    torch.save(scene_occ, f"{save_path}/torch/scene_occ.torch")
                    torch.save(output_probs, f"{save_path}/torch/output_probs.torch")
                    torch.save(world_to_grids, f"{save_path}/torch/world_to_grids.torch")
                    torch.save(depth_images, f"{save_path}/torch/depth_images.torch")
                    torch.save(color_images, f"{save_path}/torch/color_images.torch")
                    torch.save(poses, f"{save_path}/torch/poses.torch")

                    output_probs = output_probs.reshape(self.conf.num_classes, -1).T # check shapes
                    likelihoods = F.softmax(output_probs, dim=1).cpu().numpy()
                    masked_likelihoods = likelihoods[(scene_occ[0]*scene_occ[1]).cpu().numpy().flat]

                    pred_pcd = o3d.geometry.PointCloud()
                    pred_pcd.points = o3d.utility.Vector3dVector(pcd)
                    o3d.io.write_point_cloud(f"{save_path}/pcd.ply", pred_pcd)
                    np.savez_compressed(f"{save_path}/probs", likelihoods=masked_likelihoods)
                except KeyboardInterrupt as e:
                    try:
                        logger.error(f"Detected [ctrl+c]. Performing cleanup and then exiting...")
                        shutil.rmtree(save_path)
                    except Exception:
                        pass
                    sys.exit(0)
                except Exception as e:
                    try:
                        logger.error(f"Exception when running {self.conf.alg_name} on {dataset_conf.id}:{scene.id}. "
                                     f"Skipping scene and moving on...")
                        logger.error(str(e))
                        shutil.rmtree(save_path)
                    except Exception:
                        pass
