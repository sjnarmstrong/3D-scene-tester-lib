from segtester.types import Scene, Dataset
import open3d as o3d  # https://github.com/pytorch/pytorch/issues/19739 Thanks
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from segtester import logger
from PIL import Image
from segtester.algs.mv3d import util
import os
import math
from .projection import ProjectionHelper
from .enet import create_enet_for_3d
from .model import Model2d3d

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
        intrinsic = torch.from_numpy(scene.get_intrinsic_depth()[:3, :3])
        intrinsic_image_height, intrinsic_image_width = scene.get_depth_size() # Todo assert that is correct should be 640, 480
        intrinsic = util.adjust_intrinsic(intrinsic, [intrinsic_image_width, intrinsic_image_height],
                                          self.proj_image_dims)
        intrinsic = intrinsic.to(self.device)

        projection = ProjectionHelper(intrinsic, self.conf.depth_min, self.conf.depth_max, self.proj_image_dims,
                                      self.grid_dims, self.conf.voxel_size)

        self.model.intrinsic = intrinsic

        return projection

    def fetch_and_scale_images(self, scene: Scene):
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        depth_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST),
            transforms.CenterCrop([new_image_dims[1], new_image_dims[0]]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x/scene.get_depth_scale())
        ])

        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        colour_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST),
            transforms.CenterCrop([new_image_dims[1], new_image_dims[0]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.color_mean, std=self.color_std)
        ])
        for c_img, colour_img, location in
        color_image = misc.imread(color_file)
        depth_image = misc.imread(depth_file)
        pose = load_pose(pose_file)
        # preprocess
        depth_image = resize_crop_image(depth_image, depth_image_dims)
        color_image = resize_crop_image(color_image, color_image_dims)
        depth_image = depth_image.astype(np.float32) / 1000.0
        color_image = np.transpose(color_image, [2, 0, 1])  # move feature to front
        color_image = normalize(torch.Tensor(color_image.astype(np.float32) / 255.0))


    def __call__(self, base_result_path, dataset_conf, *_, **__):
        dataset: Dataset = dataset_conf.get_dataset()

        with torch.no_grad():
            for scene in tqdm(dataset.scenes, desc="scene"):
                try:

                    projection = self.init_for_scenes(scene)

                    scene_occ, occ_start = scene.get_tensor_occ(voxel_size=self.conf.voxel_size,
                                                                padding_x=self.grid_padX,
                                                                padding_y=self.grid_padY,
                                                                device=self.device)

                    world_to_grids = scene.get_world_to_grids(scene_occ.shape, occ_start,
                                                              self.conf.voxel_size, self.device)
                    image_viewpoint_grid = scene.get_image_viewpoints_grid(
                        scene_occ.shape, occ_start, self.conf.voxel_size, self.conf.process_nth_frame, self.device)

                    scene_occ = scene_occ.permute(2, 0, 1)
                    scene_occ = torch.stack([scene_occ, scene_occ])

                    if scene_occ.shape[1] > self.column_height:
                        scene_occ = scene_occ[:, :self.column_height, :, :]
                    scene_occ_sz = scene_occ.shape[1:]

                    # init a few things for prediction
                    depth_image = torch.empty(self.num_images, self.proj_image_dims[1], self.proj_image_dims[0],
                                              dtype=torch.float, device=self.device)
                    color_image = torch.empty(self.num_images, 3, self.input_image_dims[1], self.input_image_dims[0],
                                              dtype=torch.float, device=self.device)
                    world_to_grid = torch.empty(self.num_images, 4, 4,
                                                dtype=torch.float, device=self.device)
                    pose = torch.empty(self.num_images, 4, 4,
                                       dtype=torch.float, device=self.device)
                    output_probs = torch.zeros(self.conf.num_classes, scene_occ_sz[0], scene_occ_sz[1], scene_occ_sz[2],
                                               dtype=torch.float, device=self.device)
                    # make sure nonsingular
                    for k in range(self.num_images):
                        pose[k] = torch.eye(4)
                        world_to_grid[k] = torch.eye(4)

                    # go thru all columns
                    # note, voxels are split into grid of 31x31x62. In order to ensure that convolution is proper,
                    # a total of 31//2 zero voxels are used for padding
                    for y in range(self.grid_padY, scene_occ_sz[1] - self.grid_padY):
                        for x in range(self.grid_padX, scene_occ_sz[2] - self.grid_padX):
                            input_occ = scene_occ[:, :,
                                                  y-self.grid_padY:y+self.grid_padY+1,
                                                  x-self.grid_padX:x+self.grid_padX+1].unsqueeze(0)
                            cur_frame_ids = image_viewpoint_grid[y][x]
                            if len(cur_frame_ids) < self.num_images or \
                                    torch.sum(input_occ[0, 0, :, self.grid_padY, self.grid_padX]) == 0:
                                continue


                            # get first few images that cover location
                            for k in range(self.num_images):
                                # depth image is 1/8 the dimentions of colour
                                c_img, d_img, p = scene.get_image_info_index(k)
                                color_image[k] = torch.from_numpy(c_img)
                                depth_image[k] = torch.from_numpy(d_img)
                                pose[k] = torch.from_numpy(p)
                                world_to_grid[k] = torch.from_numpy(world_to_grids[y, x])
                            proj_mapping = [projection.compute_projection(d, c, t) for d, c, t in
                                            zip(depth_image, pose, world_to_grid)]
                            if None in proj_mapping:  # invalid sample
                                continue

                            proj_mapping = zip(*proj_mapping)
                            # stack the orelation between 3D coords wrt grid and the 2D pixel values
                            proj_ind_3d = torch.stack(proj_mapping[0])
                            proj_ind_2d = torch.stack(proj_mapping[1])

                            imageft_fixed = self.model2d_fixed(torch.autograd.Variable(color_image))
                            imageft = self.model2d_trainable(imageft_fixed)

                            out = self.model(torch.autograd.Variable(input_occ), imageft,
                                             torch.autograd.Variable(proj_ind_3d),
                                             torch.autograd.Variable(proj_ind_2d), grid_dims)
                            output = out.data[0].permute(1, 0)
                            output_probs[:, :, y, x] = output.cpu()[:, :scene_occ_sz[0]]



                except Exception as e:
                    logger.error(f"Exception when running ME on {dataset_conf.id}:{scene.id}. "
                                 f"Skipping scene and moving on...")
                    logger.error(str(e))
