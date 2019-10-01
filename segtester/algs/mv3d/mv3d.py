from segtester.types import Scene, Dataset
import torch
from tqdm import tqdm
from segtester import logger
from segtester.algs.mv3d import util
import os
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
        self.model = self.create_model()

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
        model = model.cuda()
        model.eval()
        model2d_fixed = model2d_fixed.cuda()
        model2d_fixed.eval()
        model2d_trainable = model2d_trainable.cuda()
        model2d_trainable.eval()
        # model2d_classifier = model2d_classifier.cuda()
        # model2d_classifier.eval()
        return model

    def init_for_scenes(self, scene: Scene):

        # Camera init
        intrinsic = torch.from_numpy(scene.get_intrinsic_depth()[:3, :3])
        intrinsic_image_width, intrinsic_image_height = scene.get_depth_size() # Todo assert that is correct should be 640, 480
        assert intrinsic_image_width == 640 #todo remove me im only for testing
        intrinsic = util.adjust_intrinsic(intrinsic, [intrinsic_image_width, intrinsic_image_height],
                                          self.proj_image_dims)
        intrinsic = intrinsic.to(self.device)

        projection = ProjectionHelper(intrinsic, self.conf.depth_min, self.conf.depth_max, self.proj_image_dims,
                                      self.grid_dims, self.conf.voxel_size)

        self.model.intrinsic = intrinsic

    def __call__(self, base_result_path, dataset_conf, *_, **__):
        dataset: Dataset = dataset_conf.get_dataset()

        with torch.no_grad():
            for scene in tqdm(dataset.scenes, desc="scene"):
                try:

                    scene.get


                except Exception as e:
                    logger.error(f"Exception when running ME on {dataset_conf.id}:{scene.id}. "
                                 f"Skipping scene and moving on...")
                    logger.error(str(e))
