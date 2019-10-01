from segtester.types import Scene, Dataset
import torch
from tqdm import tqdm
from segtester import logger
from segtester.algs.mv3d import util
import os

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

        projection = ProjectionHelper(intrinsic, opt.depth_min, opt.depth_max, proj_image_dims, grid_dims,
                                      opt.voxel_size)

    def __call__(self, base_result_path, dataset_conf, *_, **__):
        dataset: Dataset = dataset_conf.get_dataset()

        with torch.no_grad():
            for scene in tqdm(dataset.scenes, desc="scene"):
                try:
                    pass
                except Exception as e:
                    logger.error(f"Exception when running ME on {dataset_conf.id}:{scene.id}. "
                                 f"Skipping scene and moving on...")
                    logger.error(str(e))

    def init_for_scenes(self, scene: Scene):

        # Camera init
        input_image_dims = [328, 256]
        proj_image_dims = [41, 32]
        intrinsic = torch.from_numpy(scene.get_intrinsic_depth()[:3, :3])
        intrinsic_image_width, intrinsic_image_height = scene.get_depth_size() # Todo assert that is correct should be 640, 480
        assert intrinsic_image_width == 640 #todo remove me im only for testing
        intrinsic = util.adjust_intrinsic(intrinsic, [intrinsic_image_width, intrinsic_image_height], proj_image_dims)
        intrinsic = intrinsic.to(self.device)

    def create_model(self):
        # create model
        num_classes = opt.num_classes
        model2d_fixed, model2d_trainable, model2d_classifier = create_enet_for_3d(ENET_TYPES[opt.model2d_type],
                                                                                  opt.model2d_orig_path, num_classes)
        model2dt_path = opt.model_path.replace('model.pth', 'model2d.pth')
        fixedname = os.path.basename(opt.model_path).split('model.pth')[0] + 'model2dfixed.pth'
        model2dfixed_path = os.path.join(os.path.dirname(opt.model_path), fixedname)
        model2d_fixed.load_state_dict(torch.load(model2dfixed_path))
        model2d_trainable.load_state_dict(torch.load(model2dt_path))
        # if opt.test_2d_model:
        #    model2dc_path = opt.model_path.replace('model.pth', 'model2dc.pth')
        #    model2d_classifier.load_state_dict(torch.load(model2dc_path))
        model = Model2d3d(num_classes, num_images, intrinsic, proj_image_dims, grid_dims, opt.depth_min, opt.depth_max,
                          opt.voxel_size)
        model.load_state_dict(torch.load(opt.model_path))

        # move to gpu
        model = model.cuda()
        model.eval()
        model2d_fixed = model2d_fixed.cuda()
        model2d_fixed.eval()
        model2d_trainable = model2d_trainable.cuda()
        model2d_trainable.eval()
        # model2d_classifier = model2d_classifier.cuda()
        # model2d_classifier.eval()

