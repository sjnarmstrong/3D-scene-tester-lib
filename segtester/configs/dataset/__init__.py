from segtester.configs.dataset.kitti import KITTIConfig
from segtester.configs.dataset.scannet import SCANNETConfig
from segtester.configs.dataset.nyuv2 import NYUv2Config

DATASET_MAP = {}
KITTIConfig.register_type(DATASET_MAP)
SCANNETConfig.register_type(DATASET_MAP)
NYUv2Config.register_type(DATASET_MAP)
