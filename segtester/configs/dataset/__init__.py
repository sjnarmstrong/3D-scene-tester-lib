from segtester.configs.dataset.kitti import KITTIConfig
from segtester.configs.dataset.scannet import SCANNETConfig

DATASET_MAP = {}
KITTIConfig.register_type(DATASET_MAP)
SCANNETConfig.register_type(DATASET_MAP)
