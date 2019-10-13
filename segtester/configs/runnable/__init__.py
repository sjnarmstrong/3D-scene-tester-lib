from segtester.configs.runnable.execalg import ExecuteAlgConfig
from segtester.configs.runnable.viewdataset import ViewDatasetConfig
from segtester.configs.runnable.visualisepredictions import VisualisePredictionsConfig

from segtester.configs.assessments.odometry import OdometryConfig
from segtester.configs.assessments.seg3d import Segmentation3d
from segtester.configs.assessments.seg2d_reproj import Segmentation2dReproj
from segtester.configs.assessments.summarizeres import SummarizeRes

RUNNABLE_MAP = {}
ExecuteAlgConfig.register_type(RUNNABLE_MAP)
ViewDatasetConfig.register_type(RUNNABLE_MAP)
VisualisePredictionsConfig.register_type(RUNNABLE_MAP)

OdometryConfig.register_type(RUNNABLE_MAP)
Segmentation3d.register_type(RUNNABLE_MAP)
Segmentation2dReproj.register_type(RUNNABLE_MAP)
SummarizeRes.register_type(RUNNABLE_MAP)
