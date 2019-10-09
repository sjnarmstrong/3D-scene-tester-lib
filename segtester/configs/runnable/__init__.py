from segtester.configs.runnable.execalg import ExecuteAlgConfig
from segtester.configs.runnable.viewdataset import ViewDatasetConfig
from segtester.configs.runnable.visualisepredictions import VisualisePredictionsConfig

from segtester.configs.assessments.odometry import OdometryConfig

RUNNABLE_MAP = {}
ExecuteAlgConfig.register_type(RUNNABLE_MAP)
ViewDatasetConfig.register_type(RUNNABLE_MAP)
VisualisePredictionsConfig.register_type(RUNNABLE_MAP)

OdometryConfig.register_type(RUNNABLE_MAP)
