from segtester.configs.runnable.execalg import ExecuteAlgConfig
from segtester.configs.runnable.viewdataset import ViewDatasetConfig
from segtester.configs.runnable.visualisepredictions import VisualisePredictionsConfig

RUNNABLE_MAP = {}
ExecuteAlgConfig.register_type(RUNNABLE_MAP)
ViewDatasetConfig.register_type(RUNNABLE_MAP)
VisualisePredictionsConfig.register_type(RUNNABLE_MAP)
