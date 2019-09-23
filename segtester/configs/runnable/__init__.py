from segtester.configs.runnable.execalg import ExecuteAlgConfig
from segtester.configs.runnable.viewdataset import ViewDatasetConfig

RUNNABLE_MAP = {}
ExecuteAlgConfig.register_type(RUNNABLE_MAP)
ViewDatasetConfig.register_type(RUNNABLE_MAP)
