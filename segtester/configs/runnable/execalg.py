import segtester.configs.base as BCNF
from segtester.configs.dataset import DATASET_MAP
from segtester.configs.alg import ALG_MAP

from typing import Callable


class ExecuteAlgConfig(BCNF.ConfigParser):

    # noinspection PyTypeChecker
    def __init__(self):
        super().__init__()
        self.id: str = BCNF.OptionalMember()
        self.alg: Callable = BCNF.RequiredMember(BCNF.MappableMember(ALG_MAP))
        self.dataset = BCNF.RequiredMember(BCNF.MappableMember(DATASET_MAP))

    def __call__(self, base_result_path, *args, **kwargs):
        self.alg(base_result_path=base_result_path, dataset_conf=self.dataset, *args, **kwargs)

    @staticmethod
    def register_type(out_dict):
        out_dict['ExecuteAlg'] = ExecuteAlgConfig
