import segtester.configs.base as BCNF
from segtester.configs.dataset import DATASET_MAP

from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from segtester.types import Dataset


class ViewDatasetConfig(BCNF.ConfigParser):

    # noinspection PyTypeChecker
    def __init__(self):
        super().__init__()
        self.id: str = BCNF.OptionalMember()
        self.view_pointcloud: Callable = BCNF.OptionalMember(default_ret=True)
        self.dataset: 'Dataset' = BCNF.RequiredMember(BCNF.MappableMember(DATASET_MAP))

    def __call__(self, *args, **kwargs):
        from segtester.algs.viewdataset import ViewDataset
        ViewDataset(self)()

    @staticmethod
    def register_type(out_dict):
        out_dict['ViewDataset'] = ViewDatasetConfig
