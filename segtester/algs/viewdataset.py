from typing import TYPE_CHECKING
import open3d as o3d
from segtester.types import Dataset
if TYPE_CHECKING:
    from segtester.configs.runnable.viewdataset import ViewDatasetConfig


class ViewDataset:
    def __init__(self, conf: 'ViewDatasetConfig'):
        self.conf = conf

    def __call__(self, *args, **kwargs):
        dataset: Dataset = self.conf.dataset.get_dataset()
        for scene in dataset.scenes:
            pcd = scene.get_pcd()
            if self.conf.view_pointcloud:
                o3d.visualization.draw_geometries([pcd])
