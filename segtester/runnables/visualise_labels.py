from typing import TYPE_CHECKING
from matplotlib import pyplot as plt
from tqdm import tqdm
import open3d as o3d
from segtester import logger


if TYPE_CHECKING:
    from segtester.configs.runnable.visualisepredictions import VisualisePredictionsConfig, ResultsConfig


class VisualisePredictions:
    def __init__(self, conf: 'VisualisePredictionsConfig'):
        self.cmap = plt.get_cmap("hsv")
        self.conf = conf

    def __call__(self, base_result_path, dataset_conf, *_, **__):
        dataset = dataset_conf.get_dataset()

        for scene in tqdm(dataset.scenes, desc="scene"):
            try:
                _, label_masks, pcd = scene.get_labeled_pcd()
                converted_labels = scene.get_converted_labels(label_masks)
                label_names = dataset.label_names[converted_labels]

                colors = self.cmap((converted_labels-2)/(dataset.max_to_label-2))[:, :3]
                colors[converted_labels == 0] = [0.35, 0.35, 0.35]
                colors[converted_labels == 1] = [0.6, 0.6, 0.6]

                pcd.colors = o3d.utility.Vector3dVector(colors)
                pcd.normals = o3d.utility.Vector3dVector()
                print(f"{scene.alg_name} -- {scene.id}")
                o3d.visualization.draw_geometries([pcd])

            except Exception as e:
                logger.exception(f"Exception when running ME on {scene.alg_name}:{scene.id}. "
                                 f"Skipping scene and moving on...")
                logger.error(str(e))
