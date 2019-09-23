from segtester.types import Dataset
import open3d as o3d  # Seems if this is lower stuff breaks....

import torch
import MinkowskiEngine as ME
from .model import MinkUNet34C
import numpy as np
from tqdm import tqdm
import os


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from segtester.configs.alg.MinkowskiEngineConfig import ExecuteMinkowskiEngineConfig


def generate_input_sparse_tensor(pcd, device, voxel_size=0.05):
    coords = np.array(pcd.points)
    feats = np.array(pcd.colors)

    quantized_coords = np.floor(coords / voxel_size)
    inds = ME.utils.sparse_quantize(quantized_coords)

    batch = [(quantized_coords[inds], feats[inds], pcd)]
    coordinates_, featrues_, pcds = list(zip(*batch))
    coordinates, features = ME.utils.sparse_collate(coordinates_, featrues_)

    return ME.SparseTensor(features - 0.5, coords=coordinates).to(device)


class ExecuteMinkowskiEngine:
    def __init__(self, conf: 'ExecuteMinkowskiEngineConfig'):
        self.conf = conf

        if conf.cpu_only:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define a model and load the weights
        self.model = MinkUNet34C(3, conf.num_classes).to(self.device)
        model_dict = torch.load(conf.weights_path)
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, base_result_path, dataset_conf, *_, **__):
        dataset: Dataset = dataset_conf.get_dataset()
        with torch.no_grad():
            for scene in tqdm(dataset.scenes, desc="scene"):
                pcd = scene.get_pcd()

                for voxel_size in tqdm(self.conf.voxel_sizes, desc="voxel_size"):
                    sparce_tensor = generate_input_sparse_tensor(pcd, self.device, voxel_size)

                    soutput = self.model(sparce_tensor)

                    likelihoods = soutput.F
                    _, max_pred = likelihoods.max(1)
                    likelihoods = likelihoods.cpu().numpy()
                    max_pred = max_pred.cpu().numpy()
                    coordinates = soutput.C.numpy()[:, :3]

                    pred_pcd = o3d.geometry.PointCloud()
                    pred_pcd.points = o3d.utility.Vector3dVector(coordinates * voxel_size)

                    save_path = self.conf.format_string_with_meta(f"{base_result_path}/{self.conf.save_path}", **{
                        "dataset_id": dataset_conf.id, "scene_id": scene.id, "alg_name": f"MinkUNet34C_{voxel_size}",
                    })
                    os.makedirs(save_path, exist_ok=True)
                    o3d.io.write_point_cloud(f"{save_path}/pcd.ply", pred_pcd)
                    np.savez_compressed(f"{save_path}/probs", likelihoods=likelihoods, class_ids=max_pred)
