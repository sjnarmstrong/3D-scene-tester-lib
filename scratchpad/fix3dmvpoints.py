import open3d as o3d
import torch
import numpy as np
import glob
import os
files = glob.glob("/media/sholto/Datasets1/results/ActualResults/20191008/SCANNET/*/3DMV/")

for file in files:
    f = np.load(f"{file}probs.npz")
    lk = f[f.files[0]]
    pcd = o3d.io.read_point_cloud(f"{file}pcd.ply")
    pts = np.array(pcd.points)

    if len(lk) != len(pts):
        print(f"detected invalid pcd {file}")

        diff = pts[np.random.randint(0, 20, 20)] - pts[np.random.randint(0, 20, 20)]
        if np.allclose(diff / 0.048, np.round(diff / 0.048).astype(np.int)):
            scale = 0.048
        elif np.allclose(diff / 0.05, np.round(diff / 0.05).astype(np.int)):
            scale = 0.05
            print("mmm seems scale is 0.05... odd")
            print(diff)
        else:
            print("Not sure about the scale")
            print(diff)
            continue

        sc_occ = torch.load(f"{file}/torch/scene_occ.torch")
        sc_occ = sc_occ[0]*sc_occ[1]
        sc_occ_st = torch.load(f"{file}/torch/occ_start.torch")
        new_pts = (torch.stack((torch.where(sc_occ))).to(dtype=torch.double)*scale+sc_occ_st[[2, 0, 1], None])[[1,2,0]].T.cpu().numpy()
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_pts)
        os.rename(f"{file}pcd.ply", f"{file}pcd_bkp.ply")
        o3d.io.write_point_cloud(f"{file}pcd.ply", new_pcd)
