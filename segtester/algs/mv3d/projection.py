import numpy as np
import torch
from torch.autograd import Function


class ProjectionHelper(object):
    def __init__(self, intrinsic, depth_min, depth_max, image_dims, volume_dims, voxel_size):
        self.intrinsic = intrinsic
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.image_dims = image_dims
        self.volume_dims = volume_dims
        self.voxel_size = voxel_size

    def depth_to_skeleton(self, ux, uy, depth):
        x = (ux - self.intrinsic[0][2]) / self.intrinsic[0][0]
        y = (uy - self.intrinsic[1][2]) / self.intrinsic[1][1]
        return torch.Tensor([depth*x, depth*y, depth])

    def skeleton_to_depth(self, p):
        x = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        y = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        return torch.Tensor([x, y, p[2]])

    def compute_frustum_bounds(self, world_to_grid, camera_to_world):
        corner_points = camera_to_world.new(8, 4, 1).fill_(1)
        # Compute 8 bounding box points from the cameras point of view in the real world
        # depth min
        corner_points[0][:3] = self.depth_to_skeleton(0, 0, self.depth_min).unsqueeze(1)
        corner_points[1][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_min).unsqueeze(1)
        corner_points[2][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_min).unsqueeze(1)
        corner_points[3][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_min).unsqueeze(1)
        # depth max
        corner_points[4][:3] = self.depth_to_skeleton(0, 0, self.depth_max).unsqueeze(1)
        corner_points[5][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_max).unsqueeze(1)
        corner_points[6][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_max).unsqueeze(1)
        corner_points[7][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_max).unsqueeze(1)

        # transform points from camera's perspective to the worlds perspective
        p = torch.bmm(camera_to_world.repeat(8, 1, 1), corner_points)
        # transform points from world to grids perspective and account for rounding inaccuracies
        pl = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.floor(p)))
        pu = torch.round(torch.bmm(world_to_grid.repeat(8, 1, 1), torch.ceil(p)))
        # assuming the transform is to the center of the grid because the box must go from negative to positive
        bbox_min0, _ = torch.min(pl[:, :3, 0], 0)
        bbox_min1, _ = torch.min(pu[:, :3, 0], 0)
        bbox_min = np.minimum(bbox_min0.cpu(), bbox_min1.cpu())
        bbox_max0, _ = torch.max(pl[:, :3, 0], 0)
        bbox_max1, _ = torch.max(pu[:, :3, 0], 0) 
        bbox_max = np.maximum(bbox_max0.cpu(), bbox_max1.cpu())
        return bbox_min, bbox_max

    # TODO make runnable on cpu as well...
    def compute_projection(self, depth, camera_to_world, world_to_grid, pcd, xy, c_img):
        # compute projection by voxels -> image
        world_to_camera = torch.inverse(camera_to_world)
        grid_to_world = torch.inverse(world_to_grid)
        voxel_bounds_min, voxel_bounds_max = self.compute_frustum_bounds(world_to_grid, camera_to_world)
        voxel_bounds_min = np.maximum(voxel_bounds_min, 0).cuda()
        voxel_bounds_max = np.minimum(voxel_bounds_max, self.volume_dims).float().cuda()

        # coordinates within frustum bounds
        # The fist few lines here seem to be a very long way to produce (0,0,0,1), (1,0,0,1), (2,0,0,1) ... (x,y,z,1)
        # for each coordinate in the current block
        lin_ind_volume = torch.arange(0, self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2], out=torch.LongTensor()).cuda()
        coords = camera_to_world.new(4, lin_ind_volume.size(0)) # ment to construct new tensor with same values as current tensor
        coords[2] = lin_ind_volume / (self.volume_dims[0]*self.volume_dims[1])
        tmp = lin_ind_volume - (coords[2]*self.volume_dims[0]*self.volume_dims[1]).long()
        coords[1] = tmp / self.volume_dims[0]
        coords[0] = torch.remainder(tmp, self.volume_dims[0])
        coords[3].fill_(1)
        # ge is element wise greater or equil to. So logical ading all points that lie within bounds
        # not sure if coords is ment to be already augmented to cameras pov. It appears so here
        # coords = torch.mm(camera_to_world,coords) Dont think so because points are already in perspective of grid
        mask_frustum_bounds = torch.ge(coords[0], voxel_bounds_min[0]) * torch.ge(coords[1], voxel_bounds_min[1]) * torch.ge(coords[2], voxel_bounds_min[2])
        mask_frustum_bounds = mask_frustum_bounds * torch.lt(coords[0], voxel_bounds_max[0]) * torch.lt(coords[1], voxel_bounds_max[1]) * torch.lt(coords[2], voxel_bounds_max[2])
        if not mask_frustum_bounds.any():
            #print('error: nothing in frustum bounds')
            return None

        # Mask and prepopulate coords
        lin_ind_volume = lin_ind_volume[mask_frustum_bounds]
        coords = coords.resize_(4, lin_ind_volume.size(0))
        coords[2] = lin_ind_volume / (self.volume_dims[0]*self.volume_dims[1])
        tmp = lin_ind_volume - (coords[2]*self.volume_dims[0]*self.volume_dims[1]).long()
        coords[1] = tmp / self.volume_dims[0]
        coords[0] = torch.remainder(tmp, self.volume_dims[0])
        coords[3].fill_(1)

        # transform to current frame
        p = torch.mm(world_to_camera, torch.mm(grid_to_world, coords))


        ###### Start test code
        # if xy[0]%10==0 and xy[1]%10==0:
        #     print(xy)
        #     import open3d as o3d
        #     from torchvision import transforms
        #
        #     class UnNormalize(object):
        #         def __init__(self, mean, std):
        #             self.mean = mean
        #             self.std = std
        #
        #         def __call__(self, tensor):
        #             """
        #             Args:
        #                 tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        #             Returns:
        #                 Tensor: Normalized image.
        #             """
        #             for t, m, s in zip(tensor, self.mean, self.std):
        #                 t.mul_(s).add_(m)
        #                 # The normalize code -> t.sub_(m).div_(s)
        #             return tensor
        #
        #     inverse_transform = transforms.Compose([
        #         UnNormalize([0.496342, 0.466664, 0.440796], [0.277856, 0.28623, 0.291129]),
        #         transforms.ToPILImage()
        #     ])
        #     inverse_transform(c_img.cpu()).show()
        #     test = torch.mm(grid_to_world, coords[:, None].to(dtype=coords.dtype, device=coords.device)).cpu().numpy()
        #     pred_pcd = o3d.geometry.PointCloud()
        #     pred_pcd.points = o3d.utility.Vector3dVector(test[:3].T)
        #     pcd2 = o3d.geometry.PointCloud()
        #     pcd2.points = o3d.utility.Vector3dVector(pcd)
        #     pcd3 = o3d.io.read_point_cloud("/mnt/1C562D12562CEDE8/DATASETS/scannet/scenes/scans_test/scene0707_00/scene0707_00_vh_clean_2.ply")
        #     o3d.visualization.draw_geometries([pred_pcd, pcd3])
        ##### end

        # project into image Duplication of skeleton_to_depth
        p[0] = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        p[1] = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        pi = torch.round(p).long() # Nearist neighbor

        # mask out points not in frame
        valid_ind_mask = torch.ge(pi[0], 0) * torch.ge(pi[1], 0) * torch.lt(pi[0], self.image_dims[0]) * torch.lt(pi[1], self.image_dims[1])
        if not valid_ind_mask.any():
            #print('error: no valid image indices')
            return None
        valid_image_ind_x = pi[0][valid_ind_mask]
        valid_image_ind_y = pi[1][valid_ind_mask]
        valid_image_ind_lin = valid_image_ind_y * self.image_dims[0] + valid_image_ind_x
        depth_vals = torch.index_select(depth.view(-1), 0, valid_image_ind_lin)
        depth_mask = depth_vals.ge(self.depth_min) * depth_vals.le(self.depth_max) * torch.abs(depth_vals - p[2][valid_ind_mask]).le(self.voxel_size)

        if not depth_mask.any():
            #print('error: no valid depths')
            return None

        lin_ind_update = lin_ind_volume[valid_ind_mask]
        lin_ind_update = lin_ind_update[depth_mask]
        lin_indices_3d = lin_ind_update.new(self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2] + 1) #needs to be same size for all in batch... (first element has size)
        lin_indices_2d = lin_ind_update.new(self.volume_dims[0]*self.volume_dims[1]*self.volume_dims[2] + 1) #needs to be same size for all in batch... (first element has size)
        lin_indices_3d[0] = lin_ind_update.shape[0]
        lin_indices_2d[0] = lin_ind_update.shape[0]
        lin_indices_3d[1:1+lin_indices_3d[0]] = lin_ind_update
        lin_indices_2d[1:1+lin_indices_2d[0]] = torch.index_select(valid_image_ind_lin, 0, torch.nonzero(depth_mask)[:,0])
        num_ind = lin_indices_3d[0]
        return lin_indices_3d, lin_indices_2d


# Inherit from Function
class Projection(Function):

    @staticmethod
    def forward(ctx, label, lin_indices_3d, lin_indices_2d, volume_dims):
        ctx.save_for_backward(lin_indices_3d, lin_indices_2d)
        num_label_ft = 1 if len(label.shape) == 2 else label.shape[0]
        output = label.new(num_label_ft, volume_dims[2], volume_dims[1], volume_dims[0]).fill_(0)
        num_ind = lin_indices_3d[0]
        if num_ind > 0:
            vals = torch.index_select(label.view(num_label_ft, -1), 1, lin_indices_2d[1:1+num_ind])
            output.view(num_label_ft, -1)[:, lin_indices_3d[1:1+num_ind]] = vals
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_label = grad_output.clone()
        num_ft = grad_output.shape[0]
        grad_label.data.resize_(num_ft, 32, 41)
        lin_indices_3d, lin_indices_2d = ctx.saved_variables
        num_ind = lin_indices_3d.data[0]
        vals = torch.index_select(grad_output.data.contiguous().view(num_ft, -1), 1, lin_indices_3d.data[1:1+num_ind])
        grad_label.data.view(num_ft, -1)[:, lin_indices_2d.data[1:1+num_ind]] = vals
        return grad_label, None, None, None

