class Scene:
    def __init__(self):
        self.id: str = None

    def get_pcd(self):
        raise NotImplementedError()

    def get_labeled_pcd(self):
        raise NotImplementedError()

    def get_rgb_depth_image_it(self):
        raise NotImplementedError()

    def get_num_frames(self):
        raise NotImplementedError()

    def get_intrinsic_rgb(self):
        raise NotImplementedError()

    def get_intrinsic_depth(self):
        raise NotImplementedError()

    def get_extrinsic_rgb(self):
        raise NotImplementedError()

    def get_extrinsic_depth(self):
        raise NotImplementedError()

    def get_rgb_size(self):
        raise NotImplementedError()

    def get_depth_size(self):
        raise NotImplementedError()

    def get_depth_scale(self):
        return NotImplementedError()

    def get_depth_position_it(self):
        return NotImplementedError()

    def get_image_info_index(self, index):
        return NotImplementedError()

    def get_image_viewpoints_grid(self, vox_dims, occ_start, voxel_size=0.05, process_nth_frame=10, device=None):
        from segtester.util.create_image_viewpoint_grid import create_image_viewpoints_grid
        return create_image_viewpoints_grid(self, vox_dims, occ_start, voxel_size, process_nth_frame, device)

    def get_tensor_occ(self, voxel_size=0.05, padding_x=31 // 2, padding_y=31 // 2, device=None):
        from segtester.util.create_tensor_occ_from_scene import create_tensor_occ
        return create_tensor_occ(self, voxel_size, padding_x, padding_y, device)

    @staticmethod
    def get_world_to_grids(occ_grid_shape, occ_start, voxel_size=0.05, device=None):
        from segtester.util.create_world_to_grid import create_world_to_grids
        return create_world_to_grids(occ_grid_shape, occ_start, voxel_size, device)
