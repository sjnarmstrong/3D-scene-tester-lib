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

    def get_image_viewpoints_grid(self, vox_dims, min_pcd, voxel_size, padding, process_nth_frame):
        return NotImplementedError()
