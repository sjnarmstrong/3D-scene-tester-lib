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
