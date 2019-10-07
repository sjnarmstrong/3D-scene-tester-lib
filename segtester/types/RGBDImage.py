class RGBDFrame(object):
    def get_color_image(self):
        raise NotImplementedError

    def get_depth_image(self):
        raise NotImplementedError

    def get_camera_to_world(self):
        raise NotImplementedError

    def get_timestamp(self):
        raise NotImplementedError
