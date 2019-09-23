import segtester.configs.base as BCNF


class KITTIConfig(BCNF.ConfigParser):

    def __init__(self):
        super().__init__()
        self.id = BCNF.OptionalMember()
        self.file_map = BCNF.RequiredMember()

    @staticmethod
    def register_type(out_dict):
        out_dict['KITTI'] = KITTIConfig