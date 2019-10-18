import segtester.configs.base as BCNF


class NYUv2Config(BCNF.ConfigParser):

    def __init__(self):
        super().__init__()
        self.id = BCNF.OptionalMember()
        self.zip_file_loc = BCNF.RequiredMember()
        self.gt_file_loc = BCNF.RequiredMember()

    @staticmethod
    def register_type(out_dict):
        out_dict['NYUv2'] = NYUv2Config

    def get_dataset(self):
        from segtester.dataloaders.NYUv2 import NYUv2Dataset
        return NYUv2Dataset(self)
