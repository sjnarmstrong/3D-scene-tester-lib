import segtester.configs.base as BCNF


class SCANNETConfig(BCNF.ConfigParser):

    def __init__(self):
        super().__init__()
        self.id = BCNF.OptionalMember()
        self.file_map = BCNF.RequiredMember()

    @staticmethod
    def register_type(out_dict):
        out_dict['SCANNET'] = SCANNETConfig

    def get_dataset(self):
        from segtester.dataloaders.scannet import ScannetDataset
        return ScannetDataset(self)
