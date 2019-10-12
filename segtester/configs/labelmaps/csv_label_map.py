import segtester.configs.base as BCNF


class CSVLabelMap(BCNF.ConfigParser):

    def __init__(self):
        super().__init__()
        self.csv_path = BCNF.RequiredMember()

    @staticmethod
    def register_type(out_dict):
        out_dict['CSVLabelMap'] = CSVLabelMap

    def get_label_map(self):
        from segtester.labelmaps.csv_label_map import CSVLabelMap
        return CSVLabelMap(self)
