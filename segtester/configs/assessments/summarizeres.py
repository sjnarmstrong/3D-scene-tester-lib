import segtester.configs.base as BCNF
from segtester.configs.dataset.results import ResultsConfig
from segtester.configs.labelmaps.csv_label_map import CSVLabelMap


class SummarizeRes(BCNF.ConfigParser):

    # noinspection PyTypeChecker
    def __init__(self):
        super().__init__()
        self.id: str = BCNF.OptionalMember()
        self.est_dataset: 'ResultsConfig' = BCNF.RequiredMember(ResultsConfig)
        self.label_map: 'CSVLabelMap' = BCNF.RequiredMember(CSVLabelMap)
        self.label_map_dest_col: str = BCNF.RequiredMember()
        self.label_map_dest_name_col: str = BCNF.RequiredMember()

    def __call__(self, base_result_path, *args, **kwargs):
        from segtester.assessments.summarizeresults import SummarizeRes
        SummarizeRes(self)(base_result_path=base_result_path, est_dataset_conf=self.est_dataset, *args, **kwargs)

    @staticmethod
    def register_type(out_dict):
        out_dict['SummarizeRes'] = SummarizeRes
