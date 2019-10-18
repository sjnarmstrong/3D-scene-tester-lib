import segtester.configs.base as BCNF
from segtester.configs.dataset.results import ResultsConfig
from segtester.configs.labelmaps.csv_label_map import CSVLabelMap


class VisualisePredictionsConfig(BCNF.ConfigParser):

    # noinspection PyTypeChecker
    def __init__(self):
        super().__init__()
        self.id: str = BCNF.OptionalMember()
        self.result_dataset: 'ResultsConfig' = BCNF.RequiredMember(ResultsConfig)
        self.label_map: 'CSVLabelMap' = BCNF.RequiredMember(CSVLabelMap)
        self.label_map_dest_col: str = BCNF.RequiredMember()
        self.label_map_name_col: str = BCNF.RequiredMember()
        self.skip_existing = BCNF.OptionalMember(default_ret=False)
        self.pause_on_scene = BCNF.OptionalMember(default_ret=False)
        self.save_path = BCNF.OptionalMember(default_ret="{dataset_id}/{scene_id}/{alg_name}/seg/vis")

    def __call__(self, base_result_path, *args, **kwargs):
        from segtester.runnables.visualise_labels import VisualisePredictions
        VisualisePredictions(self)(base_result_path=base_result_path, dataset_conf=self.result_dataset, *args, **kwargs)

    @staticmethod
    def register_type(out_dict):
        out_dict['VisualisePredictions'] = VisualisePredictionsConfig