import segtester.configs.base as BCNF
from segtester.configs.dataset.results import ResultsConfig


class VisualisePredictionsConfig(BCNF.ConfigParser):

    # noinspection PyTypeChecker
    def __init__(self):
        super().__init__()
        self.id: str = BCNF.OptionalMember()
        self.result_dataset: 'ResultsConfig' = BCNF.RequiredMember(ResultsConfig)

    def __call__(self, base_result_path, *args, **kwargs):
        from segtester.runnables.visualise_labels import VisualisePredictions
        VisualisePredictions(self)(base_result_path=base_result_path, dataset_conf=self.result_dataset, *args, **kwargs)

    @staticmethod
    def register_type(out_dict):
        out_dict['VisualisePredictions'] = VisualisePredictionsConfig