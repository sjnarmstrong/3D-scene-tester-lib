import segtester.configs.base as BCNF
from segtester.configs.dataset import DATASET_MAP
from segtester.configs.dataset.results import ResultsConfig
from segtester.configs.labelmaps.csv_label_map import CSVLabelMap


class Segmentation3d(BCNF.ConfigParser):

    # noinspection PyTypeChecker
    def __init__(self):
        super().__init__()
        self.id: str = BCNF.OptionalMember()
        self.gt_dataset = BCNF.RequiredMember(BCNF.MappableMember(DATASET_MAP))
        self.est_dataset: 'ResultsConfig' = BCNF.RequiredMember(ResultsConfig)
        self.label_map: 'CSVLabelMap' = BCNF.RequiredMember(CSVLabelMap)
        self.label_map_dest_col: str = BCNF.RequiredMember()
        self.skip_existing = BCNF.OptionalMember(default_ret=False)
        self.save_path = BCNF.OptionalMember(default_ret="{dataset_id}/{scene_id}/{alg_name}/seg/seg3d")
        self.point_dist_thresh = BCNF.OptionalMember(default_ret=0.5)

    def __call__(self, base_result_path, *args, **kwargs):
        from segtester.assessments.segmentation3d import Segmentation3DAssessment
        Segmentation3DAssessment(self)(base_result_path=base_result_path, gt_dataset_conf=self.gt_dataset,
                                       est_dataset_conf=self.est_dataset, *args, **kwargs)

    @staticmethod
    def register_type(out_dict):
        out_dict['Segmentation3d'] = Segmentation3d
