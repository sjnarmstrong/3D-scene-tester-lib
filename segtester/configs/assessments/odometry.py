import segtester.configs.base as BCNF
from segtester.configs.dataset import DATASET_MAP
from segtester.configs.dataset.results import ResultsConfig
from typing import List


class OdometryConfig(BCNF.ConfigParser):

    # noinspection PyTypeChecker
    def __init__(self):
        from evo.core.metrics import PoseRelation, Unit
        super().__init__()
        self.id: str = BCNF.OptionalMember()
        self.gt_dataset = BCNF.RequiredMember(BCNF.MappableMember(DATASET_MAP))
        self.est_dataset: 'ResultsConfig' = BCNF.RequiredMember(ResultsConfig)
        self.alignment_options: List[str] = BCNF.OptionalMember(default_ret=["noalign"])
        self.pose_relations: List[PoseRelation] = BCNF.OptionalMember(default_ret=[
            PoseRelation.full_transformation,
            PoseRelation.translation_part,
            PoseRelation.rotation_part])
        self.run_ape_tests = BCNF.OptionalMember(default_ret=True)
        self.run_rpe_tests = BCNF.OptionalMember(default_ret=True)
        self.create_plots = BCNF.OptionalMember(default_ret=True)
        self.confirm_overwrite = BCNF.OptionalMember(default_ret=False)
        self.save_path = BCNF.OptionalMember(default_ret="{dataset_id}/{scene_id}/{alg_name}/odo/"
                                                         "{alignment_opt}_{pose_relation}")

    def __call__(self, base_result_path, *args, **kwargs):
        from segtester.assessments.odometry import OdometryAssessment
        OdometryAssessment(self)(base_result_path=base_result_path, gt_dataset_conf=self.gt_dataset,
                                 est_dataset_conf=self.est_dataset, *args, **kwargs)

    @staticmethod
    def register_type(out_dict):
        out_dict['OdometryAssessment'] = OdometryConfig

    def get_dataset(self):
        from segtester.dataloaders.scannet import ScannetDataset
        return ScannetDataset(self)
