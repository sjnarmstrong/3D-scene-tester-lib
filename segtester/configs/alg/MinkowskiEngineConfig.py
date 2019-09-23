import segtester.configs.base as BCNF


class ExecuteMinkowskiEngineConfig(BCNF.ConfigParser):
    def __init__(self):
        super().__init__()
        self.id = BCNF.OptionalMember()
        self.weights_path = BCNF.RequiredMember()
        self.voxel_sizes = BCNF.OptionalMember(default_ret=[0.1, 0.05, 0.02])
        self.num_classes = BCNF.OptionalMember(default_ret=20)
        self.cpu_only = BCNF.OptionalMember(default_ret=False)
        self.save_path = BCNF.OptionalMember(default_ret="{dataset_id}/{scene_id}/{alg_name}/")

    def __call__(self, base_result_path, dataset_conf, *args, **kwargs):
        from segtester.algs.minkowski.minkowski import ExecuteMinkowskiEngine
        ExecuteMinkowskiEngine(self)(base_result_path=base_result_path, dataset_conf=dataset_conf, *args, **kwargs)

    @staticmethod
    def register_type(out_dict):
        out_dict['MinkowskiEngine'] = ExecuteMinkowskiEngineConfig
