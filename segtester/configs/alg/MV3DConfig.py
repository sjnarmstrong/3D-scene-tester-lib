import segtester.configs.base as BCNF


class Execute3DMVConfig(BCNF.ConfigParser):
    def __init__(self):
        super().__init__()
        self.id = BCNF.OptionalMember()
        self.model_path = BCNF.RequiredMember()
        self.model2d_orig_path = BCNF.RequiredMember()
        self.cpu_only = BCNF.OptionalMember(default_ret=False)
        self.grid_dimX = BCNF.OptionalMember(default_ret=31)
        self.grid_dimY = BCNF.OptionalMember(default_ret=31)
        self.grid_dimZ = BCNF.OptionalMember(default_ret=62)
        self.num_nearest_images = BCNF.OptionalMember(default_ret=8)
        self.model2d_type = BCNF.OptionalMember(default_ret='scannet')
        self.depth_min = BCNF.OptionalMember(default_ret=0.4)
        self.depth_max = BCNF.OptionalMember(default_ret=4.0)
        self.voxel_size = BCNF.OptionalMember(default_ret=0.05)
        self.num_classes = BCNF.OptionalMember(default_ret=42)
        self.process_nth_frame = BCNF.OptionalMember(default_ret=10)
        self.alg_name = BCNF.OptionalMember(default_ret="3DMV")
        self.save_path = BCNF.OptionalMember(default_ret="{dataset_id}/{scene_id}/{alg_name}/")
        self.save_frames = BCNF.OptionalMember(default_ret=False)

    def __call__(self, base_result_path, dataset_conf, *args, **kwargs):
        from segtester.algs.mv3d.mv3d import Execute3DMV
        Execute3DMV(self)(base_result_path=base_result_path, dataset_conf=dataset_conf, *args, **kwargs)

    @staticmethod
    def register_type(out_dict):
        out_dict['3DMV'] = Execute3DMVConfig
