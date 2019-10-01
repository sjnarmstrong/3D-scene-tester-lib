import segtester.configs.base as BCNF


class Execute3DMVConfig(BCNF.ConfigParser):
    def __init__(self):
        super().__init__()
        self.id = BCNF.OptionalMember()
        self.cpu_only = BCNF.RequiredMember()
        self.grid_dimX = BCNF.OptionalMember(default_ret=31)
        self.grid_dimY = BCNF.OptionalMember(default_ret=31)
        self.grid_dimZ = BCNF.OptionalMember(default_ret=62)
        self.num_nearest_images = BCNF.OptionalMember(default_ret=1)

    def __call__(self, base_result_path, dataset_conf, *args, **kwargs):
        from segtester.algs.mv3d.mv3d import Execute3DMV
        Execute3DMV(self)(base_result_path=base_result_path, dataset_conf=dataset_conf, *args, **kwargs)

    @staticmethod
    def register_type(out_dict):
        out_dict['3DMV'] = Execute3DMVConfig
