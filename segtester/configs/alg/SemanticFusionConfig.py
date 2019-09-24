import segtester.configs.base as BCNF


class ExecuteSemanticFusionConfig(BCNF.ConfigParser):
    def __init__(self):
        super().__init__()
        self.id = BCNF.OptionalMember()
        self.cpu_only = BCNF.OptionalMember(default_ret=False)
        self.save_path = BCNF.OptionalMember(default_ret="{dataset_id}/{scene_id}/{alg_name}/")
        self.cnn_skip_frames = BCNF.OptionalMember(default_ret=10)
        self.use_crf = BCNF.OptionalMember(default_ret=False)
        self.caffe_use_cpu = BCNF.OptionalMember(default_ret=False)
        self.crf_skip_frames = BCNF.OptionalMember(default_ret=500)
        self.crf_iterations = BCNF.OptionalMember(default_ret=10)
        self.prototext_model_path = BCNF.RequiredMember()
        self.caffemodel_path = BCNF.RequiredMember()
        self.class_colour_lookup_path = BCNF.RequiredMember()
        self.save_path = BCNF.OptionalMember(default_ret="{dataset_id}/{scene_id}/{alg_name}/")
        self.alg_name = BCNF.OptionalMember(default_ret="SemanticFusion")

    def __call__(self, base_result_path, dataset_conf, *args, **kwargs):
        from segtester.algs.semanticfusion.semanticfusion import ExecuteSemanticFusion
        ExecuteSemanticFusion(self)(base_result_path=base_result_path, dataset_conf=dataset_conf, *args, **kwargs)

    @staticmethod
    def register_type(out_dict):
        out_dict['SemanticFusion'] = ExecuteSemanticFusionConfig
