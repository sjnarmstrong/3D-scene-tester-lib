import segtester.configs.base as BCNF


class ResultsConfig(BCNF.ConfigParser):

    def __init__(self):
        super().__init__()
        self.id = BCNF.OptionalMember()
        self.file_map = BCNF.RequiredMember()
        self.dataset_id = BCNF.OptionalMember()
        self.base_result_path = BCNF.OptionalMember()
        self.load_path = BCNF.OptionalMember(default_ret="{dataset_id}/{scene_id}/{alg_name}/")
        self.save_path = BCNF.OptionalMember(default_ret="{dataset_id}/{scene_id}/{alg_name}/")

    @staticmethod
    def register_type(out_dict):
        out_dict['Results'] = ResultsConfig

    def get_dataset(self):
        from segtester.dataloaders.results import ResultsDataset
        return ResultsDataset(self)
