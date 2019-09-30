class Execute3DMV:
    def __init__(self, conf: 'Execute3DMVConfig'):
        self.conf = conf

        if conf.cpu_only:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define a model and load the weights
        self.model = MinkUNet34C(3, conf.num_classes).to(self.device)
        model_dict = torch.load(conf.weights_path)
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def __call__(self, base_result_path, dataset_conf, *_, **__):
        dataset: Dataset = dataset_conf.get_dataset()

        with torch.no_grad():
            for scene in tqdm(dataset.scenes, desc="scene"):
                try: