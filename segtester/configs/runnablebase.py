import segtester.configs.base as BCNF
from segtester.configs.runnable import RUNNABLE_MAP
from segtester import logger, SEP
from typing import List, Callable


class RunnableConfig(BCNF.ConfigParser):

    # noinspection PyTypeChecker
    def __init__(self):
        super().__init__()
        self.name: str = BCNF.OptionalMember()
        self.description: str = BCNF.OptionalMember()
        self.base_result_path: str = BCNF.RequiredMember()
        self.tests: List[Callable] = BCNF.RequiredMember(BCNF.IterableMember(BCNF.MappableMember(RUNNABLE_MAP)))

    def __call__(self, *args, **kwargs):
        logger.info("Starting Evaluation")
        for i, test in enumerate(self.tests):
            logger.info(f"Running test {i}/{len(self.tests)}")
            logger.info(SEP)
            try:
                test(base_result_path=self.format_string_with_meta(self.base_result_path))
            except Exception as e:
                logger.error(f"An unknown error has occured. Skipping test {i}")
                logger.error(f"{e}")
