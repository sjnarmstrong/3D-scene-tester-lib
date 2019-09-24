from segtester.configs.alg.MinkowskiEngineConfig import ExecuteMinkowskiEngineConfig
from segtester.configs.alg.SemanticFusionConfig import ExecuteSemanticFusionConfig

ALG_MAP = {}
ExecuteMinkowskiEngineConfig.register_type(ALG_MAP)
ExecuteSemanticFusionConfig.register_type(ALG_MAP)
