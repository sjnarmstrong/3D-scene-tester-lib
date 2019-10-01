from segtester.configs.alg.MinkowskiEngineConfig import ExecuteMinkowskiEngineConfig
from segtester.configs.alg.SemanticFusionConfig import ExecuteSemanticFusionConfig
from segtester.configs.alg.MV3DConfig import Execute3DMVConfig

ALG_MAP = {}
ExecuteMinkowskiEngineConfig.register_type(ALG_MAP)
ExecuteSemanticFusionConfig.register_type(ALG_MAP)
Execute3DMVConfig.register_type(ALG_MAP)
