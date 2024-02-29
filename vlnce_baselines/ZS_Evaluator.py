from habitat import Config
from vlnce_baselines.common.env_utils import construct_envs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_trainer(name="ZS-Evaluator")
class ZeroShotVlnEvaluator(BaseTrainer):
    def __init__(self, config: Config, segment_module=None, mapping_module=None) -> None:
        super().__init__()
        self.config = config
        self._flush_secs = 30
        self.segment_module = segment_module
        self.mapping_module = mapping_module
    
    @property
    def flush_secs(self):
        return self._flush_secs

    @flush_secs.setter
    def flush_secs(self, value: int):
        self._flush_secs = value
    
    def _set_eval_config(self):
        print("set eval configs")
        
    def _init_envs(self):
        print("start to initialize environments")
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.config.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        print("initializing environments finished!")
    
    def _initialize_policy(self):
        print("start to initialize policy")
    
    def rollout(self):
        print("start to rollout")
    
    def eval(self):
        self._set_eval_config()
        self._init_envs()
        self._initialize_policy()
        self.rollout()