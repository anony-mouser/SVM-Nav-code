import numpy as np
from PIL import Image

import torch
import torch.distributed as distr
from torchvision import transforms

from habitat import Config
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.map.semantic_prediction import GroundedSAM
from vlnce_baselines.map.mapping import Semantic_Mapping


@baseline_registry.register_trainer(name="ZS-Evaluator")
class ZeroShotVlnEvaluator(BaseTrainer):
    def __init__(self, config: Config, segment_module=None, mapping_module=None) -> None:
        super().__init__()
        self._flush_secs = 30 # for tensorboard
        self.config = config
        self.map_args = config.MAP
        self.classes = ["sink", "kitchen counter"]
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        # initialize transform for RGB observations
        self.trans = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((self.map_args.FRAME_HEIGHT, self.map_args.FRAME_WIDTH), # (120, 160)
                               interpolation=Image.NEAREST)])
    
    # for tensorboard
    # @property
    # def flush_secs(self):
    #     return self._flush_secs

    # @flush_secs.setter
    # def flush_secs(self, value: int):
    #     self._flush_secs = value
    
    def _set_eval_config(self):
        print("set eval configs")
        self.config.defrost()
        self.config.MAP.DEVICE = self.config.TORCH_GPU_ID
        self.config.MAP.HFOV = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        self.config.MAP.AGENT_HEIGHT = self.config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT
        self.config.MAP.NUM_ENVIRONMENTS = self.config.NUM_ENVIRONMENTS
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
        self.config.freeze()
        torch.cuda.set_device(self.device)
        
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
        # Semantic Segmentation
        self.segment_module = GroundedSAM(self.config)
        # Semantic Mapping
        self.mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        self.mapping_module.eval()
        
    def _concat_obs(self, obs: np.ndarray) -> np.ndarray:
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1) # (h, w, c)->(c, h, w)
        
        return state
    
    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        state = state.transpose(1, 2, 0)
        rgb = state[:, :, :3].astype(np.uint8) #[3, h, w]
        depth = state[:, :, 3:4] #[1, h, w]
        min_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        env_frame_width = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        
        sem_seg_pred = self._get_sem_pred(rgb) #[num_detected_classes, h, w]
        depth = self._preprocess_depth(depth, min_depth, max_depth) #[1, h, w]
        
        # ds: Downscaling factor
        # args.env_frame_width = 640, args.frame_width = 160
        ds = env_frame_width // self.map_args.FRAME_WIDTH # ds = 4
        if ds != 1:
            rgb = np.asarray(self.trans(rgb.astype(np.uint8))) # resize
            depth = depth[ds // 2::ds, ds // 2::ds] # down scaling start from 2, step=4
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2) # recover depth.shape to (height, width, 1)
        state = np.concatenate((rgb, depth, sem_seg_pred),axis=2).transpose(2, 0, 1) # (4+num_detected_classes, h, w)
        
        return state
        
    def _get_sem_pred(self, rgb: np.ndarray):
        # mask.shape=[num_detected_classes, h, w]
        # labels looks like: ["kitchen counter 0.69", "floor 0.37"]
        masks, labels, annotated_images = self.segment_module.segment(rgb, classes=self.classes)
        
        return masks.transpose(1, 2, 0)
    
    def _preprocess_depth(self, depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        # Preprocesses a depth map by handling missing values, removing outliers, and scaling the depth values.
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99 # turn too far pixels to invalid
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0 # then turn all invalid pixels to vision_range(100)
        depth = min_depth * 100.0 + depth * max_depth * 100.0
        return depth
    
    def _preprocess_obs(self, obs):
        concated_obs = self._concat_obs(obs)
        state = self._preprocess_state(concated_obs)
        
        return state
    
    def rollout(self):
        """
        execute a whole episode which consists of a sequence of sub-steps
        """
        print("start to rollout")
        # self.envs.resume_all()
        # import pdb;pdb.set_trace()
        obs = self.envs.reset() #type(obs): list
        state0 = self._preprocess_obs(obs[0])
        # batch = batch_obs(obs, self.device)
    
    def eval(self):
        self._set_eval_config()
        self._init_envs()
        self._initialize_policy()
        self.rollout()