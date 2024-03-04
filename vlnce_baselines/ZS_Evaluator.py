import numpy as np
from PIL import Image
from typing import List
from collections import defaultdict

import torch
from torch import Tensor
import torch.distributed as distr
from torchvision import transforms

from habitat import Config
from habitat.core.simulator import Observations
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.utils.data_utils import OrderedSet
from vlnce_baselines.map.mapping import Semantic_Mapping
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.map.semantic_prediction import GroundedSAM


@baseline_registry.register_trainer(name="ZS-Evaluator")
class ZeroShotVlnEvaluator(BaseTrainer):
    def __init__(self, config: Config, segment_module=None, mapping_module=None) -> None:
        super().__init__()
        self._flush_secs = 30 # for tensorboard
        self.config = config
        self.map_args = config.MAP
        self.classes = ["floor", "sink", "kitchen counter"]
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
    
    def _set_eval_config(self) -> None:
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
        
    def _init_envs(self) -> None:
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
        self.detected_classes = OrderedSet()
        print("initializing environments finished!")
    
    def _initialize_policy(self) -> None:
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
        
    def _get_sem_pred(self, rgb: np.ndarray) -> np.ndarray:
        # mask.shape=[num_detected_classes, h, w]
        # labels looks like: ["kitchen counter 0.69", "floor 0.37"]
        masks, labels, annotated_images = self.segment_module.segment(rgb, classes=self.classes)
        self.mapping_module.rgb_vis = annotated_images
        assert len(masks) == len(labels), \
        f"The number of masks not equal to the number of labels!"
        class_names = self._process_labels(labels)
        masks = self._process_masks(masks, class_names)
        
        return masks.transpose(1, 2, 0)
    
    def _process_labels(self, labels: List[str]) -> List:
        class_names = []
        for label in labels:
            class_name = " ".join(label.split(' ')[:-1])
            class_names.append(class_name)
            self.detected_classes.add(class_name)
        
        return class_names
        
    def _process_masks(self, masks: np.ndarray, labels: List[str]):
        """Since we are now handling the open-vocabulary semantic mapping problem,
        we need to maintain a mask tensor with dynamic channels. The idea is to combine
        all same class tensors into one tensor, then let the "detected_classes" to 
        record all classes without duplication. Finally we can use each class's index
        in the detected_classes to determine as it's channel in the mask tensor.
        The organization of mask is similar to chaplot's Sem_Exp, please refer to this link:
        https://github.com/devendrachaplot/Object-Goal-Navigation/blob/master/agents/utils/semantic_prediction.py#L41
        
        Args:
            masks (np.ndarray): shape:(c,h,w), each instance(even the same class) has one channel
            labels (List[str]): masks' corresponding labels. len(masks) = len(labels)

        Returns:
            final_masks (np.ndarray): each mask will find their channel in self.detected_classes.
            len(final_masks) = len(self.detected_classes)
        """
        same_label_indexs = defaultdict(list)
        for idx, item in enumerate(labels):
            same_label_indexs[item].append(idx) #dict {class name: [idx]}
        combined_mask = np.zeros((len(same_label_indexs), *masks.shape[1:]))
        for i, indexs in enumerate(same_label_indexs.values()):
            combined_mask[i] = np.sum(masks[indexs, ...], axis=0)
        
        idx = [self.detected_classes.index(label) for label in same_label_indexs.keys()]
        max_idx = max(idx)
        final_masks = np.zeros((max_idx + 1, *masks.shape[1:])) # init final masks as [max_idx, h, w]
        final_masks[idx, ...] = combined_mask
        
        return final_masks
    
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
    
    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        concated_obs = self._concat_obs(obs)
        state = self._preprocess_state(concated_obs)
        
        return state # (c,h,w)
    
    def _batch_obs(self, n_obs: List[Observations]) -> Tensor:
        n_states = [self._preprocess_obs(obs) for obs in n_obs]
        max_channels = max([len(state) for state in n_states])
        batch = np.stack([np.pad(state, 
                [(0, max_channels - state.shape[0]), 
                 (0, 0), 
                 (0, 0)], 
                mode='constant') 
         for state in n_states], axis=0)
        
        return torch.from_numpy(batch).to(self.device)
            
    def rollout(self):
        """
        execute a whole episode which consists of a sequence of sub-steps
        """
        print("start to rollout")
        # self.envs.resume_all()
        # import pdb;pdb.set_trace()
        obs = self.envs.reset() #type(obs): list
        batch_obs = self._batch_obs(obs)
        
        self.mapping_module.init_map_and_pose(num_detected_classes=len(self.detected_classes))
        poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
        _, local_map, _, local_pose = self.mapping_module(batch_obs, poses)
        self.mapping_module.update_map(step=1)
        # batch = batch_obs(obs, self.device)
    
    def eval(self):
        self._set_eval_config()
        self._init_envs()
        self._initialize_policy()
        self.rollout()