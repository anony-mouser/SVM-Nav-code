import os
import gzip
import json
import numpy as np
from PIL import Image
from fastdtw import fastdtw
from typing import List, Any, Dict
from collections import defaultdict

import torch
from torch import Tensor
import torch.distributed as distr
from torchvision import transforms

from habitat import Config, logger
from habitat_extensions.measures import NDTW
from habitat.core.simulator import Observations
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.environments import get_env_class
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.map.value_map import ValueMap
from vlnce_baselines.utils.data_utils import OrderedSet
from vlnce_baselines.map.mapping import Semantic_Mapping
from vlnce_baselines.models.Policy import FusionMapPolicy
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import gather_list_and_concat
from vlnce_baselines.map.semantic_prediction import GroundedSAM


@baseline_registry.register_trainer(name="ZS-Evaluator")
class ZeroShotVlnEvaluator(BaseTrainer):
    def __init__(self, config: Config, segment_module=None, mapping_module=None) -> None:
        super().__init__()
        self._flush_secs = 30 # for tensorboard
        self.current_detections = None
        self.classes = ["giraffi"]
        
        self.config = config
        self.map_args = config.MAP
        self.keyboard_control = config.KEYBOARD_CONTROL
        self.width = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        self.height = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT
        self.max_step = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS
        
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
        self.trans = transforms.Compose([transforms.ToPILImage(), 
                                         transforms.Resize(
                                             (self.map_args.FRAME_HEIGHT, self.map_args.FRAME_WIDTH), 
                                             interpolation=Image.NEAREST)
                                        ])
        self.current_episode_id = None
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
        self.config.MAP.RESULTS_DIR = self.config.RESULTS_DIR
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
        
        """for DDP to load different data"""
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
        
    def _collect_val_traj(self) -> None:
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        with gzip.open(self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split)) as f:
            gt_data = json.load(f)

        self.gt_data = gt_data
        
    def _calculate_metric(self, infos: List):
        curr_eps = self.envs.current_episodes()
        info = infos[0]
        ep_id = curr_eps[0].episode_id
        gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
        pred_path = np.array(info['position']['position'])
        distances = np.array(info['position']['distance'])
        gt_length = distances[0]
        dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
        metric = {}
        metric['steps_taken'] = info['steps_taken']
        metric['distance_to_goal'] = distances[-1]
        metric['success'] = 1. if distances[-1] <= 3. else 0.
        metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
        metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
        # metric['collisions'] = info['collisions']['count'] / len(pred_path)
        metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
        metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
        metric['sdtw'] = metric['ndtw'] * metric['success']
        self.state_eps[ep_id] = metric
        
    def _initialize_policy(self) -> None:
        print("start to initialize policy")
        
        self.segment_module = GroundedSAM(self.config)
        self.mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        self.mapping_module.eval()
        
        self.value_map_module = ValueMap(self.config, self.mapping_module.map_shape)
        self.policy = FusionMapPolicy(self.config, self.mapping_module.map_shape[0])
        self.policy.reset()
        
    def _concat_obs(self, obs: Observations) -> np.ndarray:
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
        
        """
        ds: Downscaling factor
        args.env_frame_width = 640, args.frame_width = 160
        """
        ds = env_frame_width // self.map_args.FRAME_WIDTH # ds = 4
        if ds != 1:
            rgb = np.asarray(self.trans(rgb.astype(np.uint8))) # resize
            depth = depth[ds // 2::ds, ds // 2::ds] # down scaling start from 2, step=4
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2) # recover depth.shape to (height, width, 1)
        state = np.concatenate((rgb, depth, sem_seg_pred),axis=2).transpose(2, 0, 1) # (4+num_detected_classes, h, w)
        
        return state
        
    def _get_sem_pred(self, rgb: np.ndarray) -> np.ndarray:
        """
        mask.shape=[num_detected_classes, h, w]
        labels looks like: ["kitchen counter 0.69", "floor 0.37"]
        """
        masks, labels, annotated_images, self.current_detections = \
            self.segment_module.segment(rgb, classes=self.classes)
        self.mapping_module.rgb_vis = annotated_images
        assert len(masks) == len(labels), f"The number of masks not equal to the number of labels!"
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
        if masks.shape != (0,):
            same_label_indexs = defaultdict(list)
            for idx, item in enumerate(labels):
                same_label_indexs[item].append(idx) #dict {class name: [idx]}
            combined_mask = np.zeros((len(same_label_indexs), *masks.shape[1:]))
            for i, indexs in enumerate(same_label_indexs.values()):
                combined_mask[i] = np.sum(masks[indexs, ...], axis=0)
            
            idx = [self.detected_classes.index(label) for label in same_label_indexs.keys()]
            
            """
            max_idx = max(idx) + 1, attention: remember to add one becaure index start from 0
            init final masks as [max_idx + 1, h, w]; add not_a_category channel at last
            """
            final_masks = np.zeros((len(self.detected_classes), *masks.shape[1:]))
            final_masks[idx, ...] = combined_mask
        else:
            final_masks = np.zeros((len(self.detected_classes), self.height, self.width))
        
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
        
        return state # state.shape=(c,h,w)
    
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

    def _preprocess_sub_instruction(self, obs: Observations) -> Any:
        pass
    
    def _random_policy(self):
        action = np.random.choice([
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ])
        
        return {"action": action}

    def _process_llm_reply(self, obs: Observations):
        llm_reply = obs['llm_reply']
        destination = llm_reply['destination']
        sub_instructions = llm_reply['sub-instructions']
        sub_constraints = llm_reply['state-constraints']
        decisions = llm_reply['decisions']
        
    
    def _maps_initialization(self):
        obs = self.envs.reset() #type(obs): list
        batch_obs = self._batch_obs(obs)
        self.current_episode_id = self.envs.current_episodes()[0].episode_id
        print("current episode id: ", self.current_episode_id)
        
        self.mapping_module.init_map_and_pose(num_detected_classes=len(self.detected_classes))
        poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
        self.mapping_module(batch_obs, poses)
        full_map, full_pose, _, _ = self.mapping_module.update_map(0, self.detected_classes, self.current_episode_id)
        self.mapping_module.one_step_full_map.fill_(0.)
        self.mapping_module.one_step_local_map.fill_(0.)
        
        blip_value = self.value_map_module.get_blip_value(Image.fromarray(obs[0]['rgb']), 
                                                   "The Giraffi in the living room.")
        blip_value = blip_value.detach().cpu().numpy()
        self.value_map_module(0, full_map[0], blip_value, full_pose[0], self.current_episode_id)
        
    def _look_around(self):
        print("\n========== LOOK AROUND ==========\n")
        for step in range(0, 11):
            actions = []
            for _ in range(self.config.NUM_ENVIRONMENTS):
                actions.append({"action": HabitatSimActions.TURN_LEFT})
            outputs = self.envs.step(actions)
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]
            import pdb;pdb.set_trace()
            batch_obs = self._batch_obs(obs)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            self.mapping_module(batch_obs, poses)
            full_map, full_pose, frontiers, one_step_full_map = \
                self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
            
            blip_value = self.value_map_module.get_blip_value(Image.fromarray(obs[0]['rgb']), 
                                                       "go through exercise room, then find the giraffi in the living room.")
            blip_value = blip_value.detach().cpu().numpy()
            self.value_map_module(step, full_map[0], blip_value, full_pose[0], self.current_episode_id)
            # torch.save(self.value_map_module.value_map[1], 
            #            "/data/ckh/Zero-Shot-VLN-FusionMap/tests/value_maps/value_map%d.pt"%step)
            self._action = self.policy(self.value_map_module.value_map[1], 
                                       full_map[0], full_pose[0], frontiers, 
                                       self.classes[0], self.classes, self.detected_classes, 
                                       one_step_full_map[0], self.current_detections, self.current_episode_id, step)
        # import pdb;pdb.set_trace()
    
    def _use_keyboard_control(self):
        a = input("action:")
        if a == 'w':
           return {"action": HabitatSimActions.MOVE_FORWARD}
        elif a == 'a':
            return {"action": HabitatSimActions.TURN_LEFT}
        elif a == 'd':
            return {"action": HabitatSimActions.TURN_RIGHT}
        else:
            return {"action": HabitatSimActions.STOP}
    
    def reset(self) -> None:
        self.mapping_module.reset()
        self.value_map_module.reset()
        self.policy.reset()
    
    def rollout(self):
        """
        execute a whole episode which consists of a sequence of sub-steps
        """
        self._maps_initialization()
        self._look_around()
        print("\n ========== START TO NAVIGATE ==========\n")
        
        for step in range(11, self.max_step):
            actions = []
            for _ in range(self.config.NUM_ENVIRONMENTS):
                if self.keyboard_control:
                    actions.append(self._use_keyboard_control())
                else:
                    actions.append(self._action)
            outputs = self.envs.step(actions)
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]
            # import pdb;pdb.set_trace()
            if dones[0]:
                self._calculate_metric(infos)
                break
            batch_obs = self._batch_obs(obs)
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            self.mapping_module(batch_obs, poses)
            full_map, full_pose, frontiers, one_step_full_map = \
                self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
            
            blip_value = self.value_map_module.get_blip_value(Image.fromarray(obs[0]['rgb']), 
                                                       "The Giraffi in the living room.")
            blip_value = blip_value.detach().cpu().numpy()
            self.value_map_module(step, full_map[0], blip_value, full_pose[0], self.current_episode_id)
            # torch.save(self.value_map_module.value_map[1], 
            #            "/data/ckh/Zero-Shot-VLN-FusionMap/tests/value_maps/value_map%d.pt"%step)
            self._action = self.policy(self.value_map_module.value_map[1], 
                                       full_map[0], full_pose[0], frontiers, 
                                       self.classes[0], self.classes, self.detected_classes, 
                                       one_step_full_map[0], self.current_detections, self.current_episode_id, step)
    
    def eval(self):
        self._set_eval_config()
        self._init_envs()
        self._collect_val_traj()
        self._initialize_policy()
        
        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
            
        self.state_eps = {}
        aggregated_states = {}
        
        while len(self.state_eps) < eps_to_eval:
            self.rollout()
            self.reset()
        self.envs.close()
        
        if self.world_size > 1:
            distr.barrier()
            
        num_episodes = len(self.state_eps)
        for stat_key in next(iter(self.state_eps.values())).keys():
            aggregated_states[stat_key] = (
                sum(v[stat_key] for v in self.state_eps.values()) / num_episodes
            )
            
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k,v in aggregated_states.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_states[k] = v
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR, 
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR, f"stats_ckpt_{split}.json")
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)

            logger.info(f"Episodes evaluated: {total}")