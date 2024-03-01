import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import vlnce_baselines.utils.depth_utils as du
from vlnce_baselines.utils.map_utils import get_grid, ChannelPool


class Semantic_Mapping(nn.Module):

    """
    Semantic_Mapping
    """

    def __init__(self, args):
        super(Semantic_Mapping, self).__init__()

        self.device = args.DEVICE
        self.screen_h = args.FRAME_HEIGHT
        self.screen_w = args.FRAME_WIDTH
        self.resolution = args.MAP_RESOLUTION
        self.z_resolution = args.MAP_RESOLUTION
        self.map_size_cm = args.MAP_SIZE_CM // args.GLOBAL_DOWNSCALING
        self.n_channels = 3
        self.vision_range = args.VISION_RANGE # args.vision_range=100(cm)
        self.dropout = 0.5
        self.fov = args.HFOV
        self.du_scale = args.DU_SCALE # depth unit
        self.cat_pred_threshold = args.CAT_PRED_THRESHOLD
        self.exp_pred_threshold = args.EXP_PRED_THRESHOLD
        self.map_pred_threshold = args.MAP_PRED_THRESHOLD
        self.max_sem_categories = args.MAX_SEM_CATEGORIES # presuppose max sem categories since we are handling the open-vocabulary problem

        # 72; 3.6m is about the height of one floor
        self.max_height = int(360 / self.z_resolution)
        
        # -8; we can use negative height to ensure information on the floor is contained
        self.min_height = int(-40 / self.z_resolution)
        self.agent_height = args.AGENT_HEIGHT * 100. # 0.88 * 100 = 88cm
        self.shift_loc = [self.vision_range *
                          self.resolution // 2, 0, np.pi / 2.0] # [250, 0, pi/2]
        self.camera_matrix = du.get_camera_matrix(
            self.screen_w, self.screen_h, self.fov)

        self.pool = ChannelPool(1)

        vr = self.vision_range

        # both init_grid and feat are added one dimensoin for obstacle map;
        # and feat[...,0,...] are initialized as all ones.
        # feat is predicted by mask-rcnn, which has only 1 or 0
        self.init_grid = torch.zeros(
            args.NUM_ENVIRONMENTS, 1 + self.max_sem_categories, vr, vr,
            self.max_height - self.min_height
        ).float().to(self.device)
        self.feat = torch.ones(
            args.NUM_ENVIRONMENTS, 1 + self.max_sem_categories,
            self.screen_h // self.du_scale * self.screen_w // self.du_scale
        ).float().to(self.device)

    def forward(self, obs, pose_obs, maps_last, poses_last):
        # (c, h, w), c = 3(RGB) + 1(Depth) + num_detected_categories
        # if use CoCo the number of categories is 16, but now open-vocabulary; 
        # h = 120; w = 160
        bs, c, h, w = obs.size()
        depth = obs[:, 3, :, :] # depth.shape = (bs, H, W)
        
        # cut out the needed tensor from presupposed categories dimension
        num_detected_categories = c - 4 # 4=3(RGB) + 1(Depth)
        self.init_grid = self.init_grid[:, :1 + num_detected_categories, :, :, :]
        self.feat = self.feat[:, :1 + num_detected_categories, :]

        point_cloud_t = du.get_point_cloud_from_z_t(
            depth, self.camera_matrix, self.device, scale=self.du_scale) # shape: [bs, h, w, 3] 3 is (x, y, z) for each point in (h, w)

        agent_view_t = du.transform_camera_view_t(
            point_cloud_t, self.agent_height, 0, self.device) # elevation = 0, only use forward obs; shape: [bs, h, w, 3]

        # point cloud in world axis
        # self.shift_loc=[250, 0, pi/2] => heading is always 90(degree), change with turn left
        agent_view_centered_t = du.transform_pose_t(
            agent_view_t, self.shift_loc, self.device) # shape: [bs, h, w, 3] => (bs, 120, 160, 3)

        max_h = self.max_height # 72
        min_h = self.min_height # -8
        xy_resolution = self.resolution
        z_resolution = self.z_resolution
        
        # vision_range = 100(cm)
        # in sem_exp.py _preprocess_depth(), all invalid depth values are set as 100 
        vision_range = self.vision_range
        XYZ_cm_std = agent_view_centered_t.float() # (bs, x, y, 3) => (bs, 120, 160, 3)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] / xy_resolution)
        XYZ_cm_std[..., :2] = (XYZ_cm_std[..., :2] -
                               vision_range // 2.) / vision_range * 2. # normalize to (-1, 1)
        XYZ_cm_std[..., 2] = XYZ_cm_std[..., 2] / z_resolution
        XYZ_cm_std[..., 2] = (XYZ_cm_std[..., 2] -
                              (max_h + min_h) // 2.) / (max_h - min_h) * 2. # normalize
        self.feat[:, 1:, :] = nn.AvgPool2d(self.du_scale)(
            obs[:, 4:, :, :] # obs: [b, c, h*w] => [b, 17, 19200], feat is a tensor contains all predicted semantic features
        ).view(bs, c - 4, h // self.du_scale * w // self.du_scale)

        XYZ_cm_std = XYZ_cm_std.permute(0, 3, 1, 2)
        XYZ_cm_std = XYZ_cm_std.view(XYZ_cm_std.shape[0],
                                     XYZ_cm_std.shape[1],
                                     XYZ_cm_std.shape[2] * XYZ_cm_std.shape[3]) # [bs, 3, x*y]

        # self.init_grid: [bs, categories + 1, x=vr, y=vr, z=(max_height - min_height)] => [bs, 17, 100, 100, 80]
        # feat: average of all categories's predicted semantic features, [bs, 17, 19200]
        # XYZ_cm_std: point cloud in physical world, [bs, 3, 19200]
        # splat_feat_nd:
        voxels = du.splat_feat_nd(
            self.init_grid * 0., self.feat, XYZ_cm_std).transpose(2, 3) # shape: [bs, 17, 100, 100, 80]

        min_z = int(25 / z_resolution - min_h) # 25 / 5 - (-8) = 13
        max_z = int((self.agent_height + 1) / z_resolution - min_h) # int((88 + 1) / 5 - (-8))= 25

        agent_height_proj = voxels[..., min_z:max_z].sum(4) # shape: [bs, 17, 100, 100]
        all_height_proj = voxels.sum(4) # shape: [bs, 17, 100, 100]

        fp_map_pred = agent_height_proj[:, 0:1, :, :] # obstacle map
        fp_exp_pred = all_height_proj[:, 0:1, :, :] # explored map
        fp_map_pred = fp_map_pred / self.map_pred_threshold
        fp_exp_pred = fp_exp_pred / self.exp_pred_threshold
        fp_map_pred = torch.clamp(fp_map_pred, min=0.0, max=1.0)
        fp_exp_pred = torch.clamp(fp_exp_pred, min=0.0, max=1.0)

        pose_pred = poses_last

        agent_view = torch.zeros(bs, c,
                                 self.map_size_cm // self.resolution,
                                 self.map_size_cm // self.resolution
                                 ).to(self.device) # (bs, c, 480, 480) => full_map

        x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
        x2 = x1 + self.vision_range
        y1 = self.map_size_cm // (self.resolution * 2)
        y2 = y1 + self.vision_range
        agent_view[:, 0:1, y1:y2, x1:x2] = fp_map_pred # obstacle map
        agent_view[:, 1:2, y1:y2, x1:x2] = fp_exp_pred # explored area
        agent_view[:, 4:, y1:y2, x1:x2] = torch.clamp(
            agent_height_proj[:, 1:, :, :] / self.cat_pred_threshold,
            min=0.0, max=1.0) # semantic categories

        corrected_pose = pose_obs # sensor pose

        def get_new_pose_batch(pose, rel_pose_change):
            # pose: (bs, 3) -> x, y, ori(degree)
            # 57.29577951308232 = 180 / pi
            pose[:, 1] += rel_pose_change[:, 0] * \
                torch.sin(pose[:, 2] / 57.29577951308232) \
                + rel_pose_change[:, 1] * \
                torch.cos(pose[:, 2] / 57.29577951308232)
            pose[:, 0] += rel_pose_change[:, 0] * \
                torch.cos(pose[:, 2] / 57.29577951308232) \
                - rel_pose_change[:, 1] * \
                torch.sin(pose[:, 2] / 57.29577951308232)
            pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

            pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
            pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

            return pose

        current_poses = get_new_pose_batch(poses_last, corrected_pose)
        st_pose = current_poses.clone().detach()

        st_pose[:, :2] = - (st_pose[:, :2]
                            * 100.0 / self.resolution
                            - self.map_size_cm // (self.resolution * 2)) /\
            (self.map_size_cm // (self.resolution * 2))
        st_pose[:, 2] = 90. - (st_pose[:, 2])

        # get rotation matrix and translation matrix according to new pose (x, y, theta(degree))
        rot_mat, trans_mat = get_grid(st_pose, agent_view.size(),
                                      self.device)

        rotated = F.grid_sample(agent_view, rot_mat, align_corners=True)
        translated = F.grid_sample(rotated, trans_mat, align_corners=True)

        maps2 = torch.cat((maps_last.unsqueeze(1), translated.unsqueeze(1)), 1)

        map_pred, _ = torch.max(maps2, 1)

        return fp_map_pred, map_pred, pose_pred, current_poses
