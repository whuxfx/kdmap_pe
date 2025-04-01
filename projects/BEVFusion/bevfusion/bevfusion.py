from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization
from collections import Sequence

from .diffkd import DiffKD
# from torch.nn import functional as F, Conv2d


def gather_map(data):
    if isinstance(data["map_mask"], Sequence):
        map = torch.stack(data["map_mask"], dim=0)
    else:
        map = data["map_mask"]
    return map.float()

@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        map_feat_encoder: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        **kwargs,
    ) -> None:
        # === 安全处理 voxelize_cfg ===
        voxelize_cfg = None
        if data_preprocessor is not None and 'voxelize_cfg' in data_preprocessor:
            voxelize_cfg = data_preprocessor.pop('voxelize_cfg')

        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if voxelize_cfg is not None:
            self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce', False)
            self.pts_voxel_layer = Voxelization(**voxelize_cfg)
        else:
            self.voxelize_reduce = False
            self.pts_voxel_layer = None

        # === 点云分支 ===
        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder) if pts_voxel_encoder is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder) if pts_middle_encoder is not None else None
        self.pts_backbone = MODELS.build(pts_backbone) if pts_backbone is not None else None
        self.pts_neck = MODELS.build(pts_neck) if pts_neck is not None else None

        # === 图像分支 ===
        self.img_backbone = MODELS.build(img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(view_transform) if view_transform is not None else None

        # === 地图分支 ===
        self.map_feat_encoder = MODELS.build(map_feat_encoder) if map_feat_encoder is not None else None

        # === 蒸馏模块 DiffKD ===
        self.distill_cfg = kwargs.get("distill_cfg", {})
        self.diffkd_map_img = None
        self.diffkd_map_pts = None

        if self.distill_cfg.get("enable_img_distill", False):
            self.diffkd_map_img = DiffKD(
                student_channels=80,
                teacher_channels=64,
                inference_steps=self.distill_cfg.get("inference_steps", 5),
                num_train_timesteps=self.distill_cfg.get("num_train_timesteps", 1000),
                distill_cfg=self.distill_cfg,
            )

        if self.distill_cfg.get("enable_pts_distill", False):
            self.diffkd_map_pts = DiffKD(
                student_channels=256,
                teacher_channels=64,
                inference_steps=self.distill_cfg.get("inference_steps", 5),
                num_train_timesteps=self.distill_cfg.get("num_train_timesteps", 1000),
                distill_cfg=self.distill_cfg,
            )

        # === 融合层和检测头 ===
        self.fusion_layer = MODELS.build(fusion_layer) if fusion_layer is not None else None
        self.bbox_head = MODELS.build(bbox_head) if bbox_head is not None else None

        self.init_weights()

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return x

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        # print("---------------------------------------------")
        # print("🚘 LiDAR BEV 特征 shape:", x.shape)

        return x

    def extract_map_feat(self, batch_inputs_dict) -> torch.Tensor:
        """提取地图特征，同时保留一份用于 query 初始化的 BEV 掩码"""
        map_mask = batch_inputs_dict.get('map_mask', None)
        
        
        x = map_mask
        if self.map_feat_encoder:
            x = self.map_feat_encoder(x)
        return x


    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        map_mask = batch_inputs_dict.get('map_mask', None)
        # ✅ 初始化三个模态变量为 None，确保后续不会因未定义变量出错

        img_feature = None
        map_feature = None
        pts_feature = None

        features = []
        diffkd_losses = {}
        

        # 1. 提取图像特征
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []

            for meta in batch_input_metas:
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))

            img_feature = self.extract_img_feat(
                imgs, deepcopy(points),
                lidar2image, camera_intrinsics,
                camera2lidar, img_aug_matrix,
                lidar_aug_matrix, batch_input_metas
            )

        # 2. 提取地图特征（教师）
        if map_mask is not None:
            map_feature = self.extract_map_feat(batch_inputs_dict)


        # 3. 提取点云特征
        # pts_feature = self.extract_pts_feat(batch_inputs_dict)

        # ============ DiffKD 蒸馏过程（map -> img 和 map -> pts） ============

        if self.training:
            # 蒸馏到图像分支
            if self.diffkd_map_img is not None and img_feature is not None:
                refined_img, ddim_loss_img, t_feat_img, rec_loss_img = self.diffkd_map_img(img_feature, map_feature.detach())
                kd_loss_img = F.mse_loss(refined_img, t_feat_img) * self.distill_cfg.get("kd_loss_weight", 1.0)

                if rec_loss_img is not None:
                    diffkd_losses["loss_diffkd_img_rec"] = rec_loss_img
                if ddim_loss_img is not None:
                    diffkd_losses["loss_diffkd_img_ddim"] = ddim_loss_img
                diffkd_losses["loss_diffkd_img_feat"] = kd_loss_img

                # 如果要融合 refined 特征
                # img_feature = img_feature + refined_img

            # 蒸馏到 LiDAR 分支
            if self.diffkd_map_pts is not None:
                refined_pts, ddim_loss_pts, t_feat_pts, rec_loss_pts = self.diffkd_map_pts(pts_feature, map_feature.detach())
                kd_loss_pts = F.mse_loss(refined_pts, t_feat_pts) * self.distill_cfg.get("kd_loss_weight", 1.0)

                if rec_loss_pts is not None:
                    diffkd_losses["loss_diffkd_pts_rec"] = rec_loss_pts
                if ddim_loss_pts is not None:
                    diffkd_losses["loss_diffkd_pts_ddim"] = ddim_loss_pts
                diffkd_losses["loss_diffkd_pts_feat"] = kd_loss_pts

                # 融合 refined 特征（如果需要）
                # pts_feature = pts_feature + refined_pts


        # ============ 三模态融合 ============
        # features = [img_feature, map_feature, pts_feature]
        features = []
        if img_feature is not None:
            # print("----------------------------")
            features.append(img_feature)
        if map_feature is not None:
            # print("-----------1111-----------------")
            features.append(map_feature)
        if pts_feature is not None:
            # print("---------------22222-------------")
            features.append(pts_feature)
        if self.fusion_layer is not None:
            # print("----------------2223333------------")
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        # x = self.pts_backbone(x)
        # x = self.pts_neck(x)

        if self.training:
            return x, diffkd_losses
        else:
            return x


    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats, diffkd_losses = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)
            losses.update(bbox_loss)

        # 加入蒸馏损失
        if self.training and diffkd_losses:
            losses.update(diffkd_losses)

        return losses

# from collections import OrderedDict
# from copy import deepcopy
# from typing import Dict, List, Optional, Tuple

# import numpy as np
# import torch
# import torch.distributed as dist
# from mmengine.utils import is_list_of
# from torch import Tensor
# from torch.nn import functional as F

# from mmdet3d.models import Base3DDetector
# from mmdet3d.registry import MODELS
# from mmdet3d.structures import Det3DDataSample
# from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
# from .ops import Voxelization
# from collections import Sequence

# from .diffkd import DiffKD
# # from torch.nn import functional as F, Conv2d


# # def gather_map(data):
# #     if isinstance(data["map_mask"], Sequence):
# #         map = torch.stack(data["map_mask"], dim=0)
# #     else:
# #         map = data["map_mask"]
# #     return map.float()

# @MODELS.register_module()
# class BEVFusion(Base3DDetector):

#     def __init__(
#         self,
#         data_preprocessor: OptConfigType = None,
#         pts_voxel_encoder: Optional[dict] = None,
#         pts_middle_encoder: Optional[dict] = None,
#         fusion_layer: Optional[dict] = None,
#         img_backbone: Optional[dict] = None,
#         pts_backbone: Optional[dict] = None,
#         view_transform: Optional[dict] = None,
#         img_neck: Optional[dict] = None,
#         pts_neck: Optional[dict] = None,
#         map_feat_encoder: Optional[dict] = None,
#         bbox_head: Optional[dict] = None,
#         init_cfg: OptMultiConfig = None,
#         seg_head: Optional[dict] = None,
#         **kwargs,
#     ) -> None:
#         voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
#         super().__init__(
#             data_preprocessor=data_preprocessor, init_cfg=init_cfg)

#         self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
#         self.pts_voxel_layer = Voxelization(**voxelize_cfg)

#         self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

#         self.img_backbone = MODELS.build(
#             img_backbone) if img_backbone is not None else None
#         self.img_neck = MODELS.build(
#             img_neck) if img_neck is not None else None
#         self.view_transform = MODELS.build(
#             view_transform) if view_transform is not None else None
#         self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

#         self.map_feat_encoder = MODELS.build(map_feat_encoder)
#         self.distill_cfg = kwargs.get("distill_cfg", {})
#         # 实例化 DiffKD 模块：只在启用时创建
#         self.diffkd_map_img = None
#         self.diffkd_map_pts = None
#         # 蒸馏模块：将 map_feat 蒸馏到 img_bev 和 pts_bev
#         if self.distill_cfg.get("enable_img_distill", False):
#             self.diffkd_map_img = DiffKD(
#                 student_channels=80,
#                 teacher_channels=64,
#                 inference_steps=self.distill_cfg.get("inference_steps", 5),
#                 num_train_timesteps=self.distill_cfg.get("num_train_timesteps", 1000),
#                 distill_cfg=self.distill_cfg,
#             )

#         if self.distill_cfg.get("enable_pts_distill", False):
#             self.diffkd_map_pts = DiffKD(
#                 student_channels=256,
#                 teacher_channels=64,
#                 inference_steps=self.distill_cfg.get("inference_steps", 5),
#                 num_train_timesteps=self.distill_cfg.get("num_train_timesteps", 1000),
#                 distill_cfg=self.distill_cfg,
#             )


#         # self.proj_img_refined = Conv2d(64, 80, 1)
#         # self.proj_pts_refined = Conv2d(64, 256, 1)

#         self.fusion_layer = MODELS.build(
#             fusion_layer) if fusion_layer is not None else None

#         self.pts_backbone = MODELS.build(pts_backbone)
#         self.pts_neck = MODELS.build(pts_neck)

#         self.bbox_head = MODELS.build(bbox_head)

#         self.init_weights()

#     def _forward(self,
#                  batch_inputs: Tensor,
#                  batch_data_samples: OptSampleList = None):
#         """Network forward process.

#         Usually includes backbone, neck and head forward without any post-
#         processing.
#         """
#         pass

#     def parse_losses(
#         self, losses: Dict[str, torch.Tensor]
#     ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         """Parses the raw outputs (losses) of the network.

#         Args:
#             losses (dict): Raw output of the network, which usually contain
#                 losses and other necessary information.

#         Returns:
#             tuple[Tensor, dict]: There are two elements. The first is the
#             loss tensor passed to optim_wrapper which may be a weighted sum
#             of all losses, and the second is log_vars which will be sent to
#             the logger.
#         """
#         log_vars = []
#         for loss_name, loss_value in losses.items():
#             if isinstance(loss_value, torch.Tensor):
#                 log_vars.append([loss_name, loss_value.mean()])
#             elif is_list_of(loss_value, torch.Tensor):
#                 log_vars.append(
#                     [loss_name,
#                      sum(_loss.mean() for _loss in loss_value)])
#             else:
#                 raise TypeError(
#                     f'{loss_name} is not a tensor or list of tensors')

#         loss = sum(value for key, value in log_vars if 'loss' in key)
#         log_vars.insert(0, ['loss', loss])
#         log_vars = OrderedDict(log_vars)  # type: ignore

#         for loss_name, loss_value in log_vars.items():
#             # reduce loss when distributed training
#             if dist.is_available() and dist.is_initialized():
#                 loss_value = loss_value.data.clone()
#                 dist.all_reduce(loss_value.div_(dist.get_world_size()))
#             log_vars[loss_name] = loss_value.item()

#         return loss, log_vars  # type: ignore

#     def init_weights(self) -> None:
#         if self.img_backbone is not None:
#             self.img_backbone.init_weights()

#     @property
#     def with_bbox_head(self):
#         """bool: Whether the detector has a box head."""
#         return hasattr(self, 'bbox_head') and self.bbox_head is not None

#     @property
#     def with_seg_head(self):
#         """bool: Whether the detector has a segmentation head.
#         """
#         return hasattr(self, 'seg_head') and self.seg_head is not None

#     def extract_img_feat(
#         self,
#         x,
#         points,
#         lidar2image,
#         camera_intrinsics,
#         camera2lidar,
#         img_aug_matrix,
#         lidar_aug_matrix,
#         img_metas,
#     ) -> torch.Tensor:
#         B, N, C, H, W = x.size()
#         x = x.view(B * N, C, H, W).contiguous()

#         x = self.img_backbone(x)
#         x = self.img_neck(x)

#         if not isinstance(x, torch.Tensor):
#             x = x[0]

#         BN, C, H, W = x.size()
#         x = x.view(B, int(BN / B), C, H, W)

#         with torch.autocast(device_type='cuda', dtype=torch.float32):
#             x = self.view_transform(
#                 x,
#                 points,
#                 lidar2image,
#                 camera_intrinsics,
#                 camera2lidar,
#                 img_aug_matrix,
#                 lidar_aug_matrix,
#                 img_metas,
#             )
#         return x

#     def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
#         points = batch_inputs_dict['points']
#         with torch.autocast('cuda', enabled=False):
#             points = [point.float() for point in points]
#             feats, coords, sizes = self.voxelize(points)
#             batch_size = coords[-1, 0] + 1
#         x = self.pts_middle_encoder(feats, coords, batch_size)
#         return x

    
#     def extract_map_feat(self, batch_inputs_dict) -> torch.Tensor:
#         """提取地图特征"""
#         # print("111111----------------------------------------")

#         # 获取 map_mask
#         map_mask = batch_inputs_dict.get('map_mask', None)

#         # **改正这里：包装成 dict 传给 `gather_map`**
#         x = map_mask
#         # print(f"✅ --------------------Using map data directly----------------, shape: {x.shape}")
#         # **使用 map_feat_encoder 提取地图特征**
#         if self.map_feat_encoder:
#             x = self.map_feat_encoder(x)
#         # print(f"✅ --------------------------Encoded map feature shape-----------------------: {x.shape}")
#         return x

    


#     @torch.no_grad()
#     def voxelize(self, points):
#         feats, coords, sizes = [], [], []
#         for k, res in enumerate(points):
#             ret = self.pts_voxel_layer(res)
#             if len(ret) == 3:
#                 # hard voxelize
#                 f, c, n = ret
#             else:
#                 assert len(ret) == 2
#                 f, c = ret
#                 n = None
#             feats.append(f)
#             coords.append(F.pad(c, (1, 0), mode='constant', value=k))
#             if n is not None:
#                 sizes.append(n)

#         feats = torch.cat(feats, dim=0)
#         coords = torch.cat(coords, dim=0)
#         if len(sizes) > 0:
#             sizes = torch.cat(sizes, dim=0)
#             if self.voxelize_reduce:
#                 feats = feats.sum(
#                     dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
#                 feats = feats.contiguous()

#         return feats, coords, sizes

#     def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
#                 batch_data_samples: List[Det3DDataSample],
#                 **kwargs) -> List[Det3DDataSample]:
#         """Forward of testing.

#         Args:
#             batch_inputs_dict (dict): The model input dict which include
#                 'points' keys.

#                 - points (list[torch.Tensor]): Point cloud of each sample.
#             batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
#                 Samples. It usually includes information such as
#                 `gt_instance_3d`.

#         Returns:
#             list[:obj:`Det3DDataSample`]: Detection results of the
#             input sample. Each Det3DDataSample usually contain
#             'pred_instances_3d'. And the ``pred_instances_3d`` usually
#             contains following keys.

#             - scores_3d (Tensor): Classification scores, has a shape
#                 (num_instances, )
#             - labels_3d (Tensor): Labels of bboxes, has a shape
#                 (num_instances, ).
#             - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
#                 contains a tensor with shape (num_instances, 7).
#         """
#         batch_input_metas = [item.metainfo for item in batch_data_samples]
#         feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

#         if self.with_bbox_head:
#             outputs = self.bbox_head.predict(feats, batch_input_metas)

#         res = self.add_pred_to_datasample(batch_data_samples, outputs)

#         return res

#     def extract_feat(
#         self,
#         batch_inputs_dict,
#         batch_input_metas,
#         **kwargs,
#     ):
#         imgs = batch_inputs_dict.get('imgs', None)
#         points = batch_inputs_dict.get('points', None)
#         map_mask = batch_inputs_dict.get('map_mask', None)

#         features = []
#         diffkd_losses = {}

#         # 1. 提取图像特征
#         if imgs is not None:
#             imgs = imgs.contiguous()
#             lidar2image, camera_intrinsics, camera2lidar = [], [], []
#             img_aug_matrix, lidar_aug_matrix = [], []

#             for meta in batch_input_metas:
#                 lidar2image.append(meta['lidar2img'])
#                 camera_intrinsics.append(meta['cam2img'])
#                 camera2lidar.append(meta['cam2lidar'])
#                 img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
#                 lidar_aug_matrix.append(meta.get('lidar_aug_matrix', np.eye(4)))

#             lidar2image = imgs.new_tensor(np.asarray(lidar2image))
#             camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
#             camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
#             img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
#             lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))

#             img_feature = self.extract_img_feat(
#                 imgs, deepcopy(points),
#                 lidar2image, camera_intrinsics,
#                 camera2lidar, img_aug_matrix,
#                 lidar_aug_matrix, batch_input_metas
#             )

#         # 2. 提取地图特征（教师）
#         if map_mask is not None:
#             map_feature = self.extract_map_feat(batch_inputs_dict)

#         # 3. 提取点云特征
#         pts_feature = self.extract_pts_feat(batch_inputs_dict)

#         # ============ DiffKD 蒸馏过程（map -> img 和 map -> pts） ============
#         if self.training:
#             # 蒸馏到图像分支
#             if self.diffkd_map_img is not None and img_feature is not None:
#                 refined_img, ddim_loss_img, t_feat_img, rec_loss_img = self.diffkd_map_img(img_feature, map_feature.detach())
#                 kd_loss_img = F.mse_loss(refined_img, t_feat_img) * self.distill_cfg.get("kd_loss_weight", 1.0)

#                 if rec_loss_img is not None:
#                     diffkd_losses["loss_diffkd_img_rec"] = rec_loss_img
#                 if ddim_loss_img is not None:
#                     diffkd_losses["loss_diffkd_img_ddim"] = ddim_loss_img
#                 diffkd_losses["loss_diffkd_img_feat"] = kd_loss_img

#                 # 如果要融合 refined 特征
#                 # img_feature = img_feature + refined_img

#             # 蒸馏到 LiDAR 分支
#             if self.diffkd_map_pts is not None:
#                 refined_pts, ddim_loss_pts, t_feat_pts, rec_loss_pts = self.diffkd_map_pts(pts_feature, map_feature.detach())
#                 kd_loss_pts = F.mse_loss(refined_pts, t_feat_pts) * self.distill_cfg.get("kd_loss_weight", 1.0)

#                 if rec_loss_pts is not None:
#                     diffkd_losses["loss_diffkd_pts_rec"] = rec_loss_pts
#                 if ddim_loss_pts is not None:
#                     diffkd_losses["loss_diffkd_pts_ddim"] = ddim_loss_pts
#                 diffkd_losses["loss_diffkd_pts_feat"] = kd_loss_pts


#         # ============ 三模态融合 ============
#         features = [img_feature, map_feature, pts_feature]
#         if self.fusion_layer is not None:
#             x = self.fusion_layer(features)
#         else:
#             assert len(features) == 1, features
#             x = features[0]

#         x = self.pts_backbone(x)
#         x = self.pts_neck(x)

#         # 如果训练模式，返回特征 + 蒸馏损失
#         if self.training:
#             return x, diffkd_losses
#         else:
#             return x


#     def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
#              batch_data_samples: List[Det3DDataSample],
#              **kwargs) -> List[Det3DDataSample]:
#         batch_input_metas = [item.metainfo for item in batch_data_samples]
#         feats, diffkd_losses = self.extract_feat(batch_inputs_dict, batch_input_metas)

#         losses = dict()
#         if self.with_bbox_head:
#             bbox_loss = self.bbox_head.loss(feats, batch_data_samples)
#             losses.update(bbox_loss)

#         # 加入蒸馏损失
#         if self.training and diffkd_losses:
#             losses.update(diffkd_losses)

#         return losses

