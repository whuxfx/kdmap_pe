# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional

import mmcv
import numpy as np
from mmengine.fileio import get

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS
from lyft_dataset_sdk.utils.map_mask import MapMask

import os.path as osp
from tkinter import N
from typing import Any, Dict, Tuple, Optional, Union
from PIL import Image
from pathlib import Path


from mmcv.transforms import to_tensor
from mmdet3d.structures import BaseInstance3DBoxes
from mmdet3d.structures.points import BasePoints

import functools
from typing import Callable, Type, Union

import numpy as np
import torch

def assert_tensor_type(func: Callable) -> Callable:

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not isinstance(args[0].data, torch.Tensor):
            raise AttributeError(
                f'{args[0].__class__.__name__} has no attribute '
                f'{func.__name__} for type {args[0].datatype}')
        return func(*args, **kwargs)

    return wrapper


class DataContainer:
    """A container for any type of objects.

    Typically tensors will be stacked in the collate function and sliced along
    some dimension in the scatter function. This behavior has some limitations.
    1. All tensors have to be the same size.
    2. Types are limited (numpy array or Tensor).

    We design `DataContainer` and `MMDataParallel` to overcome these
    limitations. The behavior can be either of the following.

    - copy to GPU, pad all tensors to the same size and stack them
    - copy to GPU without stacking
    - leave the objects as is and pass it to the model
    - pad_dims specifies the number of last few dimensions to do padding
    """

    def __init__(self,
                 data: Union[torch.Tensor, np.ndarray],
                 stack: bool = False,
                 padding_value: int = 0,
                 cpu_only: bool = False,
                 pad_dims: int = 2):
        self._data = data
        self._cpu_only = cpu_only
        self._stack = stack
        self._padding_value = padding_value
        assert pad_dims in [None, 1, 2, 3]
        self._pad_dims = pad_dims

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.data)})'

    def __len__(self) -> int:
        return len(self._data)

    @property
    def data(self) -> Union[torch.Tensor, np.ndarray]:
        return self._data

    @property
    def datatype(self) -> Union[Type, str]:
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self) -> bool:
        return self._cpu_only

    @property
    def stack(self) -> bool:
        return self._stack

    @property
    def padding_value(self) -> int:
        return self._padding_value

    @property
    def pad_dims(self) -> int:
        return self._pad_dims

    @assert_tensor_type
    def size(self, *args, **kwargs) -> torch.Size:
        return self.data.size(*args, **kwargs)

    @assert_tensor_type
    def dim(self) -> int:
        return self.data.dim()


@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # TODO: consider split the multi-sweep part out of this pipeline
        # Derive the mask and transform for loading of multi-sweep data
        if self.num_ref_frames > 0:
            # init choice with the current frame
            init_choice = np.array([0], dtype=np.int64)
            num_frames = len(results['img_filename']) // self.num_views - 1
            if num_frames == 0:  # no previous frame, then copy cur frames
                choices = np.random.choice(
                    1, self.num_ref_frames, replace=True)
            elif num_frames >= self.num_ref_frames:
                # NOTE: suppose the info is saved following the order
                # from latest to earlier frames
                if self.test_mode:
                    choices = np.arange(num_frames - self.num_ref_frames,
                                        num_frames) + 1
                # NOTE: +1 is for selecting previous frames
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=False) + 1
            elif num_frames > 0 and num_frames < self.num_ref_frames:
                if self.test_mode:
                    base_choices = np.arange(num_frames) + 1
                    random_choices = np.random.choice(
                        num_frames,
                        self.num_ref_frames - num_frames,
                        replace=True) + 1
                    choices = np.concatenate([base_choices, random_choices])
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=True) + 1
            else:
                raise NotImplementedError
            choices = np.concatenate([init_choice, choices])
            select_filename = []
            for choice in choices:
                select_filename += results['img_filename'][choice *
                                                           self.num_views:
                                                           (choice + 1) *
                                                           self.num_views]
            results['img_filename'] = select_filename
            for key in ['cam2img', 'lidar2cam']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += results[key][choice *
                                                       self.num_views:(choice +
                                                                       1) *
                                                       self.num_views]
                    results[key] = select_results
            for key in ['ego2global']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += [results[key][choice]]
                    results[key] = select_results
            # Transform lidar2cam to
            # [cur_lidar]2[prev_img] and [cur_lidar]2[prev_cam]
            for key in ['lidar2cam']:
                if key in results:
                    # only change matrices of previous frames
                    for choice_idx in range(1, len(choices)):
                        pad_prev_ego2global = np.eye(4)
                        prev_ego2global = results['ego2global'][choice_idx]
                        pad_prev_ego2global[:prev_ego2global.
                                            shape[0], :prev_ego2global.
                                            shape[1]] = prev_ego2global
                        pad_cur_ego2global = np.eye(4)
                        cur_ego2global = results['ego2global'][0]
                        pad_cur_ego2global[:cur_ego2global.
                                           shape[0], :cur_ego2global.
                                           shape[1]] = cur_ego2global
                        cur2prev = np.linalg.inv(pad_prev_ego2global).dot(
                            pad_cur_ego2global)
                        for result_idx in range(choice_idx * self.num_views,
                                                (choice_idx + 1) *
                                                self.num_views):
                            results[key][result_idx] = \
                                results[key][result_idx].dot(cur2prev)
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        filename, cam2img, lidar2cam, cam2lidar, lidar2img = [], [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])

            lidar2cam_array = np.array(cam_item['lidar2cam']).astype(
                np.float32)
            lidar2cam_rot = lidar2cam_array[:3, :3]
            lidar2cam_trans = lidar2cam_array[:3, 3:4]
            camera2lidar = np.eye(4)
            camera2lidar[:3, :3] = lidar2cam_rot.T
            camera2lidar[:3, 3:4] = -1 * np.matmul(
                lidar2cam_rot.T, lidar2cam_trans.reshape(3, 1))
            cam2lidar.append(camera2lidar)

            cam2img_array = np.eye(4).astype(np.float32)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img']).astype(
                np.float32)
            cam2img.append(cam2img_array)
            lidar2img.append(cam2img_array @ lidar2cam_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['cam2lidar'] = np.stack(cam2lidar, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        imgs = [
            mmcv.imfrombytes(
                img_byte,
                flag=self.color_type,
                backend='pillow',
                channel_order='rgb') for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results
    
@TRANSFORMS.register_module()
class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = DataContainer(to_tensor(imgs), stack=True)
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = DataContainer(to_tensor(img), stack=True)
        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers2d', 'depths'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DataContainer([to_tensor(res) for res in results[key]])
            else:
                results[key] = DataContainer(to_tensor(results[key]))
        if 'gt_bboxes_3d' in results:
            if isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = DataContainer(
                    results['gt_bboxes_3d'], cpu_only=True)
            else:
                results['gt_bboxes_3d'] = DataContainer(
                    to_tensor(results['gt_bboxes_3d']))

        if 'gt_masks' in results:
            results['gt_masks'] = DataContainer(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DataContainer(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__

@TRANSFORMS.register_module()
class DefaultFormatBundle3D(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        super(DefaultFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = DataContainer(results['points'].tensor)

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = DataContainer(to_tensor(results[key]), stack=False)

        if self.with_gt:
            # Clean GT bboxes in the final
            if 'gt_bboxes_3d_mask' in results:
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][
                    gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][
                        gt_bboxes_3d_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][
                        gt_bboxes_3d_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                 dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                                                    dtype=np.int64)
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n)
                        for n in results['gt_names_3d']
                    ],
                                                       dtype=np.int64)
        results = super(DefaultFormatBundle3D, self).__call__(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str
    
# @PIPELINES.register_module(name = "DefaultFormatBundle3D", force=True)
@TRANSFORMS.register_module(name = "DefaultFormatBundle3D", force=True)
class DefaultFormatBundle3DMap(DefaultFormatBundle3D):
    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = DataContainer(results['points'].tensor)

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = DataContainer(to_tensor(results[key]), stack=False)

        if "map_mask" in results:
            results['map_mask'] = DataContainer(to_tensor(results['map_mask'].copy()))

        if self.with_gt:
            # Clean GT bboxes in the final
            if 'gt_bboxes_3d_mask' in results:
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][
                    gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][
                        gt_bboxes_3d_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][
                        gt_bboxes_3d_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                 dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                                                    dtype=np.int64)
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n)
                        for n in results['gt_names_3d']
                    ],
                                                       dtype=np.int64)
        results = super(DefaultFormatBundle3D, self).__call__(results)
        return results


@TRANSFORMS.register_module()
class LoadMapMask:
    """Load BEV map mask

    Args:
        data_root (str): The path of the dateset root directory.
        xbound (Tuple(float, float, float)): xmin, xmax, map resolution in x direction
        ybound (Tuple(float, float, float)): ymin, ymax, map resolution in y direction
        classes (Tuple(str, ...)): Classes that the map contains.
        translate_noise (Union[int, float, list, None]): Translation noise(m).
        rotate_noise (Union[int, float, None]): Rotation noise(degree).
        drop_out (Union[float, None]): The probability of losing the map.
    """
    def __init__(
        self,
        data_root: str,
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        dataset: str = "nuscenes",
        classes: Tuple[str, ...] = None,
        translate_noise: Union[int, float, list, None] = None,
        rotate_noise: Union[int ,float, None] = None,
        drop_out: Union[float, None] = None
    ) -> None:
        super().__init__()
        patch_h = ybound[1] - ybound[0]
        patch_w = xbound[1] - xbound[0]
        self.canvas_h = int(patch_h / ybound[2])
        self.canvas_w = int(patch_w / xbound[2])
        self.xbound = xbound
        self.ybound = ybound
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (self.canvas_h, self.canvas_w)
        self.dataset = dataset
        
        if translate_noise is not None:
            if isinstance(translate_noise, float):
                self.translate_noise = [translate_noise, translate_noise]
            elif isinstance(translate_noise, list):
                self.translate_noise = translate_noise
            else:
                raise NotImplementedError("`translate_noise` must be float or list now.")
        else:
            self.translate_noise = None
        self.rotate_noise = rotate_noise # degree

        self.drop_out = drop_out

        if self.dataset.lower() == "nuscenes":
            self.classes = classes
            self.map_channels = len(self.classes)
            self.maps = {}
            for location in LOCATIONS:
                self.maps[location] = NuScenesMap(data_root, location)
            self.mappings = {}
            for name in self.classes:
                if name == "drivable_area*":
                    self.mappings[name] = ["road_segment", "lane"]
                elif name == "divider":
                    self.mappings[name] = ["road_divider", "lane_divider"]
                else:
                    self.mappings[name] = [name]
            self.layer_names = []
            for name in self.mappings:
                self.layer_names.extend(self.mappings[name])
            self.layer_names = list(set(self.layer_names))
        elif self.dataset.lower() == "lyft":
            self.map_channels = 3
            assert xbound[2] == ybound[2], \
                    "For Lyfy Dataset, resolution of dimension x and y must be same."
            map_path = osp.join(data_root, "v1.01-train/maps", "map_raster_palo_alto.png")
            self.map_mask = MapMask(Path(map_path), xbound[2])
            self.mask_raster = self.map_mask.mask()
            self.map_layers_color = {
                "drivable_area": [128, 128, 128],
                "ped_crossing": [250, 235, 215],
                "walkway": [211, 211, 211]
            }
        else:
            raise NotImplementedError("Only surpport NuScenes and Lyft dataset now.")
        # Lyft special

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # print("✅ --------------------LoadMapMask is running!")  # 确保执行了
        data["xbound"] = self.xbound
        data["ybound"] = self.ybound
        data["translate_std"] = self.translate_noise
        data["rotate_std"] = self.rotate_noise
        
        if self.drop_out is None or np.random.uniform() > self.drop_out:
            # compute map pose
            loc, angle, data = self._getMapPose(data)

            if self.dataset.lower() == "nuscenes":
                map_mask = self._getNusMapMask(loc, angle, data["location"])
            elif self.dataset.lower() == "lyft":
                map_mask = self._getLyftMapMask(loc, angle)
            else:
                raise NotImplementedError("Only surpport nuscenes and lyft dataset now.")
        else:
            map_mask = np.zeros((self.map_channels, *self.canvas_size), dtype=np.float32)
        data["map_mask"] = map_mask
        return data


    def _getMapPose(self, data):
        """
        获取地图位姿 (Map Pose)。
        
        Args:
            data (dict): 数据字典，包含必要的位姿信息。
            
        Returns:
            tuple: (loc, angle, data)
                - loc (list): 地图位置 [x, y]。
                - angle (float): 地图朝向角度。
                - data (dict): 更新后的数据字典，包含噪声信息。
        """
        try:
            # 从 'lidar_points' 中读取 'lidar2ego'
            lidar_points = data.get("lidar_points", {})
            lidar2ego = lidar_points.get("lidar2ego", None)
            if lidar2ego is None:
                # logging.warning(f"'lidar2ego' missing in 'lidar_points'. Using identity matrix for sample {data.get('sample_idx', 'unknown')}.")
                lidar2ego = np.eye(4)
            else:
                lidar2ego = np.array(lidar2ego)

            # 从顶层读取 'ego2global'
            ego2global = data.get("ego2global", None)
            if ego2global is None:
                # logging.warning(f"'ego2global' missing in data. Using identity matrix for sample {data.get('sample_idx', 'unknown')}.")
                ego2global = np.eye(4)
            else:
                ego2global = np.array(ego2global)

            # 处理可能存在的 'lidar_aug_matrix'
            lidar2point = data.get("lidar_aug_matrix", None)
            if lidar2point is not None:
                try:
                    lidar2point = np.array(lidar2point)
                    point2lidar = np.linalg.inv(lidar2point)
                    lidar2global = ego2global @ lidar2ego @ point2lidar
                except np.linalg.LinAlgError:
                    # logging.error(f"Cannot invert 'lidar_aug_matrix' for sample {data.get('sample_idx', 'unknown')}. Using ego2global @ lidar2ego.")
                    lidar2global = ego2global @ lidar2ego
            else:
                lidar2global = ego2global @ lidar2ego

            # 计算地图位置
            map_pose = lidar2global[:2, 3]

            # 添加平移噪声
            if self.translate_noise is not None:
                tran_noise0 = np.random.normal(0, self.translate_noise[0])
                tran_noise1 = np.random.normal(0, self.translate_noise[1])
                map_pose[0] += tran_noise0
                map_pose[1] += tran_noise1
                data["translate_noise"] = [tran_noise0, tran_noise1]
                # logging.debug(f"Added translate noise: {data['translate_noise']} for sample {data.get('sample_idx', 'unknown')}.")

            loc = [map_pose[0], map_pose[1]]

            # 计算地图朝向角度
            rotation = lidar2global[:3, :3]
            v = np.dot(rotation, np.array([1, 0, 0]))
            yaw = np.arctan2(v[1], v[0])
            angle = yaw / np.pi * 180

            # 添加旋转噪声
            if self.rotate_noise is not None:
                rotate_noise = np.random.normal(0, self.rotate_noise)
                angle += rotate_noise
                data["rotate_noise"] = rotate_noise
                # logging.debug(f"Added rotate noise: {data['rotate_noise']} for sample {data.get('sample_idx', 'unknown')}.")

            return loc, angle, data

        except KeyError as e:
            # logging.error(f"Missing key {e} in data: {data}")
            # 根据需求返回默认值或引发异常
            return [0.0, 0.0], 0.0, data
        except Exception as e:
            # logging.error(f"Error in _getMapPose: {e}")
            # 根据需求返回默认值或引发异常
            return [0.0, 0.0], 0.0, data

    def _getNusMapMask(self, loc, angle, location):
        """Get map mask in nuscenes dataset
        
        Args:
            loc (list[float]): global location of the lidar. (x,y)
            angle (float): yaw angle (degree)
            location (str): The location of current sample.
        """
        map_mask = np.zeros((self.map_channels, *self.canvas_size), dtype=np.float32)
        patch_box = (loc[0], loc[1], self.patch_size[0], self.patch_size[1])
        masks = self.maps[location].get_map_mask(
            patch_box=patch_box,
            patch_angle=angle,
            layer_names=self.layer_names,
            canvas_size=self.canvas_size,
        )
        # masks = masks[:, ::-1, :].copy()
        masks = masks.transpose(0, 2, 1)
        # masks = masks.astype(np.bool)
        masks = masks.astype(bool)


        for k, name in enumerate(self.classes):
            for layer_name in self.mappings[name]:
                index = self.layer_names.index(layer_name)
                map_mask[k, masks[index]] = 1
        return map_mask

    def _getLyftMapMask(self, loc, angle):
        """Get map mask in nuscenes dataset
        
        Args:
            loc (list[float]): global location of the lidar. (x,y)
            angle (float): yaw angle (degree)
        """
        def crop_image(image: np.array,
                       pixel_coords: list,
                       size: int) -> np.array:
            x, y = pixel_coords
            x_min = int(x - size//2)
            x_max = int(x + size//2)
            y_min = int(y - size//2)
            y_max = int(y + size//2)
            cropped_image = image[y_min:y_max, x_min:x_max]
            return cropped_image
        
        pixel_coords = self.map_mask.to_pixel_coords(loc[0], loc[1])
        cropped = crop_image(self.mask_raster, pixel_coords, int(self.canvas_size[0] * np.sqrt(2)))

        yaw_deg = -angle
        rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))
        center = [rotated_cropped.shape[1] / 2, rotated_cropped.shape[0] / 2]
        ego_centric_map = crop_image(rotated_cropped, center, self.canvas_size[0])[::-1]
        ego_centric_map = np.transpose(ego_centric_map, (2, 1, 0))
        ego_centric_map = self._rgb2binary(ego_centric_map)
        return ego_centric_map

    def _rgb2binary(self, map_rgb):
        map_layers = {}
        for layer, color in self.map_layers_color.items():
            map_layers[layer] = map_rgb == np.array(color).reshape(3,1,1)
            map_layers[layer] = np.all(map_layers[layer], axis=0)
        map_mask_binary = np.stack(map_layers.values())
        return map_mask_binary