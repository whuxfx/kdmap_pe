# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import List, Optional

import mmengine
import numpy as np
from mmengine.fileio import get_local_path

from mmdet3d.datasets.transforms import data_augment_utils
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.ops import box_np_ops
# from mmdet3d.datasets.transforms import DataBaseSampler, data_augment_utils

class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str, optional): The category of samples. Defaults to None.
        epoch (int, optional): Sampling epoch. Defaults to None.
        shuffle (bool): Whether to shuffle indices. Defaults to False.
        drop_reminder (bool): Drop reminder. Defaults to False.
    """

    def __init__(self,
                 sampled_list: List[dict],
                 name: Optional[str] = None,
                 epoch: Optional[int] = None,
                 shuffle: bool = True,
                 drop_reminder: bool = False) -> None:
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num: int) -> List[int]:
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self) -> None:
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num: int) -> List[dict]:
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]


@TRANSFORMS.register_module()
class DataBaseSampler(object):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str], optional): List of classes. Defaults to None.
        points_loader (dict): Config of points loader. Defaults to
            dict(type='LoadPointsFromFile', load_dim=4, use_dim=[0, 1, 2, 3]).
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 info_path: str,
                 data_root: str,
                 rate: float,
                 prepare: dict,
                 sample_groups: dict,
                 classes: Optional[List[str]] = None,
                 points_loader: dict = dict(
                     type='LoadPointsFromFile',
                     coord_type='LIDAR',
                     load_dim=4,
                     use_dim=[0, 1, 2, 3],
                     backend_args=None),
                 backend_args: Optional[dict] = None) -> None:
        super().__init__()
        self.data_root = data_root
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}
        self.points_loader = TRANSFORMS.build(points_loader)
        self.backend_args = backend_args

        # load data base infos
        with get_local_path(
                info_path, backend_args=self.backend_args) as local_path:
            # loading data from a file-like object needs file format
            db_infos = mmengine.load(open(local_path, 'rb'), file_format='pkl')

        # filter database infos
        from mmengine.logging import MMLogger
        logger: MMLogger = MMLogger.get_current_instance()
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos in DataBaseSampler')
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        logger.info('After filter database:')
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos in DataBaseSampler')

        self.db_infos = db_infos

        # load sample groups
        # TODO: more elegant way to load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos
        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)
        # TODO: No group_sampling currently

    @staticmethod
    def filter_by_difficulty(db_infos: dict, removed_difficulty: list) -> dict:
        """Filter ground truths by difficulties.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos: dict, min_gt_points_dict: dict) -> dict:
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    def sample_all(self,
                   gt_bboxes: np.ndarray,
                   gt_labels: np.ndarray,
                   img: Optional[np.ndarray] = None,
                   ground_plane: Optional[np.ndarray] = None) -> dict:
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.
            img (np.ndarray, optional): Image array. Defaults to None.
            ground_plane (np.ndarray, optional): Ground plane information.
                Defaults to None.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels
                  of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`):
                  sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(max_sample_num -
                              np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = gt_bboxes

        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_v2(class_name, sampled_num,
                                                   avoid_coll_boxes)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]['box3d_lidar'][
                            np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            # center = sampled_gt_bboxes[:, 0:3]

            # num_sampled = len(sampled)
            s_points_list = []
            count = 0
            for info in sampled:
                file_path = os.path.join(
                    self.data_root,
                    info['path']) if self.data_root else info['path']
                results = dict(lidar_points=dict(lidar_path=file_path))
                s_points = self.points_loader(results)['points']
                s_points.translate(info['box3d_lidar'][:3])

                count += 1

                s_points_list.append(s_points)

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled],
                                 dtype=np.int64)

            if ground_plane is not None:
                xyz = sampled_gt_bboxes[:, :3]
                dz = (ground_plane[:3][None, :] *
                      xyz).sum(-1) + ground_plane[3]
                sampled_gt_bboxes[:, 2] -= dz
                for i, s_points in enumerate(s_points_list):
                    s_points.tensor[:, 2].sub_(dz[i])

            ret = {
                'gt_labels_3d':
                gt_labels,
                'gt_bboxes_3d':
                sampled_gt_bboxes,
                'points':
                s_points_list[0].cat(s_points_list),
                'group_ids':
                np.arange(gt_bboxes.shape[0],
                          gt_bboxes.shape[0] + len(sampled))
            }

        return ret

    def sample_class_v2(self, name: str, num: int,
                        gt_bboxes: np.ndarray) -> List[dict]:
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6])

        sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0]:]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples
    
class DataBaseSamplerV2(DataBaseSampler):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str], optional): List of classes. Default: None.
        bbox_code_size (int, optional): The number of bbox dimensions.
            Default: None.
        points_loader(dict, optional): Config of points loader. Default:
            dict(type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
    """
    def sample_all(self, gt_info):
        """Sampling all categories of bboxes.

        Args:
            gt_info (dict): 
            
                - gt_bboxes (np.ndarray): Ground truth bounding boxes.
                - gt_labels (np.ndarray): Ground truth labels of boxes.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels \
                    of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): \
                    sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """
        gt_bboxes = gt_info["gt_bboxes_3d"]
        gt_labels = gt_info["gt_labels_3d"]

        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(
            self.sample_classes, self.sample_max_nums
        ):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(
                max_sample_num - np.sum([n == class_label for n in gt_labels])
            )
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_bboxes = []
        avoid_coll_boxes = copy.deepcopy(gt_bboxes)

        for class_name, sampled_num in zip(self.sample_classes, sample_num_per_class):
            if sampled_num > 0:
                gt_info["gt_bboxes_3d"] = avoid_coll_boxes
                sampled_cls = self.sample_class_v2(
                    class_name, sampled_num, gt_info
                )

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        sampled_gt_box = sampled_cls[0]["box3d_lidar"][np.newaxis, ...]
                    else:
                        sampled_gt_box = np.stack(
                            [s["box3d_lidar"] for s in sampled_cls], axis=0
                        )

                    sampled_gt_bboxes += [sampled_gt_box]
                    avoid_coll_boxes = np.concatenate(
                        [avoid_coll_boxes, sampled_gt_box], axis=0
                    )

        ret = None
        if len(sampled) > 0:
            sampled_gt_bboxes = np.concatenate(sampled_gt_bboxes, axis=0)
            # center = sampled_gt_bboxes[:, 0:3]

            # num_sampled = len(sampled)
            s_points_list = []
            count = 0
            for info in sampled:
                file_path = (
                    os.path.join(self.data_root, info["path"])
                    if self.data_root
                    else info["path"]
                )
                results = dict(pts_filename=file_path)
                s_points = self.points_loader(results)["points"]
                s_points.translate(info["box3d_lidar"][:3])

                count += 1

                s_points_list.append(s_points)

            gt_labels = np.array(
                [self.cat2label[s["name"]] for s in sampled], dtype=np.long
            )
            ret = {
                "gt_labels_3d": gt_labels,
                "gt_bboxes_3d": sampled_gt_bboxes,
                "points": s_points_list[0].cat(s_points_list),
                "group_ids": np.arange(
                    gt_bboxes.shape[0], gt_bboxes.shape[0] + len(sampled)
                ),
            }

        return ret

    def sample_class_v2(self, name, num, gt_info):
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_info (dict):
                - gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        gt_bboxes = gt_info["gt_bboxes_3d"]
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6]
        )
        
        if len(sampled) == 0:
            return []

        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0] :]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]
        )

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples

# 地图采样器
# @OBJECTSAMPLERS.register_module()
@TRANSFORMS.register_module(name='MapEnhancedDataBaseSampler', force=True)
class MapEnhancedDataBaseSampler(DataBaseSamplerV2):
    """Class for sampling data from the ground truth database.
        Compared to `DataBaseSampler`, `MapEnhancedDataBaseSampler` consider 
        surrounding environments and ONLY sample boxes in drivable areas.

    Args:
        info_path (str): Path of groundtruth database info.
        dataset_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str]): List of classes. Default: None.
        points_loader(dict): Config of points loader. Default: dict(
            type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
    """

    def __init__(
        self,
        info_path,
        data_root,
        rate,
        prepare,
        sample_groups,
        xbound,
        ybound,
        dataset="nuscenes",
        map_classes=None,
        classes=None,
        bbox_code_size=None,
        points_loader=dict(
            type="LoadPointsFromFile",
            coord_type="LIDAR",
            load_dim=4,
            use_dim=[0, 1, 2, 3],
            backend_args=dict(backend='disk')  
        ),
    ):
        super().__init__(info_path, data_root, rate, prepare, 
            sample_groups, classes, points_loader) # , bbox_code_size not supported by base class!!
        self.xbound = xbound
        self.ybound = ybound

        self.dataset = dataset
        self.map_classes = {cls: i for i, cls in enumerate(map_classes)}
        if dataset.lower() == "nuscenes":
            assert "drivable_area" in map_classes, "`map_classes` must include 'drivable_area'."
            self.vehicle_names = ["car", "truck", "construction_vehicle", "bus", "trailer"]
        elif dataset.lower() == "lyft":
            self.vehicle_names = ['car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle']
        else:
            raise NotImplementedError("Only surpport NuScenes and Lyft dataset now.")

    def sample_class_v2(self, name, num, gt_info):
        gt_bboxes = gt_info["gt_bboxes_3d"]
        # gt_labels = gt_info["gt_labels_3d"]
        masks_bev = gt_info["map_mask"]
        assert masks_bev is not None, "Map mask should be loaded first."

        drivable_area_vechiles = (masks_bev[self.map_classes["drivable_area"]] >= 1)
        drivable_area_others = (masks_bev.sum(0) >= 1)
        # if self.dataset.lower() == "nuscenes":
        #     drivable_area_vechiles = (masks_bev[self.map_classes["drivable_area"]] >= 1)
        #     drivable_area_others = (masks_bev.sum(0) >= 1)
        # elif self.dataset.lower() == "lyft":
        #     drivable_area_vechiles = (masks_bev[self.map_classes["drivable_area"]] >= 1)
        #     drivable_area_others = (masks_bev.sum(0) >= 1)
        # else:
        #     raise NotImplementedError("Only surpport NuScenes and Lyft dataset now.")

        sampled = []
        num_left = num
        counter = 0
        while len(sampled) < num and counter < 10:
            candidate_sampled = self.sampler_dict[name].sample(num_left)
            for sample in candidate_sampled:
                # construct drivable area
                if sample["name"] in self.vehicle_names:
                    drivable_area = drivable_area_vechiles
                else:
                    drivable_area = drivable_area_others
                
                # check whether box in drivable area
                box_loc2d = copy.deepcopy(sample["box3d_lidar"][:2])
                loc_x = int((box_loc2d[0] - self.xbound[0]) / self.xbound[-1])
                loc_y = int((box_loc2d[1] - self.ybound[0]) / self.ybound[-1])
                if loc_x < 0 or loc_x >= drivable_area.shape[-1] or \
                   loc_y < 0 or loc_y >= drivable_area.shape[0]:
                   continue
                if drivable_area[loc_x, loc_y]:
                    sampled.append(sample)
                    
            num_left = num - len(sampled)
            counter += 1
        
        sampled = copy.deepcopy(sampled)
        num_gt = gt_bboxes.shape[0]
        num_sampled = len(sampled)
        gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
            gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6]
        )

        if len(sampled) == 0:
            return []
            
        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

        sp_boxes_new = boxes[gt_bboxes.shape[0] :]
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(
            sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6]
        )

        total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
        coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples
