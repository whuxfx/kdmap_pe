# Copyright (c) Phigent Robotics. All rights reserved.
import argparse
import json
import os
import pickle

import cv2
import numpy as np
from pyquaternion import Quaternion

from mmdet3d.structures import LiDARInstance3DBoxes as LB

# python tools/analysis_tools/vis.py \
# ./exp/nus/results/bevmap-bs1-lr05-aug10/val_results_nusc.json 
# --vis-frames 6000 
# --draw-gt 
# --version val 
# --save_path ./n01kt4/viz 
# --video-prefix best-map-val-full


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    valid = np.logical_and(
        valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    return valid


def depth2color(depth):
    gray = max(0, min((depth + 2.5) / 3.0, 1.0))
    max_lumi = 200
    colors = np.array(
        [[max_lumi, 0, max_lumi], [max_lumi, 0, 0], [max_lumi, max_lumi, 0],
         [0, max_lumi, 0], [0, max_lumi, max_lumi], [0, 0, max_lumi]],
        dtype=np.float32)
    if gray == 1:
        return tuple(colors[-1].tolist())
    num_rank = len(colors) - 1
    rank = np.floor(gray * num_rank).astype(np.int)
    diff = (gray - rank / num_rank) * num_rank
    return tuple(
        (colors[rank] + (colors[rank + 1] - colors[rank]) * diff).tolist())


# def lidar2img(points_lidar, camera_info):
#     points_lidar_homogeneous = \
#         np.concatenate([points_lidar,
#                         np.ones((points_lidar.shape[0], 1),
#                                 dtype=points_lidar.dtype)], axis=1)
#     camera2lidar = np.eye(4, dtype=np.float32)
#     camera2lidar[:3, :3] = camera_info['sensor2lidar_rotation']
#     camera2lidar[:3, 3] = camera_info['sensor2lidar_translation']
#     lidar2camera = np.linalg.inv(camera2lidar)
#     points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
#     points_camera = points_camera_homogeneous[:, :3]
#     valid = np.ones((points_camera.shape[0]), dtype=bool)
#     valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
#     points_camera = points_camera / points_camera[:, 2:3]
#     camera2img = camera_info['camera_intrinsics']
#     points_img = points_camera @ camera2img.T
#     points_img = points_img[:, :2]
#     return points_img, valid
def lidar2img(points_lidar, image_info):
    """
    将 LiDAR 点云投影到图像上（使用 `images` 中的信息）。

    Args:
        points_lidar (np.ndarray): LiDAR 点云 (N, 3) 或 (N, 4)
        image_info (dict): 该视角对应的相机信息，例如 images['CAM_FRONT']

    Returns:
        points_img (np.ndarray): 投影后的图像坐标 (N, 2)
        valid (np.ndarray): 是否在前方视野 (N,)
    """
    points_lidar_homogeneous = np.concatenate(
        [points_lidar, np.ones((points_lidar.shape[0], 1), dtype=points_lidar.dtype)],
        axis=1
    )

    # ✅ 直接使用 `lidar2cam` 4x4 变换矩阵（LiDAR -> 相机）
    if 'lidar2cam' in image_info:
        lidar2camera = np.array(image_info['lidar2cam'])
    else:
        raise KeyError("Missing 'lidar2cam' in image_info")

    # 变换到相机坐标系
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]

    # 过滤掉相机后方的点
    valid = points_camera[:, -1] > 0.5

    # 归一化相机坐标
    points_camera = points_camera / points_camera[:, 2:3]

    # ✅ 获取 `cam2img` 3x3 相机内参矩阵
    if 'cam2img' in image_info:
        camera2img = np.array(image_info['cam2img'])
    else:
        raise KeyError("Missing 'cam2img' in image_info")

    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]

    return points_img, valid


def get_lidar2global(data_list):
    lidar2ego = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3] = Quaternion(data_list['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = data_list['lidar2ego_translation']
    ego2global = np.eye(4, dtype=np.float32)
    ego2global[:3, :3] = Quaternion(
        data_list['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = data_list['ego2global_translation']
    return ego2global @ lidar2ego


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the predicted '
                                     'result of nuScenes')
    parser.add_argument(
        'res', help='Path to the predicted result in json format')
    parser.add_argument(
        '--show-range',
        type=int,
        default=50,
        help='Range of visualization in BEV')
    # parser.add_argument(
    #     '--canva-size', type=int, default=1000, help='Size of canva(BEV) in pixel')
    parser.add_argument(
        '--vis-frames',
        type=int,
        default=500,
        help='Number of frames for visualization')
    parser.add_argument(
        '--scale-factor',
        type=int,
        default=4,
        help='Resize (smaller) ratio of camera images')
    parser.add_argument(
        '--vis-thred',
        type=float,
        default=0.30,
        help='Threshold the predicted results')
    parser.add_argument('--draw-gt', action='store_true')
    parser.add_argument(
        '--version',
        type=str,
        default='val',
        help='Version of nuScenes dataset')
    parser.add_argument(
        '--root_path',
        type=str,
        default='./data/nuscenes',
        help='Path to nuScenes dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./vis',
        help='Path to save visualization results')
    parser.add_argument(
        '--format',
        type=str,
        default='video',
        choices=['video', 'image'],
        help='The desired format of the visualization result')
    parser.add_argument(
        '--fps', type=int, default=20, help='Frame rate of video')
    parser.add_argument(
        '--video-prefix', type=str, default='vis', help='name of video')
    args = parser.parse_args()
    return args

# BGR color for gt and pred
color_map = {0: (255, 255, 0), 1: (0, 255, 255)}


def main():
    args = parse_args()
    # load predicted results
    res = json.load(open(args.res, 'r'))
    # load dataset information
    info_path = \
        args.root_path + '/nuscenes_infos_%s.pkl' % args.version
    dataset = pickle.load(open(info_path, 'rb'))
    # prepare save path and medium
    vis_dir = args.save_path
    # if not os.path.exists(vis_dir):
    #     os.makedirs(vis_dir)
    print('saving visualized result to %s' % vis_dir)
    scale_factor = args.scale_factor
    # fixed, no manual setting
    canva_size = 900 * 2 / int(scale_factor)
    show_range = args.show_range
    if args.format == 'video':
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        vout_cam = cv2.VideoWriter(
            os.path.join(vis_dir, '%s.mp4' % args.video_prefix), fourcc,
            args.fps, (int((1600 * 3 + 900 * 2) / scale_factor),
                       int(900 / scale_factor * 2))) # + canva_size for bev

    draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
    draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                                   (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                                   (2, 6), (3, 7)]
    views = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    print('start visualizing results')
    for cnt, data_list in enumerate(
            dataset['data_list'][:min(args.vis_frames, len(dataset['data_list']))]):
        if cnt % 10 == 0:
            print('%d/%d' % (cnt, min(args.vis_frames, len(dataset['data_list']))))
        # collect instances
        pred_res = res['results'][data_list['token']]
        # re-implement for more comprehensive code
        # filtering bbox with vis_thred is added
        pred_boxes = []
        scores = []
        for pred_one_res in pred_res:
            # acquire score at first
            score = pred_one_res['detection_score']
            # analyze geometries
            box_x, box_y, box_z = pred_one_res['translation']
            box_l, box_w, box_h = pred_one_res['size']
            box_theta = Quaternion(pred_one_res['rotation']).yaw_pitch_roll[0]
            # construct list regarding detection score
            if score >= args.vis_thred:
                scores.append(score)
                # matching modified bbox format
                pred_boxes.append([
                    box_x, box_y, box_z - .5 * box_h,
                    box_l, box_w, box_h,
                    -box_theta + np.pi / 2
                ])
        
        if len(pred_boxes) == 0:
            corners_lidar = np.zeros((0, 3), dtype=np.float32)
        else:
            pred_boxes = np.array(pred_boxes, dtype=np.float32)
            boxes = LB(pred_boxes, origin=(0.5, 0.5, 0.0))
            corners_global = boxes.corners.numpy().reshape(-1, 3)
            corners_global = np.concatenate(
                [corners_global,
                 np.ones([corners_global.shape[0], 1])],
                axis=1)
            l2g = get_lidar2global(data_list)
            corners_lidar = corners_global @ np.linalg.inv(l2g).T
            corners_lidar = corners_lidar[:, :3]
        pred_flag = np.ones((corners_lidar.shape[0] // 8, ), dtype=np.bool)
        if args.draw_gt:
            # 从新格式中提取 GT Boxes
            gt_boxes = np.array([inst['bbox_3d'] for inst in data_list['instances']])

            # 调整角度：加上 pi/2
            gt_boxes[:, -1] = gt_boxes[:, -1] + np.pi / 2

            # 交换宽度和长度
            width = gt_boxes[:, 4].copy()
            gt_boxes[:, 4] = gt_boxes[:, 3]
            gt_boxes[:, 3] = width

            # 将 GT boxes 转换为 `LB()` 适配格式
            corners_lidar_gt = LB(
                gt_boxes,  # 这里原来是 `data_list['gt_boxes']`，改成 `gt_boxes`
                origin=(0.5, 0.5, 0.5)
            ).corners.numpy().reshape(-1, 3)

            # 组合 GT 和 Pred 的 corner 点
            corners_lidar = np.concatenate([corners_lidar, corners_lidar_gt], axis=0)

            # 生成 GT 和 Pred 的标记
            gt_flag = np.ones((corners_lidar_gt.shape[0] // 8), dtype=np.bool_)
            pred_flag = np.concatenate([pred_flag, np.logical_not(gt_flag)], axis=0)

            # 生成 GT 分数（全部设为 0）
            scores = scores + [0 for _ in range(gt_boxes.shape[0])]

        # 计算排序索引
        scores = np.array(scores, dtype=np.float32)
        sort_ids = np.argsort(scores)

        # if args.draw_gt:
        #     gt_boxes = [inst['bbox_3d'] for inst in data_list['instances']]

        #     gt_boxes[:, -1] = gt_boxes[:, -1] + np.pi / 2
        #     width = gt_boxes[:, 4].copy()
        #     gt_boxes[:, 4] = gt_boxes[:, 3]
        #     gt_boxes[:, 3] = width
        #     corners_lidar_gt = \
        #         LB(data_list['gt_boxes'],
        #            origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
        #     corners_lidar = np.concatenate([corners_lidar, corners_lidar_gt],
        #                                    axis=0)
        #     gt_flag = np.ones((corners_lidar_gt.shape[0] // 8), dtype=np.bool)
        #     pred_flag = np.concatenate(
        #         [pred_flag, np.logical_not(gt_flag)], axis=0)
        #     scores = scores + [0 for _ in range(data_list['gt_boxes'].shape[0])]
        # scores = np.array(scores, dtype=np.float32)
        # sort_ids = np.argsort(scores)

        # image view
        imgs = []
        for view in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
            img_path = data_list['images'][view]['img_path']
            #print(f"Original img_path for {view}: {img_path}")

            # ✅ 如果 img_path 不是绝对路径，加上数据集根目录
            if not os.path.isabs(img_path):
                dataset_root = "/home/user/xfx_map_align/mmdetection3d-main/data/nuscenes"
                
                # ✅ 这里拼接 'sweeps/CAM_FRONT/' 或者正确的路径
                img_path = os.path.join(dataset_root, "samples", view, img_path)

            #print(f"Full image path for {view}: {img_path}")

            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")
            # draw instances
            corners_img, valid = lidar2img(corners_lidar, data_list['images'][view])
            valid = np.logical_and(
                valid,
                check_point_in_img(corners_img, img.shape[0], img.shape[1])
            )
            valid = valid.reshape(-1, 8)
            corners_img = corners_img.reshape(-1, 8, 2).astype(np.int_)

            for aid in range(valid.shape[0]):
                for index in draw_boxes_indexes_img_view:
                    if valid[aid, index[0]] and valid[aid, index[1]]:
                        cv2.line(
                            img,
                            tuple(corners_img[aid, index[0]]),
                            tuple(corners_img[aid, index[1]]),
                            color=color_map[int(pred_flag[aid])],
                            thickness=scale_factor
                        )
            imgs.append(img)

        '''bird-eye-view visualize code'''
        # bird-eye-view
        canvas = np.ones((int(canva_size), int(canva_size), 3),
                          dtype=np.uint8) * 255 # was zeros, use white background
        # # draw lidar points infos['lidar_points'][view]['lidar_path']
        # # lidar_points = np.fromfile(data_list['lidar_path'], dtype=np.float32)
        # lidar_path = data_list['lidar_points']['lidar_path']  
        # lidar_points = np.fromfile(lidar_path, dtype=np.float32)  
        # lidar_points = lidar_points.reshape(-1, 5)[:, :3]
        # lidar_points[:, 1] = -lidar_points[:, 1]
        # lidar_points[:, :2] = \
        #     (lidar_points[:, :2] + show_range) / show_range / 2.0 * canva_size
        # ✅ 读取 LiDAR 点云数据

        # imgs = []
        # for view in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
        #     img_path = data_list['images'][view]['img_path']

        # draw lidar points
# 这里是在 bird-eye-view 相关的代码段里
        lidar_path = data_list['lidar_points']['lidar_path']
        #print(f"[Debug] Original lidar_path: {lidar_path}")

        # 如果不是绝对路径，自动加上 root_path/samples/LIDAR_TOP
        if not os.path.isabs(lidar_path):
            lidar_path = os.path.join(args.root_path, "samples/LIDAR_TOP", lidar_path)

        if not os.path.exists(lidar_path):
            raise FileNotFoundError(f"Cannot find LiDAR file: {lidar_path}")

        #print(f"[Debug] Full lidar_path: {lidar_path}")

        # 正常读取文件
        lidar_points = np.fromfile(lidar_path, dtype=np.float32)
        lidar_points = lidar_points.reshape(-1, 5)[:, :3]
        lidar_points[:, 1] = -lidar_points[:, 1]
        lidar_points[:, :2] = \
            (lidar_points[:, :2] + show_range) / show_range / 2.0 * canva_size
        for p in lidar_points:
            if check_point_in_img(
                    p.reshape(1, 3), canvas.shape[1], canvas.shape[0])[0]:
                # color = depth2color(p[2])
                color = tuple([96, 96, 96])
                cv2.circle(
                    canvas, (int(p[0]), int(p[1])),
                    radius=0,
                    color=color,
                    thickness=1)

        # draw instances
        corners_lidar = corners_lidar.reshape(-1, 8, 3)
        corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
        bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
        bottom_corners_bev = \
            (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
        bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
        center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
        head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
        canter_canvas = \
            (center_bev + show_range) / show_range / 2.0 * canva_size
        center_canvas = canter_canvas.astype(np.int32)
        head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size
        head_canvas = head_canvas.astype(np.int32)

        for rid in sort_ids:
            # threshold already applied. not updated, lazy.
            score = scores[rid]
            if score < args.vis_thred and pred_flag[rid]:
                continue
            score = min(score * 2.0, 1.0) if pred_flag[rid] else 1.0
            color = color_map[int(pred_flag[rid])]
            for index in draw_boxes_indexes_bev:
                cv2.line(
                    canvas,
                    bottom_corners_bev[rid, index[0]],
                    bottom_corners_bev[rid, index[1]],
                    [color[0] * score, color[1] * score, color[2] * score],
                    thickness=1)
            cv2.line(
                canvas,
                center_canvas[rid],
                head_canvas[rid],
                [color[0] * score, color[1] * score, color[2] * score],
                1,
                lineType=8)

        # fuse image-view and bev
        # dislike old arrangement style, codes not kept
        img = np.zeros((900 * 2, 1600 * 3 + 900 * 2, 3), # square bev, fixed size
                       dtype=np.uint8)
        # front cams assigned to top left
        img[:900, :1600 * 3, :] = np.concatenate(imgs[:3], axis=1)
        img_back = np.concatenate(
            [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]],
            axis=1)
        # back cams assigned to bottom left
        img[900:, :1600 * 3, :] = img_back
        img = cv2.resize(img, (int((1600 * 3 + 900 * 2) / scale_factor),
                               int(900 * 2 / scale_factor)))
        # bev assigned to right
        img[:, int(1600 * 3 / scale_factor):, :] = canvas
        if args.format == 'image':
            cv2.imwrite(os.path.join(vis_dir, '%s.jpg' % data_list['token']), img)
        elif args.format == 'video':
            vout_cam.write(img)
    if args.format == 'video':
        vout_cam.release()


if __name__ == '__main__':
    main()


#CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_test.sh ./projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ./checkpoint/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth 2 

#CUDA_VISIBLE_DEVICES=0,1 python tools/analysis_tools/vis.py /home/tianziyue/xiefuxin/mmdetection3d/outputs/results_nusc.json --vis-frames 6019 --draw-gt --version val --save_path ./outputs/plots/ --video-prefix best-map-val-full