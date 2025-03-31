# import torch
# import numpy as np

# from mmdet3d.visualization import Det3DLocalVisualizer
# from mmdet3d.structures import LiDARInstance3DBoxes

# points = np.fromfile('demo/data/kitti/000008.bin', dtype=np.float32)
# points = points.reshape(-1, 4)
# visualizer = Det3DLocalVisualizer()
# # set point cloud in visualizer
# visualizer.set_points(points)
# bboxes_3d = LiDARInstance3DBoxes(
#     torch.tensor([[8.7314, -1.8559, -1.5997, 4.2000, 3.4800, 1.8900,
#                    -1.5808]]))
# # Draw 3D bboxes
# visualizer.draw_bboxes_3d(bboxes_3d)
# visualizer.show()

import torch
import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes

import json

kitti_file = 'demo/data/kitti/000008.bin'
json_file = '/home/user/xfx_map_align/mmdetection3d-main/outputs/results_nusc.json'
points = np.fromfile(file=kitti_file, dtype=np.float32, count=-1).reshape([-1, 4])
# points = np.fromfile('demo/data/kitti/000008.bin', dtype=np.float32)
with open(json_file, 'r', encoding='utf-8') as file:
    # 使用json.load()方法解析JSON数据
    data = json.load(file)

bboxes = data['bboxes_3d']
points = points.reshape(-1, 4)
visualizer = Det3DLocalVisualizer()

# set point cloud in visualizer
visualizer.set_points(points)
bboxes_3d = LiDARInstance3DBoxes(torch.tensor(bboxes))
# Draw 3D bboxes
visualizer.draw_bboxes_3d(bboxes_3d)
visualizer.show()

