scp -P 2222 yuexiang@119.96.24.230:/data03/local-path-provisioner/pvc-8c4bbe21-3ee3-4b9d-827f-5810a828198a_ns-2_local-pvc-test/salience_detr.py /home/user/xfx_map_align/lisa-hdmap/projects/


数据集转换：
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes


训练

bash tools/dist_train.sh projects/BEVFusion/configs/map_fusion.py 1 --cfg-options load_from=./checkpoint/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth model.img_backbone.init_cfg.checkpoint=./checkpoint/swin_tiny_patch4_window7_224.pth


bash tools/dist_test.sh ./work_dirs/map_fusion/map_fusion.py ./work_dirs/map_fusion/epoch_2.pth 2

推断


bash tools/dist_test.sh ./work_dirs/map_fusion/map_fusion.py ./work_dirs/map_fusion/epoch_2.pth 2

demo

python projects/BEVFusion/demo/multi_modality_demo.py demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin demo/data/nuscenes/ demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ./checkpoint/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth --cam-type all --score-thr 0.2 --show

python tools/test.py projects/BEVFusion/configs/map_fusion.py ./work_dirs/map_fusion/epoch_2.pth --show --show-dir ./result/map --task 'multi-modality_det'

python tools/analysis_tools/vis.py /home/user/xfx_map_align/mmdetection3d-main/outputs/results_nusc.json --vis-frames 600 --draw-gt --version val --save_path ./work_dirs/test --video-prefix best-map-val-full

./tools/dist_test.sh configs/fusion_all.py /home/user/xfx_map_align/MENet-main/work_dirs/menet/latest.pth 1 --format-only --eval-options jsonfile_prefix=/home/user/xfx_map_align/MENet-main/work_dirs/test


python tools/misc/visualize_results.py projects/BEVFusion/configs/map_fusion.py --result ./outputs/results_nusc.pkl --show-dir /home/user/xfx_map_align/mmdetection3d-main/outputs


python tools/test.py ${CONFIG_FILE} ${CKPT_PATH} --show --show-dir ${SHOW_DIR}


CUDA_VISIBLE_DEVICES=1 bash tools/dist_test.sh ./projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py ./checkpoint/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth 2 


tar cf - * | ssh xiefuxin@202.114.114.179 "cd /home/xiefuxin/HDmapfusion/ && tar xf -"

