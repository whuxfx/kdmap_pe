import torch

# 1️⃣ 加载原始 checkpoint
checkpoint_path = "/home/user/xfx_map_align/lisa-hdmap/checkpoint/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# 2️⃣ 遍历 state_dict 中的所有参数
for key in checkpoint["state_dict"]:
    tensor = checkpoint["state_dict"][key]

    # 只处理 5D 的 Conv3D 权重，跳过 2D Conv 和其他层
    if tensor.dim() == 5:
        print(f"Fixing shape for: {key} | Old Shape: {tensor.shape}")
        
        # 交换维度: [C_out, 3, 3, 3, C_in] -> [3, 3, 3, C_in, C_out]
        tensor = tensor.permute(1, 2, 3, 4, 0)

        # 更新 checkpoint
        checkpoint["state_dict"][key] = tensor
        print(f"New Shape: {tensor.shape}")

# 3️⃣ 保存修正后的 checkpoint
fixed_checkpoint_path = "checkpoint/bevfusionL_fixed.pth"
torch.save(checkpoint, fixed_checkpoint_path)
print(f"✅ 转换完成，新的权重文件保存在 {fixed_checkpoint_path}")
