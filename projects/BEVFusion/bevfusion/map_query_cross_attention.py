

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

class PositionEncodingLearned(nn.Module):
    def __init__(self, input_channel, num_pos_feats=128):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS


@MODELS.register_module()
class MapQueryCrossAttention(nn.Module):
    def __init__(self, embed_dim, map_dim=None, pos_encoder=None, num_heads=8):
        super().__init__()
        self.map_dim = map_dim or embed_dim
        self.pos_encoder = pos_encoder  # ✅ 共享位置编码器

        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, self.map_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.map_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.map_dim, self.map_dim, kernel_size=3, padding=1)
        )

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query_feat, map_mask, query_pos=None, map_pos=None):
        """
        Args:
            query_feat: [B, C, N]
            map_mask: [B, 1, H, W]
            query_pos: [B, N, 2] or None
            map_pos: [B, H*W, 2] or [B, HW, 2] or None
        Returns:
            [B, C, N] after attention
        """
        B, C, N = query_feat.shape
        query_feat = query_feat.permute(0, 2, 1)  # [B, N, C]

        # Encode map mask
        map_tokens = self.map_encoder(map_mask)  # [B, C, H, W]
        map_tokens = map_tokens.flatten(2).permute(0, 2, 1)  # [B, HW, C]

        # === 位置编码 ===
        if self.pos_encoder is not None:
            if query_pos is not None:
                query_pos_embed = self.pos_encoder(query_pos)  # [B, C, N]
                query_pos_embed = query_pos_embed.permute(0, 2, 1)  # [B, N, C]
                query_feat = query_feat + query_pos_embed
            if map_pos is not None:
                map_pos_embed = self.pos_encoder(map_pos)  # [B, C, HW]
                map_pos_embed = map_pos_embed.permute(0, 2, 1)  # [B, HW, C]
                map_tokens = map_tokens + map_pos_embed

        # === Cross-Attention ===
        attn_output, _ = self.cross_attn(
            query=query_feat,  # [B, N, C]
            key=map_tokens,    # [B, HW, C]
            value=map_tokens   # [B, HW, C]
        )
        out = self.norm(query_feat + self.out_proj(attn_output))  # [B, N, C]
        return out.permute(0, 2, 1)  # [B, C, N]
        # return out.permute(0, 2, 1), query_pos_embed  # [B, C, N], [B, N, C]
    

    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmdet3d.registry import MODELS

# class PositionEncodingLearned(nn.Module):
#     def __init__(self, input_channel, num_pos_feats=128):
#         super().__init__()
#         self.position_embedding_head = nn.Sequential(
#             nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
#             nn.BatchNorm1d(num_pos_feats),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

#     def forward(self, xyz):
#         xyz = xyz.transpose(1, 2).contiguous()
#         position_embedding = self.position_embedding_head(xyz)
#         return position_embedding


# @MODELS.register_module()
# class MapQueryCrossAttention(nn.Module):
#     def __init__(self, embed_dim, map_dim=None, query_pos_encoder=None, map_pos_encoder=None, num_heads=8):
#         super().__init__()
#         self.map_dim = map_dim or embed_dim
#         self.query_pos_encoder = query_pos_encoder
#         self.map_pos_encoder = map_pos_encoder

#         self.map_encoder = nn.Sequential(
#             nn.Conv2d(1, self.map_dim, kernel_size=3, padding=1),
#             nn.BatchNorm2d(self.map_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.map_dim, self.map_dim, kernel_size=3, padding=1)
#         )

#         self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
#         self.norm = nn.LayerNorm(embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)

#     def forward(self, query_feat, map_mask, query_pos=None, map_pos=None):
#         """
#         Args:
#             query_feat: [B, C, N]
#             map_mask: [B, 1, H, W]
#             query_pos: [B, N, 2] or None
#             map_pos: [B, HW, 2] or None
#         Returns:
#             query_feat: [B, C, N] after attention
#             query_pos_embed: [B, N, C] position embedding for later use
#         """
#         B, C, N = query_feat.shape
#         query_feat = query_feat.permute(0, 2, 1)  # [B, N, C]

#         map_tokens = self.map_encoder(map_mask)  # [B, C, H, W]
#         map_tokens = map_tokens.flatten(2).permute(0, 2, 1)  # [B, HW, C]

#         query_pos_embed = None
#         if self.query_pos_encoder is not None and query_pos is not None:
#             query_pos_embed = self.query_pos_encoder(query_pos)  # [B, C, N]
#             query_pos_embed = query_pos_embed.permute(0, 2, 1)  # [B, N, C]
#             query_feat = query_feat + query_pos_embed

#         if self.map_pos_encoder is not None and map_pos is not None:
#             map_pos_embed = self.map_pos_encoder(map_pos)  # [B, C, HW]
#             map_pos_embed = map_pos_embed.permute(0, 2, 1)  # [B, HW, C]
#             map_tokens = map_tokens + map_pos_embed

#         attn_output, _ = self.cross_attn(query=query_feat, key=map_tokens, value=map_tokens)
#         out = self.norm(query_feat + self.out_proj(attn_output))
#         return out.permute(0, 2, 1), query_pos_embed  # [B, C, N], [B, N, C]
