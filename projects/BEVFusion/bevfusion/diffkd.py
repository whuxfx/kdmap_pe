import torch
from torch import nn
import torch.nn.functional as F
from .diffkd_modules import DiffusionModel, NoiseAdapter, AutoEncoder, DDIMPipeline
from .scheduling_ddim import DDIMScheduler


class DiffKD(nn.Module):
    def __init__(
            self,
            student_channels,
            teacher_channels,
            kernel_size=3,
            inference_steps=5,
            num_train_timesteps=1000,
            use_ae=False,
            ae_channels=None,
            distill_cfg=None  # ✅ 加这句
    ):
        super().__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps
        self.distill_cfg = distill_cfg or {}  # ✅ 安全地保存配置

        # AutoEncoder for compressing teacher feature
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels

        # transform student feature to match teacher dimension
        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)

        # Diffusion model to predict noise
        self.model = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            clip_sample=False,
            beta_schedule="linear"
        )
        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size)

        # Pipeline for denoising
        self.pipeline = DDIMPipeline(self.model, self.scheduler, self.noise_adapter)

        # Projection layer
        self.proj = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, 1),
            nn.BatchNorm2d(teacher_channels)
        )

    def forward(self, student_feat, teacher_feat):
        # Project student feature
        student_feat = self.trans(student_feat)

        # Autoencode teacher feature
        if self.use_ae:
            hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
            rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
            teacher_feat = hidden_t_feat.detach()
        else:
            rec_loss = None

        # Denoise student feature using diffusion process
        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=self.proj
        )
        refined_feat = self.proj(refined_feat)

        # Compute diffusion loss
        ddim_loss = self.ddim_loss(teacher_feat)

        return refined_feat, ddim_loss, teacher_feat, rec_loss

    def ddim_loss(self, gt_feat):
        noise = torch.randn_like(gt_feat)
        bs = gt_feat.shape[0]
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps)
        return F.mse_loss(noise_pred, noise)
