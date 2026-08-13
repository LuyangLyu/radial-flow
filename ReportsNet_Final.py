import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import math
import numpy as np

# ==========================================
# 辅助基础模块 (Utility Modules)
# ==========================================

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class AdaptiveSpectralEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_c, self.out_c = in_channels, out_channels
        if in_channels < out_channels:
            self.mode = 'padding'
            self.learnable_tokens = nn.Parameter(torch.randn(1, out_channels - in_channels, 1, 1) * 0.02)
        elif in_channels > out_channels:
            self.mode = 'resample'
        else:
            self.mode = 'identity'

        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.mode == 'padding':
            B, _, H, W = x.shape
            tokens = self.learnable_tokens.expand(B, -1, H, W)
            out = torch.cat([x, tokens], dim=1)
        elif self.mode == 'resample':
            B, C, H, W = x.shape
            x_seq = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)
            x_res = F.interpolate(x_seq, size=self.out_c, mode='linear', align_corners=True)
            out = x_res.reshape(B, H, W, self.out_c).permute(0, 3, 1, 2).contiguous()
        else:
            out = x
        return self.act(out)

class CircularPolarConv2d(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=0)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = F.pad(x, (0, 0, self.pad, self.pad), mode='circular')
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode='constant', value=0)
        x = self.conv(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(x)

# ==========================================
# Model 10.1: Hierarchical Frequency RoPE
# ==========================================

class HierarchicalPolarRoPE(nn.Module):
    """
    层次化极坐标旋转位置编码 (Hierarchical Frequency RoPE)

    改进点（相比Checkerboard RRPE）：
    1. L维（径向）：多频率RoPE，不同通道对使用不同频率，捕获多尺度径向结构
    2. K维（角度）：单频率真2D旋转矩阵，非标量近似，实现真正的旋转等变性

    数学公式：
    - Radial (L): x'_c = x_c * cos(l * ω_c) + x_{c+1} * sin(l * ω_c)
    - Angular (K): 对通道对 (c, c+1) 应用 2D 旋转矩阵
    """
    def __init__(self, C, L, max_relative_angle, base=10000):
        super().__init__()
        self.C = C
        self.L = L
        self.max_relative_angle = max_relative_angle

        # 1. L维：多频率RoPE (base=10000)
        # 频率: ω_c = base^(-2c/d), d = C // 2
        d = C // 2
        freqs = 1.0 / (base ** (torch.arange(0, d, 1).float() / d))  # (d,)
        self.register_buffer('freqs', freqs)  # (C//2,)

        # 2. K维：预计算所有相对角度的 sin/cos (单频率)
        angles_idx = torch.arange(max_relative_angle).float()
        rotation_angles = angles_idx * (2 * math.pi / max_relative_angle)
        self.register_buffer('sin_vals', torch.sin(rotation_angles))
        self.register_buffer('cos_vals', torch.cos(rotation_angles))

    def forward(self, x, rel_angle_idx):
        """
        对 x 应用层次化极坐标旋转位置编码

        Args:
            x: 输入特征，形状 (B, C, K, L)
            rel_angle_idx: 相对角度索引矩阵，形状 (K, K)

        Returns:
            旋转后的特征，形状 (B, C, K, K, L)
        """
        B, C, K_num, L = x.shape
        K = K_num

        # ========== Step 1: L维 - 多频率RoPE (Radial) ==========
        # 将C分为两组通道对，应用不同频率
        # x: (B, C, K, L) -> 重塑为 (B, 2, C//2, K, L)
        x_reshaped = x.view(B, 2, C // 2, K, L)  # (B, 2, C//2, K, L)

        # 生成L维的位置编码: (C//2, L)
        l_positions = torch.arange(L, device=x.device).float()  # (L,)
        # freqs: (C//2,) -> 扩展到 (C//2, L)
        freqs_l = self.freqs.view(-1, 1) * l_positions.view(1, L)  # (C//2, L)

        # 计算 sin/cos: (C//2, L)
        sin_l = torch.sin(freqs_l).unsqueeze(0).unsqueeze(2)  # (1, C//2, 1, L)
        cos_l = torch.cos(freqs_l).unsqueeze(0).unsqueeze(2)  # (1, C//2, 1, L)

        # 应用L维RoPE: x_0 * cos + x_1 * sin
        x_0 = x_reshaped[:, 0]  # (B, C//2, K, L)
        x_1 = x_reshaped[:, 1]  # (B, C//2, K, L)
        x_l_rope = torch.cat([
            x_0 * cos_l - x_1 * sin_l,
            x_0 * sin_l + x_1 * cos_l,
        ], dim=1)  # (B, C, K, L)

        # ========== Step 2: K维 - 真2D旋转 (Angular) ==========
        # 扩展 x 以生成所有相对角度的版本
        # x_l_rope: (B, C, K, L) -> (B, C, K, K, L)
        x_expanded = x_l_rope.unsqueeze(2).expand(-1, -1, K, -1, -1)  # (B, C, K, K, L)

        # 获取对应相对角度的 sin/cos 值
        sin_theta = self.sin_vals[rel_angle_idx]  # (K, K)
        cos_theta = self.cos_vals[rel_angle_idx]  # (K, K)

        # 扩展到 (1, 1, K, K, 1)
        sin_theta = sin_theta.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        cos_theta = cos_theta.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # 对通道对 (c, c+1) 应用2D旋转矩阵
        # 重塑为 (B, 2, C//2, K, K, L)
        x_pair = x_expanded.view(B, 2, C // 2, K, K, L)

        x_c = x_pair[:, 0]  # (B, C//2, K, K, L)
        x_c1 = x_pair[:, 1]  # (B, C//2, K, K, L)

        # 旋转公式: x' = x*cos - y*sin, y' = x*sin + y*cos
        x_rot = x_c * cos_theta - x_c1 * sin_theta
        x_rot_1 = x_c * sin_theta + x_c1 * cos_theta

        # 合并回去: (B, C, K, K, L)
        x_final = torch.cat([x_rot, x_rot_1], dim=1)

        return x_final


class CircularSelfAttentionWithRRPE(nn.Module):
    """
    Model 10.1: 基线-转线分离设计的注意力机制（Vision1DStyle 方式）
    使用 HierarchicalPolarRoPE 层次化频率编码

    每条线都作为基线，与其他所有转线交互
    Q 从线 i 得到，K、V 从线 j 得到（每条线独立计算）
    对 K 应用 HierarchicalPolarRoPE（多频率径向 + 单频率角度）
    """
    def __init__(self, channels, L=11, K=24, num_heads=1):
        super().__init__()
        self.K, self.L, self.C = K, L, channels
        self.num_heads = num_heads

        # Q、K、V 投影（Conv1d，Vision1DStyle 方式）
        self.q_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv1d(channels, channels, kernel_size=1)

        # HierarchicalPolarRoPE（层次化频率编码）
        self.rrpe = HierarchicalPolarRoPE(C=channels, L=L, max_relative_angle=K)

        # 输出投影
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)

        # Norm（在输入 C 维度）
        self.norm = nn.LayerNorm(channels)

        # Scale
        self.scale = (L // num_heads) ** -0.5

    def forward(self, x):
        """
        x: (B, K, L, C) - K 条线

        Returns:
            out: (B, K, L, C) - 带有 HierarchicalPolarRoPE 的注意力输出
        """
        B, K, L, C = x.shape

        # Preserve the semantic layout explicitly:
        # (B, K, L, C) -> (B, K, C, L) -> (B*K, C, L).
        # Direct reshape would scramble the radial and channel dimensions.
        x_reshaped = x.permute(0, 1, 3, 2).contiguous().reshape(B * K, C, L)

        # 2. 归一化（沿 C 维度）
        x_norm = x_reshaped.permute(0, 2, 1)  # (B*K, L, C)
        x_norm = self.norm(x_norm)  # (B*K, L, C)
        x_norm = x_norm.permute(0, 2, 1)  # (B*K, C, L)

        # 3. 一次性计算所有 Q、K、V（Vision1DStyle 方式，使用 Conv1d）
        q_all = self.q_proj(x_norm)  # (B*K, C, L)
        k_all = self.k_proj(x_norm)  # (B*K, C, L)
        v_all = self.v_proj(x_norm)  # (B*K, C, L)

        # 重塑为 (B, K, C, L)
        q_all = q_all.reshape(B, K, C, L)  # (B, K, C, L)
        k_all = k_all.reshape(B, K, C, L)  # (B, K, C, L)
        v_all = v_all.reshape(B, K, C, L)  # (B, K, C, L)

        # 4. 计算相对角度矩阵（转换为弧度）
        idx_i = torch.arange(K, device=x.device).unsqueeze(1)  # (K, 1)
        idx_j = torch.arange(K, device=x.device).unsqueeze(0)  # (1, K)
        rel_angle_idx = (idx_j - idx_i) % K  # (K, K)

        # 5. 对 K 应用 HierarchicalPolarRoPE（层次化频率编码）
        k_rrpe = self.rrpe(k_all.permute(0, 2, 1, 3), rel_angle_idx)  # (B, C, K, K, L)

        # 6. 标准矩阵乘法注意力 (Model 11.1: 标准注意力)
        # 设计: 每条线 i 独立做 L 维度的标准注意力
        # RoPE 编码了 K 维度的相对角度信息到特征中

        # Q, K, V 形状调整: (B, K, C, L) -> (B*K, L, C)
        q = q_all.permute(0, 1, 3, 2).reshape(B * K, L, C)  # (BK, L, C)
        v = v_all.permute(0, 1, 3, 2).reshape(B * K, L, C)  # (BK, L, C)

        # 对于 K: 从 k_rrpe (B, C, K, K, L) 提取每条线的编码后特征
        # k_rrpe[:, :, i, j, :] 表示线j相对于线i的编码特征
        # 简化: 对 K 维度求和/平均，将角度信息聚合到特征中
        k_rope = k_rrpe.mean(dim=3)  # (B, C, K, L) - 对 K 维度平均
        k = k_rope.permute(0, 2, 3, 1).reshape(B * K, L, C)  # (BK, L, C)

        # 标准矩阵乘法注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (BK, L, L)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (BK, L, C)

        # Output projection with the same explicit inverse layout.
        out = out.reshape(B, K, L, C)
        out = out.permute(0, 1, 3, 2).contiguous().reshape(B * K, C, L)
        out = self.out_proj(out)

        # Residual connection and exact inverse transform to (B, K, L, C).
        out = out + x_reshaped
        return out.reshape(B, K, C, L).permute(0, 1, 3, 2).contiguous()


class LGMBlock_model15(nn.Module):
    """
    Model 15 的 LGMBlock：基于 11.1_no_mamba，RLE 中禁用 Mamba 分支，保留 CircularPolarConv2d + HPRPE Attention。
    """
    def __init__(self, channels, K=12, drop_path=0.2):
        super().__init__()
        self.pre_conv = nn.Conv2d(channels, channels, 1)
        self.pre_norm = nn.LayerNorm(channels)
        self.pre_act = nn.SiLU()
        self.conv_branch = nn.Sequential(CircularPolarConv2d(channels), CircularPolarConv2d(channels))
        # Model 10: 使用带有 RRPE 的 CircularSelfAttention
        self.ang_interact = CircularSelfAttentionWithRRPE(channels, K=K*2)
        self.fusion = nn.Sequential(nn.Linear(channels * 2, channels), nn.LayerNorm(channels), nn.SiLU())
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, H, W, C = x.shape
        x_p = self.pre_conv(x.permute(0, 3, 1, 2))                           # (B, C, H, W)
        x_p = self.pre_norm(x_p.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)    # LN over C
        x_p = self.pre_act(x_p).permute(0, 2, 3, 1)                          # (B, H, W, C)
        x_norm = self.norm(x_p)
        x_conv = self.conv_branch(x_norm.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out_rad = torch.zeros_like(x_norm)
        out_ang = self.ang_interact(x_norm)
        x_inter = self.fusion(torch.cat([out_rad, out_ang], dim=-1))
        return x + self.drop_path(x_conv + x_inter)

# ==========================================
# 核心计算组件 (Model 5.52: 跨波段并行卷积)
# ==========================================

class SpeMamba(nn.Module):
    """Model 5.52: 修复了旋转敏感的 1D 卷积，改为旋转不变的 1x1 卷积，并增加了外层残差"""
    def __init__(self, channels, bidirectional=True):
        super().__init__()
        # 🔴 关键修复：采用 1x1 卷积保证旋转不变性
        self.conv_point = nn.Conv2d(channels, channels, kernel_size=1)
        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        self.bidirectional = bidirectional
        if bidirectional:
            self.mamba_rev = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape
        # 1x1 卷积支路
        x_conv = self.conv_point(x)

        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)
        out = self.mamba(x_flat)
        if self.bidirectional:
            out_rev = self.mamba_rev(x_flat.flip(dims=[1])).flip(dims=[1])
            out = out + out_rev
        x_mamba = out.reshape(B, H, W, C).permute(0, 3, 1, 2)

        fused = (x_conv + x_mamba).permute(0, 2, 3, 1)  # (B, H, W, C)
        fused = self.norm(fused).permute(0, 3, 1, 2)     # back to (B, C, H, W)
        return x + self.act(fused)

class RadialLineSampler(nn.Module):
    def __init__(self, K=12, L=11, patch_size=11):
        super().__init__()
        self.K, self.L = K, L
        angles = torch.linspace(0, math.pi, K + 1)[:-1]
        sector_width = math.pi / K
        angle_offsets = torch.tensor([-sector_width/4, 0, sector_width/4])
        multi_angles = angles.unsqueeze(1) + angle_offsets.unsqueeze(0)
        distances = torch.linspace(-1, 1, L)
        grid = torch.zeros(K, 3, L, 2)
        for k in range(K):
            for sub_i in range(3):
                theta = multi_angles[k, sub_i]
                for p_idx in range(L):
                    d = distances[p_idx]
                    grid[k, sub_i, p_idx, 0] = d * torch.cos(theta)
                    grid[k, sub_i, p_idx, 1] = d * torch.sin(theta)
        self.register_buffer('sampling_grid', grid.view(1, K * 3, L, 2))

    def forward(self, x):
        B, C, H, W = x.shape
        grid = self.sampling_grid.expand(B, -1, -1, -1).to(x.device).to(x.dtype)
        sampled = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled = sampled.view(B, C, self.K, 3, self.L).mean(dim=3)
        return sampled.permute(0, 2, 3, 1).contiguous()

class LiDARGradientModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('gx', torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3))
        self.register_buffer('gy', torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3))

    def forward(self, x):
        if x.shape[1] > 1: return x
        grad_x = F.conv2d(x, self.gx, padding=1)
        grad_y = F.conv2d(x, self.gy, padding=1)
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return torch.cat([x, magnitude], dim=1)

class RadialSymmetricCrossAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.h_to_l = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.l_to_h = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm_h = nn.LayerNorm(channels)
        self.norm_l = nn.LayerNorm(channels)
    def forward(self, h, l):
        B, H, W, C = h.shape
        h_seq, l_seq = h.reshape(B * H, W, C), l.reshape(B * H, W, C)
        a_h, _ = self.h_to_l(h_seq, l_seq, l_seq)
        a_l, _ = self.l_to_h(l_seq, h_seq, h_seq)
        return self.norm_h(h + a_h.reshape(B, H, W, C)), self.norm_l(l + a_l.reshape(B, H, W, C))

class RadialPatchPooling(nn.Module):
    def __init__(self, patch_size=11):
        super().__init__()
        center = patch_size // 2
        y, x = torch.meshgrid(torch.arange(patch_size), torch.arange(patch_size), indexing='ij')
        dist_sq = (y - center)**2 + (x - center)**2
        unique_dists = torch.unique(dist_sq).sort()[0]
        self.num_rings = len(unique_dists)
        self.ring_indices = [torch.nonzero((dist_sq == d).reshape(-1)).flatten() for d in unique_dists]
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, -1)
        res = []
        for idx in self.ring_indices:
            pixels = x_flat[:, :, idx.to(x.device)]
            res.append(torch.stack([pixels.mean(dim=-1), pixels.max(dim=-1)[0]], dim=1))
        return torch.stack(res, dim=1)

class LiDARRadialProcessor(nn.Module):
    def __init__(self, target_channels, patch_size=11, lidar_channels=1):
        super().__init__()
        self.grad_extract = LiDARGradientModule()
        self.pool = RadialPatchPooling(patch_size)
        self.in_dim = 2 if lidar_channels == 1 else lidar_channels
        self.proj = nn.Linear(self.in_dim, target_channels)
    def forward(self, x):
        x = self.grad_extract(x)
        return self.proj(self.pool(x))

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.dim = 2 * d_model
        self.h_to_l = nn.MultiheadAttention(self.dim, num_heads, batch_first=True)
        self.l_to_h = nn.MultiheadAttention(self.dim, num_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(self.dim * 2, self.dim), nn.Sigmoid())
        self.norm1, self.norm2 = nn.LayerNorm(self.dim), nn.LayerNorm(self.dim)
    def forward(self, h, l):
        B, L, S, C = h.shape
        h_seq, l_seq = h.reshape(B, L, S * C), l.reshape(B, L, S * C)
        h_en, _ = self.h_to_l(h_seq, l_seq, l_seq)
        h_en = self.norm1(h_en + h_seq)
        l_en, _ = self.l_to_h(l_seq, h_seq, h_seq)
        l_en = self.norm2(l_en + l_seq)
        g = self.gate(torch.cat([h_en, l_en], dim=-1))
        return (g * h_en + (1 - g) * l_en).reshape(B, L, S, C)

# ==========================================
# 完整网络 (Network Architectures)
# ==========================================

class RadialFlowNet(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, patch_size=11, num_classes=15, lidar_channels=1):
        super().__init__()
        self.pool = RadialPatchPooling(patch_size)
        self.lidar_proc = LiDARRadialProcessor(hidden_dim, patch_size, lidar_channels=lidar_channels)
        self.fusion = CrossModalAttentionFusion(hidden_dim)
        self.classifier = nn.Identity()

    def forward(self, hsi_feat, lidar):
        h_r, l_r = self.pool(hsi_feat), self.lidar_proc(lidar)
        f_fused = self.fusion(h_r, l_r)
        return f_fused.reshape(f_fused.shape[0], -1)

class RadialLineFlowNet(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, patch_size=11, num_classes=15, num_layers=2, lidar_channels=1, K=12):
        super().__init__()
        # Model 10: K is now configurable
        self.sampler = RadialLineSampler(K=K, L=11, patch_size=patch_size)
        self.grad_extract = LiDARGradientModule()
        self.lidar_dim = 2 if lidar_channels == 1 else lidar_channels
        self.lidar_proj = nn.Linear(self.lidar_dim, hidden_dim)
        self.classifier = nn.Identity()
    def forward(self, hsi_feat, lidar): pass


class RadialSynergyNet_model15_3(nn.Module):
    """
    Model 15: 从 11.1_no_mamba 完整复制出的论文主模型基础版。

    结构要点：
    - RLE/LGM 中禁用径向 Mamba 分支
    - 保留 CircularPolarConv2d、HierarchicalPolarRoPE attention、外部 E_rad 位置编码
    - 保留 SSA/Spectral Mamba、aux logits 和 center head
    """
    def __init__(self, hsi_channels, lidar_channels=1, hidden_dim=64, patch_size=11, num_classes=15, num_layers=2, K=12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.K = K

        self.hsi_embed = AdaptiveSpectralEmbedding(hsi_channels, hidden_dim)
        self.spe_mamba = SpeMamba(hidden_dim, bidirectional=True)

        self.ring_expert = RadialFlowNet(hsi_channels, hidden_dim, patch_size, num_classes, lidar_channels)
        self.line_expert = RadialLineFlowNet(hsi_channels, hidden_dim, patch_size, num_classes, num_layers, lidar_channels, K)

        # Model 15: standalone LGM block copied from 11.1_no_mamba.
        self.h_layers = nn.ModuleList([LGMBlock_model15(hidden_dim, K) for _ in range(num_layers)])
        self.l_layers = nn.ModuleList([LGMBlock_model15(hidden_dim, K) for _ in range(num_layers)])
        self.cross_layers = nn.ModuleList([RadialSymmetricCrossAttention(hidden_dim) for _ in range(num_layers)])

        # Synergy Head
        self.total_feat_dim = 7 * hidden_dim
        self.synergy_head = nn.Sequential(
            nn.Linear(self.total_feat_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4 * hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, num_classes)
            ) for _ in range(num_layers)
        ])

        self.center_head = nn.Sequential(
            nn.Linear(18 * hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

        self.line_sampler = self.line_expert.sampler
        self.grad_extract = self.line_expert.grad_extract
        self.lidar_proj = self.line_expert.lidar_proj

        self.pe = nn.Parameter(torch.randn(1, 1, 11, hidden_dim) * 0.02)

    def forward(self, hsi, lidar, return_feat=False):
        B = hsi.shape[0]
        c = self.hidden_dim

        hsi_f = self.spe_mamba(self.hsi_embed(hsi))
        center_f = hsi_f[:, :, 5, 5]

        feat_ring_raw = self.ring_expert(hsi_f, lidar)
        feat_line_raw = self.line_sampler(hsi_f)

        h_lines = self.line_sampler(hsi_f)
        l_lines = self.lidar_proj(self.line_sampler(self.grad_extract(lidar)))

        h_grid = torch.cat([h_lines, torch.flip(h_lines, [2])], 1) + self.pe
        l_grid = torch.cat([l_lines, torch.flip(l_lines, [2])], 1) + self.pe

        aux_logits = []

        for i in range(self.num_layers):
            h_grid = self.h_layers[i](h_grid)
            l_grid = self.l_layers[i](l_grid)
            h_grid, l_grid = self.cross_layers[i](h_grid, l_grid)
            h_stat_i = torch.stack([h_grid.mean(1), h_grid.max(1)[0]], -1).mean(dim=1).reshape(B, -1)
            l_stat_i = torch.stack([l_grid.mean(1), l_grid.max(1)[0]], -1).mean(dim=1).reshape(B, -1)

            aux_logits.append(self.aux_heads[i](torch.cat([h_stat_i, l_stat_i], dim=-1)))

        h_stat = torch.stack([h_grid.mean(1), h_grid.max(1)[0]], -1)
        l_stat = torch.stack([l_grid.mean(1), l_grid.max(1)[0]], -1)

        feat_line_pooled = torch.cat([h_stat.mean(dim=1).reshape(B, -1), l_stat.mean(dim=1).reshape(B, -1)], dim=-1)

        center_ring = feat_ring_raw.reshape(B, -1, 2, c)[:, :3, :, :].reshape(B, -1)
        h_c = h_stat[:, 4:7, :, :].reshape(B, -1)
        l_c = l_stat[:, 4:7, :, :].reshape(B, -1)

        logits_center = self.center_head(torch.cat([center_ring, h_c, l_c], dim=1))

        feat_ring_pooled = feat_ring_raw.reshape(B, -1, 2, c).mean(dim=1).reshape(B, -1)

        feat_syn = torch.cat([feat_line_pooled, feat_ring_pooled, center_f], dim=1)

        logits_syn = self.synergy_head(feat_syn)

        if return_feat: return feat_syn
        if self.training: return logits_syn, aux_logits, logits_center
        return logits_syn
