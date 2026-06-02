import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

from layers.RevIN import RevIN


# =============================
# Adaptive Spectral Masker 这里可以考虑换一个名字
# =============================
class AdaptiveSpectralMasker(nn.Module):
    def __init__(self, enc_in, freq_len, d_model):
        super().__init__()
        self.mask_net = nn.Sequential(
            nn.Linear(freq_len, d_model),
            nn.SiLU(),
            nn.Linear(d_model, freq_len),
            nn.Sigmoid()
        )

    def forward(self, x_freq):
        mag = torch.abs(x_freq)
        phase = torch.angle(x_freq)
        mask = self.mask_net(mag)
        period_mag = mag * mask
        residual_mag = mag * (1 - mask)
        period = period_mag * torch.exp(1j * phase)
        residual = residual_mag * torch.exp(1j * phase)
        return period, residual


# =============================
# ModReLU (保持不变)
# =============================
class ModReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        mag = torch.abs(x)
        scale = torch.relu(mag + self.b) / (mag + 1e-6)
        return x * scale


# =============================
# Complex Channel Mixer (保持不变)
# =============================
class ComplexChannelMixer(nn.Module):
    def __init__(self, enc_in, d_model, dropout):
        super().__init__()
        self.W_re = nn.Linear(enc_in, d_model)
        self.W_im = nn.Linear(enc_in, d_model)
        self.V_re = nn.Linear(d_model, enc_in)
        self.V_im = nn.Linear(d_model, enc_in)
        self.act = ModReLU()
        # self.dropout = nn.Dropout(dropout)
        # self.norm = nn.LayerNorm(enc_in)

    def complex_linear(self, x, W_re, W_im):
        real = W_re(x.real) - W_im(x.imag)
        imag = W_re(x.imag) + W_im(x.real)
        return torch.complex(real, imag)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        res = x
        out = self.complex_linear(x, self.W_re, self.W_im)
        out = self.act(out)
        out = self.complex_linear(out, self.V_re, self.V_im)
        combined = res + out
        return combined.permute(0, 2, 1)


# ============================================================
# 【核心修改 1】自适应单尺度 FRFT 专家块 (Heterogeneous Expert)
# ============================================================
class FRFTScaleBlock(nn.Module):
    def __init__(self, enc_in, patch_len, d_model, dropout, init_alpha, beta1, beta2, beta3,
                 learnable_beta=True, beta_mode="free"):
        """
        处理单一尺度 Patch，并拥有独立的、可学习的分数阶次 alpha
        """
        super().__init__()
        self.patch_len = patch_len

        # 【关键修改】：将 alpha 变为可学习参数，每个尺度的专家维护自己的旋转角
        # Parameterize the actual FRFT order in (0, 1) while preserving
        # the user-facing meaning of init_alpha as the initial order.
        init_alpha = torch.tensor(float(init_alpha), dtype=torch.float32)
        init_alpha = torch.clamp(init_alpha, 1e-4, 1.0 - 1e-4)
        self.alpha_raw = nn.Parameter(torch.logit(init_alpha).view(1))

        # self.alpha = torch.tensor([float(init_alpha)]).cuda()
        # self.alpha2 = nn.Parameter(torch.tensor(0.0))

        init_betas = torch.tensor([beta1, beta2, beta3], dtype=torch.float32).clamp_min(1e-8)
        self.learnable_beta = bool(learnable_beta)
        self.beta_mode = beta_mode
        if self.learnable_beta:
            if self.beta_mode == "proportional":
                # Learn relative proportions while keeping total physical strength fixed.
                beta_total = init_betas.sum()
                beta_probs = init_betas / beta_total
                self.register_buffer("beta_total", beta_total)
                self.beta_logits = nn.Parameter(torch.log(beta_probs))
            else:
                # Learn positive beta values directly; the initial CLI betas are only priors.
                self.beta_raw = nn.Parameter(torch.log(torch.expm1(init_betas)))
        else:
            self.register_buffer("fixed_betas", init_betas)

        self.period_masker = AdaptiveSpectralMasker(enc_in, patch_len, d_model)
        self.spectral_mixer_1 = ComplexChannelMixer(enc_in, d_model, dropout)
        self.spectral_mixer_2 = ComplexChannelMixer(enc_in, d_model, dropout)

    def forward(self, x):
        B, C, T = x.shape
        num_patches = T // self.patch_len

        # if self.patch_len == 96:
        #     print(self.alpha)

        # 1. 划分为 Patch
        x_patch = x.reshape(B, C, num_patches, self.patch_len)
        x_patch = x_patch.permute(0, 2, 1, 3).contiguous()
        x_patch = x_patch.reshape(B * num_patches, C, self.patch_len)

        # 2. FRFT 正变换 (使用该专家独立的可学习 alpha)
        # 限制 alpha 在合理的范围内防止数值不稳定
        # curr_alpha is always in (0, 1), matching the paper setting.
        curr_alpha = self.current_alpha()
        # curr_alpha = torch.tanh(self.alpha)
        # curr_alpha = self.alpha
        x_freq = self.frft_3d(x_patch, curr_alpha)

        # 3. 频域处理
        period, residual = self.period_masker(x_freq)
        residual = self.spectral_mixer_1(residual)
        residual = self.spectral_mixer_2(residual)
        x_and = residual + period

        # 4. iFRFT 逆变换 (使用 -alpha)
        x_time_complex = self.frft_3d(x_and, -curr_alpha)

        # 计算当前尺度的物理约束 Loss
        loss = self.compute_physical_loss(x_time_complex, curr_alpha)

        x_time_real = x_time_complex.real

        # 5. 还原形状
        x_time_real = x_time_real.reshape(B, num_patches, C, self.patch_len)
        x_time_real = x_time_real.permute(0, 2, 1, 3).contiguous()
        x_time_real = x_time_real.reshape(B, C, T)

        return x_time_real, loss #imag_val

    def compute_physical_loss(self, x, curr_alpha):
        imag = x.imag

        # 保留
        diff_imag = torch.abs(imag[:, :, 1:] - imag[:, :, :-1]).mean()
        imag_abs_loss = torch.mean(torch.abs(imag))

        # 能量
        imag_energy = x.imag.pow(2).mean()
        real_energy = x.real.pow(2).mean()

        # 修改这里（核心）
        total_energy = real_energy + imag_energy + 1e-8
        ratio = imag_energy / total_energy
        tau = curr_alpha.detach()
        loss = torch.relu(ratio - tau)
        # Use the current fractional order as a dynamic threshold, but detach it
        # so this regularizer cannot increase alpha merely to relax the constraint.
        # print(ratio, imag_abs_loss, diff_imag)

        beta1, beta2, beta3 = self.current_betas()
        return loss * beta1 + diff_imag * beta2 + imag_abs_loss * beta3

    def current_alpha(self):
        return torch.sigmoid(self.alpha_raw)

    def current_betas(self):
        if self.learnable_beta:
            if self.beta_mode == "proportional":
                return self.beta_total * torch.softmax(self.beta_logits, dim=0)
            return F.softplus(self.beta_raw)
        return self.fixed_betas

    def frft_3d(self, x, alpha):
        B, V, T = x.shape
        device = x.device
        # 确保 alpha 是标量形式进行计算，但保留梯度
        alpha = alpha % 4
        if alpha > 2: alpha -= 4
        if alpha < -2: alpha += 4

        # 极值情况加速
        if torch.abs(alpha) < 1e-4: return x.to(torch.complex64)
        if torch.abs(alpha - 1.0) < 1e-4:
            return fft.fftshift(fft.fft(fft.ifftshift(x, dim=-1), dim=-1, norm="ortho"), dim=-1)
        if torch.abs(alpha + 1.0) < 1e-4:
            return fft.fftshift(fft.ifft(fft.ifftshift(x, dim=-1), dim=-1, norm="ortho"), dim=-1)

        x_flat = x.to(torch.complex64).reshape(-1, T)
        theta = alpha * torch.pi / 2
        n = torch.arange(T, device=device).float() - T // 2
        t_sq = n ** 2 / T

        c1 = torch.exp(-1j * torch.pi * torch.tan(theta / 2) * t_sq)
        c2 = torch.exp(-1j * torch.pi * torch.sin(theta) * t_sq)
        # A_alpha 补偿
        A_alpha = torch.exp(-1j * (torch.pi * torch.tanh(100 * torch.sin(theta)) / 4 - theta / 2))

        res = x_flat * c1
        res = fft.ifftshift(res, dim=-1)
        res = fft.fft(res, dim=-1, norm="ortho")
        res = fft.fftshift(res, dim=-1)
        res = res * c2
        res = fft.ifftshift(res, dim=-1)
        res = fft.ifft(res, dim=-1, norm="ortho")
        res = fft.fftshift(res, dim=-1)
        res = res * c1 * A_alpha
        return res.reshape(B, V, T)


# =============================
# Scale-MoE Router (保持不变)
# =============================
class ScaleMoE(nn.Module):
    def __init__(self, num_scales, seq_len):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(seq_len, seq_len // num_scales),
            nn.GELU(),
            nn.Linear(seq_len // num_scales, num_scales)
        )

    def forward(self, original_x, scale_outputs):
        gates = self.router(original_x)
        gates = torch.softmax(gates, dim=-1)
        # gates = F.softplus(gates)

        gates = gates.unsqueeze(2)
        stacked_outputs = torch.stack(scale_outputs, dim=-1)
        fused_output = (stacked_outputs * gates).sum(dim=-1)
        return fused_output


# ============================================================
# CRMBranch & DynamicGatedFusion (保持不变)
# ============================================================
class CRMBranch(nn.Module):
    def __init__(self, enc_in, seq_len, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(seq_len, d_model)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        res = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x)
        x = res + self.dropout(attn_out)
        res = x
        x = self.norm2(x)
        x = res + self.dropout(self.ffn(x))
        return x


class DynamicGatedFusion(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.Sigmoid()
        )
        self.out_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x_freq_feature, x_time_feature):
        combined = torch.cat([x_freq_feature, x_time_feature], dim=-1)
        gate = self.gate_net(combined)
        fused_feature = gate * x_freq_feature + (1 - gate) * x_time_feature
        return self.out_proj(fused_feature)


# ============================================================
# 【核心修改 2】主模型：实现尺度与阶次的初始配对
# ============================================================
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.hidden_size

        self.use_revin = configs.use_norm
        self.phys_loss_reduction = getattr(configs, "phys_loss_reduction", "sum")
        self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)

        # 配置多尺度 Patch 长度；可通过 --patch_sizes 做数据集级搜索。
        patch_sizes = getattr(configs, "patch_sizes", "")
        if isinstance(patch_sizes, str) and patch_sizes.strip():
            self.patch_sizes = [int(p.strip()) for p in patch_sizes.split(",") if p.strip()]
        else:
            self.patch_sizes = [self.seq_len, self.seq_len // 4, self.seq_len // 8]
        self.patch_sizes = [p for p in self.patch_sizes if p > 0 and self.seq_len % p == 0]
        if not self.patch_sizes:
            raise ValueError("patch_sizes must contain at least one positive divisor of seq_len")

        # 【关键修改】：为不同尺度预设不同的初始阶次 alpha
        initial_alphas = getattr(configs, "initial_alphas", "")
        if isinstance(initial_alphas, str) and initial_alphas.strip():
            self.initial_alphas = [float(a.strip()) for a in initial_alphas.split(",") if a.strip()]
        else:
            self.initial_alphas = [0.5] * len(self.patch_sizes)
        if len(self.initial_alphas) < len(self.patch_sizes):
            self.initial_alphas += [self.initial_alphas[-1]] * (len(self.patch_sizes) - len(self.initial_alphas))
        self.initial_alphas = self.initial_alphas[:len(self.patch_sizes)]

        self.frft_experts = nn.ModuleList([
            FRFTScaleBlock(
                enc_in=self.enc_in,
                patch_len=p_size,
                d_model=self.d_model,
                dropout=configs.dropout,
                init_alpha=self.initial_alphas[i],  # 差异化初始化
                beta1=configs.beta1,
                beta2=configs.beta2,
                beta3=configs.beta3,
                learnable_beta=getattr(configs, "learnable_beta", 1),
                beta_mode=getattr(configs, "beta_mode", "free")
            ) for i, p_size in enumerate(self.patch_sizes)
        ])

        self.scale_router = ScaleMoE(num_scales=len(self.patch_sizes), seq_len=self.seq_len)
        self.dropout = nn.Dropout(configs.dropout)
        self.predictor = nn.Sequential(nn.Linear(self.seq_len, self.d_model), nn.LeakyReLU())

        self.Time_Interaction_Block = CRMBranch(
            enc_in=self.enc_in, seq_len=self.seq_len, d_model=self.d_model,
            n_heads=configs.heads, dropout=configs.dropout
        )

        self.dynamic_fusion = DynamicGatedFusion(self.d_model)
        self.final_projector = nn.Linear(self.d_model, self.pred_len)

    def forward(self, x, *args):
        if self.use_revin:
            x = self.revin_layer(x, 'norm')

        x = x.permute(0, 2, 1)  # [B, C, T]

        # Branch 1: 多尺度-多阶次自适应专家系统
        scale_outputs = []
        total_imag_loss = 0
        for expert in self.frft_experts:
            out, loss = expert(x)
            scale_outputs.append(out)
            total_imag_loss += loss
        if self.phys_loss_reduction == "mean":
            total_imag_loss = total_imag_loss / len(self.frft_experts)

        x_freq_fused = self.scale_router(x, scale_outputs)
        freq_hidden = self.dropout(self.predictor(x_freq_fused))

        # Branch 2: 空间流全局时域变量交互
        time_hidden = torch.zeros_like(freq_hidden)

        # 融合与预测
        fused_hidden = self.dropout(self.dynamic_fusion(freq_hidden, time_hidden))
        y = self.final_projector(fused_hidden).permute(0, 2, 1)

        if self.use_revin:
            y = self.revin_layer(y, 'denorm')

        return y, total_imag_loss

    def current_betas(self):
        return torch.stack([expert.current_betas().detach() for expert in self.frft_experts]).mean(dim=0)


