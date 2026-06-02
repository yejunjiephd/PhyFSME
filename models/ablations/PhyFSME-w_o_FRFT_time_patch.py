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
    """Time-domain window expert used for the w/o FRFT ablation.

    This variant keeps the same analysis-window partition and Scale-MoE routing
    interface, but removes both the forward FRFT and inverse FRFT. Each window is
    processed directly in the time domain, making it a strict test of whether the
    fractional time-frequency projection itself is useful.
    """

    def __init__(self, enc_in, patch_len, d_model, dropout, init_alpha, beta1, beta2, beta3,
                 learnable_beta=True, beta_mode="free"):
        super().__init__()
        self.patch_len = patch_len
        self.register_buffer("dummy_alpha", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("dummy_betas", torch.zeros(3, dtype=torch.float32))
        self.time_mixer = nn.Sequential(
            nn.Linear(patch_len, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, patch_len)
        )
        self.channel_mixer = nn.Sequential(
            nn.Linear(enc_in, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, enc_in)
        )
        self.norm_time = nn.LayerNorm(patch_len)
        self.norm_channel = nn.LayerNorm(enc_in)

    def forward(self, x):
        B, C, T = x.shape
        num_patches = T // self.patch_len
        x_patch = x.reshape(B, C, num_patches, self.patch_len)
        x_patch = x_patch.permute(0, 2, 1, 3).contiguous()
        x_patch = x_patch.reshape(B * num_patches, C, self.patch_len)

        # Direct time-domain patch/window modeling: no FRFT and no complex spectrum.
        residual = x_patch
        x_patch = residual + self.time_mixer(self.norm_time(x_patch))

        residual = x_patch
        x_channel = x_patch.permute(0, 2, 1).contiguous()
        x_channel = self.channel_mixer(self.norm_channel(x_channel))
        x_patch = residual + x_channel.permute(0, 2, 1).contiguous()

        x_time = x_patch.reshape(B, num_patches, C, self.patch_len)
        x_time = x_time.permute(0, 2, 1, 3).contiguous()
        x_time = x_time.reshape(B, C, T)
        return x_time, x.new_tensor(0.0)

    def compute_physical_loss(self, x, curr_alpha=None):
        return x.real.new_tensor(0.0) if torch.is_complex(x) else x.new_tensor(0.0)

    def current_alpha(self):
        return self.dummy_alpha

    def current_betas(self):
        return self.dummy_betas


# =============================
# ScaleMoE Router (kept unchanged)
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
        time_hidden = self.Time_Interaction_Block(x)

        # 融合与预测
        fused_hidden = self.dropout(self.dynamic_fusion(freq_hidden, time_hidden))
        y = self.final_projector(fused_hidden).permute(0, 2, 1)

        if self.use_revin:
            y = self.revin_layer(y, 'denorm')

        return y, total_imag_loss

    def current_betas(self):
        return torch.stack([expert.current_betas().detach() for expert in self.frft_experts]).mean(dim=0)


