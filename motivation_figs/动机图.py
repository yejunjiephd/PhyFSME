import torch
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import os

# 设置学术绘图风格
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.autolayout': True
})


def frft_1d(x, alpha):
    """提取自你模型中的核心 FRFT 算子，适配 1D 信号可视化"""
    T = x.shape[-1]
    device = x.device
    alpha = alpha % 4
    if alpha > 2: alpha -= 4
    if alpha < -2: alpha += 4

    if abs(alpha) < 1e-4: return x.to(torch.complex64)
    if abs(alpha - 1.0) < 1e-4:
        return fft.fftshift(fft.fft(fft.ifftshift(x, dim=-1), dim=-1, norm="ortho"), dim=-1)

    x_flat = x.to(torch.complex64)
    theta = torch.tensor(alpha * torch.pi / 2, device=device)
    n = torch.arange(T, device=device).float() - T // 2
    t_sq = n ** 2 / T

    c1 = torch.exp(1j * torch.pi * torch.tan(theta / 2) * t_sq)
    c2 = torch.exp(-1j * torch.pi * torch.sin(theta) * t_sq)
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
    return res


def generate_synthetic_mtsf_signal(T=500):
    """
    生成模拟真实物理系统（如电网）的非平稳信号：
    包含：1. 全局低频宏观周期 (Macro-trend)
          2. 局部高频频率漂移/脉冲 (Transient Chirp)
    """
    t = np.linspace(0, 1, T)
    # 1. 宏观平稳周期 (例如 24 小时恒定负载)
    macro_trend = np.sin(2 * np.pi * 5 * t)

    # 2. 局部非平稳扰动 (例如突发的频率随时间变化的负荷波动)
    # 在 t=[0.4, 0.7] 之间插入一个频率逐渐升高的 Chirp 信号
    transient = np.zeros(T)
    chirp_idx = (t >= 0.4) & (t <= 0.7)
    transient[chirp_idx] = 1.5 * np.sin(2 * np.pi * (15 * t[chirp_idx] + 20 * t[chirp_idx] ** 2))

    # 3. 混合并加入微小噪声
    signal = macro_trend + transient + 0.1 * np.random.randn(T)
    return torch.tensor(signal, dtype=torch.float32), t


def compute_st_frft(signal, alpha, window_size=50, stride=5):
    """短时分数阶傅里叶变换 (用于绘制时频/时阶热力图)"""
    T = signal.shape[0]
    specs = []
    for start in range(0, T - window_size, stride):
        windowed_sig = signal[start: start + window_size]
        spec = torch.abs(frft_1d(windowed_sig, alpha))
        specs.append(spec.numpy())
    return np.array(specs).T


def plot_motivation_figure():
    T = 500
    signal, t = generate_synthetic_mtsf_signal(T)

    # 使用更高分辨率和更紧凑的布局
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.2)

    # ==========================================
    # (a) 尺度错配分析
    # ==========================================
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, signal.numpy(), color='#2878B5', linewidth=1.5, label='Original Signal')
    ax1.set_title("(a) Multi-scale Characteristics of Non-stationary Signals", fontweight='bold', pad=15)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-3.5, 4.5)  # 适当拉高轴向，为文字留出空间
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle='--', alpha=0.3)

    # 优化后的标注：将文字放入框内上方，并增加背景
    rect_props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none')

    # 宏观尺度
    ax1.add_patch(Rectangle((0.05, -2.5), 0.3, 5, fill=True, color='green', alpha=0.05))
    ax1.add_patch(Rectangle((0.05, -2.5), 0.3, 5, fill=False, edgecolor='green', linestyle='--', lw=2))
    ax1.text(0.2, 3.2, "Macro-trend\n(Large Patch Required)", color='green',
             ha='center', fontweight='bold', fontsize=10, bbox=rect_props)

    # 局部尺度
    ax1.add_patch(Rectangle((0.4, -2.5), 0.3, 5, fill=True, color='red', alpha=0.05))
    ax1.add_patch(Rectangle((0.4, -2.5), 0.3, 5, fill=False, edgecolor='red', linestyle='--', lw=2))
    ax1.text(0.55, 3.2, "Transient Chirp\n(Small Patch Required)", color='red',
             ha='center', fontweight='bold', fontsize=10, bbox=rect_props)

    # ==========================================
    # (b) DFT 能量泄露
    # ==========================================
    ax2 = fig.add_subplot(gs[1, 0])
    spec_dft = compute_st_frft(signal, alpha=1.0, window_size=60, stride=2)
    im2 = ax2.imshow(spec_dft, aspect='auto', origin='lower', cmap='inferno', extent=[0, 1, -30, 30])
    ax2.set_title("(b) Standard DFT ($\\alpha = 1.0$)\nEnergy Leakage (Blurred Representation)", fontweight='bold',
                  fontsize=11)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Frequency Index")

    # 增强对比感的标注
    ax2.add_patch(Rectangle((0.4, -22), 0.3, 44, fill=False, edgecolor='cyan', linestyle=':', lw=2))
    ax2.text(0.55, 24, "Smeared Energy", color='cyan', ha='center', fontweight='bold', fontsize=9)

    # ==========================================
    # (c) FRFT 能量聚焦
    # ==========================================
    ax3 = fig.add_subplot(gs[1, 1])
    spec_frft = compute_st_frft(signal, alpha=0.3, window_size=60, stride=2)
    im3 = ax3.imshow(spec_frft, aspect='auto', origin='lower', cmap='inferno', extent=[0, 1, -30, 30])
    ax3.set_title("(c) Optimal FRFT ($\\alpha = 0.3$)\nEnergy Focusing (Compact Representation)", fontweight='bold',
                  fontsize=11)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Rotated Frequency Index")

    # 增强对比感的标注
    ax3.add_patch(Rectangle((0.4, -12), 0.3, 24, fill=False, edgecolor='#00FF00', linestyle='-', lw=2))
    ax3.text(0.55, 14, "Optimal Focusing", color='#00FF00', ha='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig('motivation_figs/PhyFSME_v2.png', bbox_inches='tight')
    print("V2 figure saved.")


if __name__ == "__main__":
    plot_motivation_figure()