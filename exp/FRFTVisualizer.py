import torch
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


class FRFTVisualizer:
    def __init__(self, save_dir='./test_results/frft_analysis/'):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 设置学术风格字体
        plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})

    def _frft_core(self, x, alpha):
        """完全对齐模型内部的 FRFT 逻辑"""
        B, V, T = x.shape
        device = x.device
        alpha = alpha % 4
        if alpha > 2: alpha -= 4
        if alpha < -2: alpha += 4

        if alpha == 0: return x.to(torch.complex64)
        if alpha == 1:
            return fft.fftshift(fft.fft(fft.ifftshift(x, dim=-1), dim=-1, norm="ortho"), dim=-1)
        if alpha == -1:
            return fft.fftshift(fft.ifft(fft.ifftshift(x, dim=-1), dim=-1, norm="ortho"), dim=-1)

        x_flat = x.to(torch.complex64).reshape(-1, T)
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
        return res.reshape(B, V, T)

    def plot_dual_analysis(self, batch_x, sample_idx=0, channel_idx=0):
        """
        生成两张核心图：
        1. 2D 多阶次时频对比热力图 (Motivation Fig 1)
        2. 3D 分数阶能量流形图 (Evidence Fig 2)
        """
        # 准备数据 [1, 1, T]
        seq = batch_x[sample_idx:sample_idx + 1, :, channel_idx:channel_idx + 1].permute(0, 2, 1)
        T = seq.shape[-1]

        # --- 1. 绘制 2D 热力图对比 ---
        alphas = [0.0, 1.0, 0.5, 0.8]
        fig2d, axes = plt.subplots(2, 4, figsize=(20, 8))

        for i, a in enumerate(alphas):
            # 顶层：展示当前阶次下的复数模长投影波形
            transformed = self._frft_core(seq, a)
            mag_line = torch.abs(transformed[0, 0]).cpu().numpy()
            axes[0, i].plot(mag_line, color='royalblue', lw=1)
            axes[0, i].set_title(f'Order $\\alpha = {a}$', fontsize=14)
            axes[0, i].grid(True, alpha=0.3)

            # 底层：计算 ST-FRFT (短时分数阶) 展现动态演化
            # 窗口设大一点以捕获周期性
            win_size = min(T // 2, 96)
            stride = 1
            specs = []
            for start in range(0, T - win_size, stride):
                window = seq[:, :, start: start + win_size]
                specs.append(torch.abs(self._frft_core(window, a)[0, 0]).cpu().numpy())

            heatmap = np.array(specs).T
            im = axes[1, i].imshow(heatmap, aspect='auto', cmap='turbo', origin='lower')
            if i == 0: axes[1, i].set_ylabel('Rotated Frequency Index', fontsize=12)
            axes[1, i].set_xlabel('Time Steps', fontsize=12)

        fig2d.suptitle('Multi-order Fractional Spectral Analysis (ETTh1 Example)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        fig2d.savefig(os.path.join(self.save_dir, 'motivation_2d_heatmaps.png'), dpi=300)

        # --- 2. 绘制 3D 能量流形图 ---
        alpha_range = np.linspace(0, 1.0, 60)
        manifold = []
        for a in alpha_range:
            mag = torch.abs(self._frft_core(seq, a)[0, 0]).cpu().numpy()
            manifold.append(mag)

        manifold = np.array(manifold)
        # 关键修正：裁剪极值点，平滑显示效果
        v_max = np.percentile(manifold, 99.5)
        manifold = np.clip(manifold, 0, v_max)

        X, Y = np.meshgrid(np.arange(T), alpha_range)
        fig3d = plt.figure(figsize=(12, 9))
        ax = fig3d.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, manifold, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)

        ax.set_xlabel('Index', labelpad=10)
        ax.set_ylabel(r'Fractional Order $\alpha$', labelpad=10)
        ax.set_zlabel('Magnitude', labelpad=10)
        ax.view_init(elev=30, azim=-60)  # 调整视角以凸显峰值

        plt.title('3D Fractional Spectral Energy Manifold', fontsize=15)
        fig3d.colorbar(surf, shrink=0.5, aspect=10)
        plt.savefig(os.path.join(self.save_dir, 'evidence_3d_manifold.png'), dpi=300, bbox_inches='tight')

        print(f"Visualization saved to {self.save_dir}")
        plt.close('all')