from sympy.abc import alpha

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

from exp.FRFTVisualizer import FRFTVisualizer

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import seaborn as sns
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model


    # =============================
    # 1. 专家决策与多尺度融合可视化 (MoE Explanation)
    # =============================
    def visualize_moe_decision(self, sample_input, var_idx=0):
        """
        可视化：不同尺度的专家分别学到了什么，以及 MoE 路由是如何选择的。
        """
        self.model.eval()
        with torch.no_grad():
            # 标准化处理 (与模型 forward 保持一致)
            mean = sample_input.mean(1, keepdim=True)
            std = sample_input.std(1, keepdim=True) + 1e-5
            x_norm = (sample_input - mean) / std
            x_in = x_norm.permute(0, 2, 1).to(self.device)  # [B, C, T]

            # --- 核心修复：遍历专家列表 ---
            scale_outputs = []
            for expert in self.model.frft_experts:
                # 调用专家的 forward，它会返回 (重构信号, loss)
                out, _ = expert(x_in)
                scale_outputs.append(out[0, var_idx].cpu().numpy())

            # 提取 MoE Gate 权重
            # router 输入是 [B, C, T]，输出 [B, C, num_scales]
            gates = torch.softmax(self.model.scale_router.router(x_in), dim=-1)
            gate_weights = gates[0, var_idx].cpu().numpy()

            # 模拟融合过程
            stacked = np.stack(scale_outputs, axis=-1)  # [T, num_scales]
            fused_res = (stacked * gate_weights).sum(axis=-1)

        # --- 绘图部分 (保持高级感) ---
        try:
            plt.style.use('seaborn-v0_8-talk')
        except OSError:
            plt.style.use('seaborn-talk' if 'seaborn-talk' in plt.style.available else 'default')  # 使用更粗犷清晰的学术风格
        fig = plt.figure(figsize=(14, 12))
        gs = plt.GridSpec(len(self.model.patch_sizes) + 1, 2, width_ratios=[3, 1])

        colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']

        # 左上：输入与融合结果对比
        ax_main = fig.add_subplot(gs[0, 0])
        ax_main.plot(x_in[0, var_idx].cpu().numpy(), color='#FFA500', alpha=1, linestyle='--', linewidth=4.0,
                     label='Input')
        ax_main.plot(fused_res, color='black', linewidth=1.5, label='MoE Fused Output')
        ax_main.set_title(f"Variable {var_idx}: Multi-Scale Reconstruction", fontsize=14, fontweight='bold')
        ax_main.legend()

        # 右上：权重分布
        ax_bar = fig.add_subplot(gs[0, 1])
        labels = [f'Scale {p}' for p in self.model.patch_sizes]
        ax_bar.bar(labels, gate_weights, color=colors[:len(labels)], alpha=0.8)
        ax_bar.set_ylim(0, 1.1)
        ax_bar.set_title("Expert Confidence", fontsize=12)

        # 下方：展示每个专家的细节
        for i, p_size in enumerate(self.model.patch_sizes):
            ax = fig.add_subplot(gs[i + 1, :])
            ax.plot(x_in[0, var_idx].cpu().numpy(), color='#FFA500', alpha=1, linestyle='--',linewidth=4.0)
            ax.plot(scale_outputs[i], color=colors[i], label=f'Expert (Patch={p_size})')
            ax.set_ylabel("Amp")
            ax.legend(loc='upper right')
            if i == len(self.model.patch_sizes) - 1: ax.set_xlabel("Time Steps")

        plt.tight_layout()
        plt.savefig(f"moe_decision_var{var_idx}.png", dpi=300)
        plt.close(fig)
    # =============================
    # 2. 复数物理空间与 FRFT 集中度可视化
    # =============================
    def visualize_complex_interpretation(self, sample_input, var_idx=0):
        """
        可视化：FRFT 旋转能量分布与复平面物理特性
        """
        self.model.eval()
        with torch.no_grad():
            # 前置处理
            mean = sample_input.mean(1, keepdim=True)
            std = sample_input.std(1, keepdim=True) + 1e-5
            x_norm = (sample_input - mean) / std
            x_in = x_norm.permute(0, 2, 1).to(self.device)

            # --- 核心修复：从第一个专家获取 alpha ---
            first_expert = self.model.frft_experts[0]
            alpha = first_expert.current_alpha().detach()
            alpha_value = alpha.item()
            learned_alphas = [expert.current_alpha().item() for expert in self.model.frft_experts]
            print('learned actual alphas:', learned_alphas)

            # 1. 观察 FRFT 域能量 (使用第一个专家的变换方法)
            # 注意：FRFT 变换通常在全长上观察更有论文说服力
            x_freq = first_expert.frft_3d(x_in, alpha.to(self.device))[0, var_idx]

            # 2. 获取重构后的复数信号 (包含 iFRFT 后的实部和虚部)
            # 我们看全局尺度 (Patch 96) 的处理结果
            x_patch_recon_complex = []
            # 这里为了演示，我们手动跑一下第一个专家的流程
            exp = first_expert
            # 简化版流程提取
            x_f = exp.frft_3d(x_in, alpha.to(self.device))
            period, residual = exp.period_masker(x_f)
            res_mixed = exp.spectral_mixer_2(exp.spectral_mixer_1(residual)) + residual
            x_and = res_mixed + period
            # iFRFT 还原
            x_recon_c = exp.frft_3d(x_and, -alpha.to(self.device))[0, var_idx]

        # --- 开始绘图 ---
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # A. FRFT 频谱图 (Stem Plot)
        axes[0, 0].stem(torch.abs(x_freq).cpu().numpy(), markerfmt=' ', basefmt="C0-", label='Magnitude')
        axes[0, 0].set_title(f"A. FRFT Domain (α={alpha_value:.2f})", loc='left', fontweight='bold')
        axes[0, 0].set_ylabel("Intensity")

        # B. 复平面轨迹 (Phase Portrait)
        re = x_freq.real.cpu().numpy()
        im = x_freq.imag.cpu().numpy()
        # 使用渐变色表示时间流向
        points = np.array([re, im]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='viridis', alpha=0.6)
        lc.set_array(np.linspace(0, 1, len(re)))
        axes[0, 1].add_collection(lc)
        axes[0, 1].autoscale()
        axes[0, 1].set_title("B. Complex Phase Trajectory", loc='left', fontweight='bold')
        axes[0, 1].set_xlabel("Real");
        axes[0, 1].set_ylabel("Imag")
        axes[0, 1].grid(True, alpha=0.3)

        # C. 实虚部解耦对比
        axes[1, 0].plot(x_recon_c.real.cpu().numpy(), color='#1f77b4', label='Real (Signal)')
        axes[1, 0].plot(x_recon_c.imag.cpu().numpy(), color='#d62728', alpha=0.7, label='Imag (Residual)')
        axes[1, 0].set_title("C. Component Decoupling", loc='left', fontweight='bold')
        axes[1, 0].legend()

        # D. 虚部能量密度 (证明物理约束有效)
        import seaborn as sns
        sns.kdeplot(x_recon_c.imag.cpu().numpy(), ax=axes[1, 1], fill=True, color='#d62728')
        axes[1, 1].set_title("D. Imaginary Residual Density", loc='left', fontweight='bold')
        axes[1, 1].set_xlabel("Value")

        plt.tight_layout()
        plt.savefig(f"physics_analysis_var{var_idx}.png", dpi=300)
        plt.close(fig)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        alpha_lr = getattr(self.args, 'alpha_learning_rate', 0.0)
        weight_decay = getattr(self.args, 'weight_decay', 0.0)

        if alpha_lr and alpha_lr > 0:
            alpha_params = []
            other_params = []
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if name.endswith('alpha_raw') or '.alpha_raw' in name:
                    alpha_params.append(parameter)
                else:
                    other_params.append(parameter)

            param_groups = []
            if other_params:
                param_groups.append({
                    'params': other_params,
                    'lr': self.args.learning_rate,
                    'initial_lr': self.args.learning_rate,
                    'name': 'main'
                })
            if alpha_params:
                param_groups.append({
                    'params': alpha_params,
                    'lr': alpha_lr,
                    'initial_lr': alpha_lr,
                    'name': 'alpha_raw'
                })
                print(f'Using separate alpha_raw learning rate: {alpha_lr}')

            model_optim = optim.Adam(param_groups, weight_decay=weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=weight_decay)
        return model_optim

    def _select_criterion(self):
        loss_name = str(getattr(self.args, 'loss', 'MAE')).upper()
        if loss_name == 'MSE':
            criterion = nn.MSELoss()
        elif loss_name in ('HUBER', 'SMOOTHL1'):
            criterion = nn.SmoothL1Loss(beta=1.0)
        elif loss_name in ('MAE_MSE', 'MIXED'):
            criterion = lambda pred, true: 0.5 * nn.functional.l1_loss(pred, true) + 0.5 * nn.functional.mse_loss(pred, true)
        else:
            criterion = nn.L1Loss()
        return criterion
    # def _select_criterion(self):
    #     criterion = MAE_MSE_Loss(alpha=1.0, beta=1.0)
    #     return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                imag_loss = torch.zeros((), device=self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs, imag_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, imag_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = criterion(outputs, batch_y) + imag_loss

                total_loss.append(loss.detach().cpu().item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        # if test:
        #     print('loading model')
        #     self.model.load_state_dict(
        #         torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))
        #
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param
        print(f"Total Trainable Params: {total_params}")

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, imag_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y) + imag_loss
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # 转换为MB
            # max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # 转换为MB
            # reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # 转换为MB
            # print(f" Allocated {allocated:.2f} MB, Max Allocated {max_allocated:.2f} MB, Reserved {reserved:.2f} MB")

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        # if test:
        #     print('loading model')
        #     self.model.load_state_dict(
        #         torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))

        ##################################################################################################
        ####################
        batch_x, _, _, _ = next(iter(test_loader))
        # 转换为 (B, V, T) 格式
        sample_input = batch_x[0:1].float().to(self.device).permute(0, 2, 1)

        # 2. 获取模型学习到的 alpha 值 (假设你的模型有 self.alpha 参数)
        learned_alphas = [expert.current_alpha().item() for expert in self.model.frft_experts]
        print('learned actual alphas:', learned_alphas)
        if hasattr(self.model, 'current_betas'):
            learned_betas = self.model.current_betas().cpu().numpy().tolist()
            print('learned physical betas [ratio, diff, abs]:', learned_betas)
        # 3. 绘图对比：展示 0(时域), 1(频域) 以及模型自适应学习到的角度
        if os.environ.get("PHYFSME_SKIP_VIS", "0") == "1":
            print("Skip FRFT visualizations because PHYFSME_SKIP_VIS=1.")
        else:
            first_patch_len = getattr(self.model.frft_experts[0], "patch_len", self.args.seq_len)
            if first_patch_len == self.args.seq_len:
                self.visualize_complex_interpretation(sample_input.permute(0, 2, 1),  var_idx=0)
            else:
                print(f"Skip complex visualization because first_patch_len={first_patch_len} differs from seq_len={self.args.seq_len}.")

            # self.visualize_moe_decision(sample_input.permute(0, 2, 1), var_idx=0)

            visualizer = FRFTVisualizer()
            visualizer.plot_dual_analysis(sample_input.permute(0, 2, 1))
            # visualizer.plot_3d_manifold(sample_input.permute(0, 2, 1))


        ###########
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs, imag_loss = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return


