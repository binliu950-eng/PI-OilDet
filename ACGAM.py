import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdaptiveGaborAttention(nn.Module):
    def __init__(self, in_channels, num_filters=8, kernel_size=15):
        super(AdaptiveGaborAttention, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        # --- 可学习的Gabor参数 ---
        self.thetas = nn.Parameter(torch.Tensor(num_filters))
        self.lambdas = nn.Parameter(torch.Tensor(num_filters))
        self.sigmas = nn.Parameter(torch.Tensor(num_filters))
        self.psis = nn.Parameter(torch.Tensor(num_filters))  # 相位偏移

        # 初始化参数
        nn.init.uniform_(self.thetas, 0, math.pi)
        nn.init.uniform_(self.lambdas, 2.0, 10.0)
        nn.init.uniform_(self.sigmas, 1.0, 5.0)
        nn.init.uniform_(self.psis, 0, math.pi)

        # --- 注意力机制层 ---
        self.attention_conv = nn.Sequential(
            nn.Conv2d(num_filters, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.sigmoid = nn.Sigmoid()

    def _gabor_kernel(self, theta, lambda_val, sigma, psi):
        kernel_size = self.kernel_size
        center = kernel_size // 2

        # 坐标网格
        x_grid, y_grid = torch.meshgrid(
            torch.linspace(-center, center, kernel_size),
            torch.linspace(-center, center, kernel_size),
            indexing='ij'
        )
        x_grid, y_grid = x_grid.to(theta.device), y_grid.to(theta.device)

        x_theta = x_grid * torch.cos(theta) + y_grid * torch.sin(theta)
        y_theta = -x_grid * torch.sin(theta) + y_grid * torch.cos(theta)

        # 高斯
        gb = torch.exp(-.5 * (x_theta ** 2 / sigma ** 2 + y_theta ** 2 / sigma ** 2))

        # 正弦
        sinusoid = torch.cos(2 * math.pi * x_theta / lambda_val + psi)

        gabor = gb * sinusoid
        return gabor

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # 生成可学习的Gabor核
        gabor_kernels = []
        for i in range(self.num_filters):
            kernel = self._gabor_kernel(self.thetas[i], self.lambdas[i], self.sigmas[i], self.psis[i])
            gabor_kernels.append(kernel)

        # Gabor核堆叠
        gabor_kernels = torch.stack(gabor_kernels).unsqueeze(1)

        x_mono = x.mean(dim=1, keepdim=True)  # 形状: [批次大小, 1, 高度, 宽度]
        gabor_responses = F.conv2d(x_mono, gabor_kernels, padding='same')
        # 通道注意力
        channel_attention = self.attention_conv(gabor_responses)

        channel_attention_map = self.sigmoid(channel_attention)

        enhanced_x = x * channel_attention_map + x

        return enhanced_x