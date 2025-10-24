import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

#重力损失
def Gravity_Loss(output):
    P = F.softmax(output, dim=1)  # [batch_size, num_classes, height, width]
    P1 = P[:, 1, :, :]  # 渗漏油概率
    P_above = P1[:, :-1, :]  # 上方像素
    P_below = P1[:, 1:, :]  # 下方像素
    gravity_loss = (P_above * (1 - P_below)).mean()  # 重力损失
    return gravity_loss

def Focal_Loss(inputs, target, cls_weights, num_classes=3, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

    # 总损失
    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes) + lambda_gravity * Gravity_Loss(outputs)