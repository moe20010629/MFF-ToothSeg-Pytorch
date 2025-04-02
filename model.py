import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable


# def Adj_matrix_gen(face):
#     B, N = face.shape[0], face.shape[1]
#     adj = (face.repeat(1, 1, N).view(B, N*N, 3) == face.repeat(1, N, 1))
#     adj = adj[:, :, 0] + adj[:, :, 1] + adj[:, :, 2]
#     adj = adj.view(B, N, N)
#     adj = torch.where(adj == True, 1., 0.)
#
#     return adj

class MGM(nn.Module):
    def __init__(self, dim=64, k1=40, k2=20):
        super(MGM, self).__init__()
        self.dim = dim
        self.k1 = k1
        self.k2 = k2

        self.bn1 = nn.BatchNorm2d(self.dim)
        self.bn2 = nn.BatchNorm2d(self.dim)

        self.conv1 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn3 = nn.BatchNorm2d(self.dim)
        self.bn4 = nn.BatchNorm2d(self.dim)
        self.bn5 = nn.BatchNorm2d(self.dim)

        self.conv3 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                    self.bn3,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                                    self.bn4,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(self.dim * 2, self.dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x_knn1 = get_graph_feature(x, k=self.k1)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x_knn1 = self.conv1(x_knn1)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x_knn1 = self.conv2(x_knn1)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x_k1 = x_knn1.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x_knn2 = get_graph_feature(x, self.k2)
        x_knn2 = self.conv3(x_knn2)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x_knn2 = self.conv4(x_knn2)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x_k1 = x_k1.unsqueeze(-1).repeat(1, 1, 1, self.k2)

        out = torch.cat([x_knn2, x_k1], dim=1)

        out = self.conv5(out)
        out = out.max(dim=-1, keepdim=False)[0]

        return out

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)

def Adj_matrix_gen(face):
    B, N = face.shape[0], face.shape[1]
    adj_1_1 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 0] == face.repeat(1, N, 1)[:, :, 0])
    adj_1_2 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 0] == face.repeat(1, N, 1)[:, :, 1])
    adj_1_3 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 0] == face.repeat(1, N, 1)[:, :, 2])
    adj_2_1 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 1] == face.repeat(1, N, 1)[:, :, 0])
    adj_2_2 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 1] == face.repeat(1, N, 1)[:, :, 1])
    adj_2_3 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 1] == face.repeat(1, N, 1)[:, :, 2])
    adj_3_1 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 2] == face.repeat(1, N, 1)[:, :, 0])
    adj_3_2 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 2] == face.repeat(1, N, 1)[:, :, 1])
    adj_3_3 = (face.repeat(1, 1, N).view(B, N * N, 3)[:, :, 2] == face.repeat(1, N, 1)[:, :, 2])
    adj = adj_1_1 + adj_1_2 + adj_1_3 + adj_2_1 + adj_2_2 + adj_2_3 + adj_3_1 + adj_3_2 + adj_3_3
    adj = adj.view(B, N, N)
    adj = torch.where(adj >= 1, 1., 0.)

    return adj

class AFF(nn.Module):
    '''这两个序列包含两层卷积操作，分别将通道数从 channels 减少到 inter_channels，再恢复回 channels。'''
    def __init__(self, channels=256, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1),
            nn.BatchNorm1d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )

        self.global_att = nn.Sequential(
            # nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1),
            nn.BatchNorm1d(inter_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # xa = x + y
        xl = self.local_att(x)
        xg = self.global_att(y)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = x * wei + y * (1 - wei)
        return xo

class graph(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size):
        super(graph, self).__init__()
        if kernel_size==1:
            self.conv = nn.Sequential(nn.Conv1d(inchannel, outchannel, kernel_size=1),
                                   nn.BatchNorm1d(outchannel),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv1d(outchannel, outchannel, kernel_size=1))
        if kernel_size==3:
            self.conv = nn.Sequential(nn.Conv1d(inchannel, outchannel, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(outchannel),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv1d(outchannel, outchannel, kernel_size=1))
        if kernel_size==5:
            self.conv = nn.Sequential(nn.Conv1d(inchannel, outchannel, kernel_size=5, padding=2),
                                   nn.BatchNorm1d(outchannel),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv1d(outchannel, outchannel, kernel_size=1))
    def forward(self, x, adj):
        x = self.conv(x) @ adj

        return x

class GCN(nn.Module):
    def __init__(self, inchannel, outchannel, dim, depth, kernel_size):
        super(GCN, self).__init__()
        self.gcn = nn.ModuleList([
            graph(dim, dim, kernel_size)
            for i in range(depth)])
        self.head = nn.Sequential(nn.Conv1d(inchannel, outchannel, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(outchannel),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.tail = nn.Sequential(nn.Conv1d(outchannel, outchannel, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(outchannel),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, adj):
        x = self.head(x)
        shortcut = x
        # x 进入多个图卷积层，这些层根据邻接矩阵 adj 进行迭代更新，以此来模拟图结构中的信息传递和特征提取。
        for g in self.gcn:
            x = g(x, adj)
        x = self.tail(x) + shortcut

        return x

class SpatialAttention(nn.Module):
    """Simple Spatial Attention module for 1D features"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # Use Conv1d to generate attention map across points
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, N]
        att_map = self.conv(x) # [B, 1, N]
        return self.sigmoid(att_map) # Spatial weights [B, 1, N]

class ChannelAttentionSE(nn.Module):
    """Simple Channel Attention (SE-like) module for 1D features"""
    def __init__(self, channels, reduction=4):
        super(ChannelAttentionSE, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1) # Global Average Pooling -> [B, C, 1]
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, N]
        y = self.pool(x)    # [B, C, 1]
        y = self.fc(y)      # [B, C, 1] (Channel weights)
        return y # Shape [B, C, 1]

class CrossFusionAttention(nn.Module):
    """
    Implements the fusion described in equations 8, 9, 10.
    Cross-spatial attention, residual connections, summation, and channel attention.
    """
    def __init__(self, channels, reduction=4):
        super(CrossFusionAttention, self).__init__()
        self.channels = channels

        # Spatial Attention generators (one for coords, one for normals)
        self.spatial_att_c = SpatialAttention(channels)
        self.spatial_att_n = SpatialAttention(channels)

        # Channel Attention module
        self.channel_att = ChannelAttentionSE(channels, reduction)

        # Note: 'R' for restoring dimensions seems handled by keeping channels consistent

    def forward(self, x_c, x_n):
        """
        Args:
            x_c: Coordinate features [B, C, N]
            x_n: Normal features [B, C, N]
        Returns:
            f_final: Fused features [B, C, N]
        """
        # --- Spatial Attention Cross-Calibration (Eq 8 & 9 variant) ---
        att_map_c = self.spatial_att_c(x_c) # [B, 1, N], spatial attention from coords
        att_map_n = self.spatial_att_n(x_n) # [B, 1, N], spatial attention from normals

        # Recalibrate normal features using coordinate spatial attention
        # x_tilde_n = x_n ⊙ A_s(x_c) (Eq 8)
        x_tilde_n = x_n * att_map_c # Broadcasting [B, 1, N]

        # Recalibrate coordinate features using normal spatial attention
        # x_tilde_c = x_c ⊙ A_s(x_n) (Eq 9)
        x_tilde_c = x_c * att_map_n # Broadcasting [B, 1, N]

        # --- Residual Connection ---
        # Aggregate with original features
        x_prime_n = x_n + x_tilde_n
        x_prime_c = x_c + x_tilde_c

        # --- Element-wise Summation Fusion ---
        # Fuse the enhanced coordinate and normal features
        f_sum = x_prime_c + x_prime_n # [B, C, N]

        # --- Channel Attention Refinement (Eq 10) ---
        # F = A_c(F_sum) ⊙ F_sum
        att_map_channel = self.channel_att(f_sum) # [B, C, 1]
        f_final = f_sum * att_map_channel # Broadcasting [B, C, 1]

        return f_final # [B, C, N]
class Ourmethod(nn.Module):
    def __init__(self, in_channels=12, output_channels=8):
        super(Ourmethod, self).__init__()
        feature_dim = 128

        self.bn1_c = nn.BatchNorm1d(64)
        self.conv1_c = nn.Sequential(nn.Conv1d(12, 64, kernel_size=1, bias=False),
                                   self.bn1_c, nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv1d(64, 64, kernel_size=1, bias=False))
        self.graphblock_c = MGM(dim=64, k1=40, k2=20)
        self.conv2_c = nn.Sequential(nn.Conv1d(64, feature_dim, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(feature_dim), nn.LeakyReLU(negative_slope=0.2),
                                     nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=False))


        self.bn1_n = nn.BatchNorm1d(64)
        self.conv1_n = nn.Sequential(nn.Conv1d(12, 64, kernel_size=1, bias=False),
                                   self.bn1_n, nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv1d(64, 64, kernel_size=1, bias=False))
        self.gcn_nor_1_1 = GCN(64, 128, 128, 2, 1)
        self.gcn_nor_2_1 = GCN(128, 256, 256, 2, 1)
        self.gcn_nor_3_1 = GCN(256, 512, 512, 2, 1)
        self.gcn_nor_4_1 = GCN(512, 512, 512, 2, 1)
        self.reduce_1 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.2))
        self.reduce_2 = nn.Sequential(nn.Conv1d(256, feature_dim, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(feature_dim), nn.LeakyReLU(negative_slope=0.2))

        self.CFM = CrossFusionAttention(channels=feature_dim, reduction=4) # Using 128 channels

        self.pred3 = nn.Sequential(nn.Linear(feature_dim, feature_dim, bias=False), # Changed from 256 to feature_dim (128)
                                   nn.LeakyReLU(negative_slope=0.2))
        self.pred4 = nn.Sequential(nn.Linear(feature_dim, output_channels, bias=False))


    # Modify the forward method
    def forward(self, x, index_face):
        # Adj matrix generation and stream processing (as before)
        adj = Adj_matrix_gen(index_face)
        # adj = adj @ adj # Optional diffusion

        coor_input = x[:, :12, :]
        nor_input = x[:, 12:24, :] # Assuming 24 input features total

        coor = self.conv1_c(coor_input)
        coor = self.graphblock_c(coor)
        coor = self.conv2_c(coor)     # [B, 128, N]

        nor = self.conv1_n(nor_input)
        nor1_1 = self.gcn_nor_1_1(nor, adj)
        nor2_1 = self.gcn_nor_2_1(nor1_1, adj)
        nor3_1 = self.gcn_nor_3_1(nor2_1, adj)
        nor4_1 = self.gcn_nor_4_1(nor3_1, adj)
        x1 = self.reduce_1(nor4_1)
        x1 = self.reduce_2(x1)

        x = self.CFM(coor, x1)

        # Reshape for prediction head
        x = x.transpose(-1, -2)
        x = self.pred3(x)
        score = self.pred4(x)
        score = F.log_softmax(score, dim=2)

        return score
# class Ourmethod(nn.Module):
#
#     def __init__(self, in_channels=12, output_channels=8):
#         super(Ourmethod, self).__init__()
#         # self.k = k
#         ''' coordinate stream '''
#         self.bn1_c = nn.BatchNorm1d(64)
#         self.bn2_c = nn.BatchNorm1d(256)
#         self.bn3_c = nn.BatchNorm1d(512)
#         self.conv1_c = nn.Sequential(nn.Conv1d(12, 64, kernel_size=1, bias=False),
#                                    self.bn1_c,
#                                    nn.LeakyReLU(negative_slope=0.2),
#                                    nn.Conv1d(64, 64, kernel_size=1, bias=False))
#         self.graphblock_c = MGM(dim=64, k1=40, k2=20)
#
#         self.conv1_n = nn.Sequential(nn.Conv1d(12, 64, kernel_size=1, bias=False),
#                                    self.bn1_c,
#                                    nn.LeakyReLU(negative_slope=0.2),
#                                    nn.Conv1d(64, 64, kernel_size=1, bias=False))
#         self.conv2_c = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
#                                      nn.BatchNorm1d(128),
#                                      nn.LeakyReLU(negative_slope=0.2),
#                                      nn.Conv1d(128, 128, kernel_size=1, bias=False))
#
#         '''STNkd空间学习网络：用于学习一个优化的局部坐标框架，增强模型对输入数据变换的鲁棒性 '''
#         self.FTM_c1 = STNkd(k=12)
#         self.FTM_n1 = STNkd(k=12)
#
#         # self.fusion = nn.Conv1d(128, 256, kernel_size=1)
#
#
#         self.aff_nor_1 = AFF(128, 0.5)
#         self.aff_nor_2 = AFF(256, 0.5)
#         self.aff_nor_3 = AFF(512, 0.5)
#         self.aff_nor_4 = AFF(512, 0.5)
#
#         self.aff_1 = AFF(128, 1)
#         self.aff_2 = AFF(256, 1)
#         self.aff_3 = AFF(512, 1)
#         self.aff_4 = AFF(512, 1)
#
#
#         self.gcn_nor_1_1 = GCN(64, 128, 128, 2, 1)
#         self.gcn_nor_1_2 = GCN(64, 128, 128, 2, 1)
#
#
#         self.gcn_nor_2_1 = GCN(128, 256, 256, 2, 1)
#         self.gcn_nor_2_2 = GCN(128, 256, 256, 2, 1)
#
#         self.gcn_nor_3_1 = GCN(256, 512, 512, 2, 1)
#         self.gcn_nor_3_2 = GCN(256, 512, 512, 2, 1)
#
#         self.gcn_nor_4_1 = GCN(512, 512, 512, 2, 1)
#         self.gcn_nor_4_2 = GCN(512, 512, 512, 2, 1)
#
#         self.fu_1 = nn.Sequential(nn.Conv1d(128+256, 512, kernel_size=1, bias=False),
#                                    self.bn3_c,
#                                    nn.LeakyReLU(negative_slope=0.2),
#                                    nn.Conv1d(512, 512, kernel_size=1, bias=False))
#
#         self.fu_2 = nn.Sequential(nn.Conv1d(512+512, 512, kernel_size=1, bias=False),
#                                   self.bn3_c,
#                                   nn.LeakyReLU(negative_slope=0.2),
#                                   nn.Conv1d(512, 512, kernel_size=1, bias=False))
#
#         self.fu_3 = nn.Sequential(nn.Conv1d(64+64, 256, kernel_size=1, bias=False),
#                                   self.bn2_c,
#                                   nn.LeakyReLU(0.2),
#                                   nn.Conv1d(256, 256, kernel_size=1, bias=False))
#
#         self.reduce_1 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
#                                         nn.BatchNorm1d(256),
#                                         nn.LeakyReLU(negative_slope=0.2))
#         self.reduce_2 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
#                                       nn.BatchNorm1d(128),
#                                       nn.LeakyReLU(negative_slope=0.2))
#         self.reduce_3 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
#                                       nn.BatchNorm1d(64),
#                                       nn.LeakyReLU(negative_slope=0.2))
#         '''feature-wise attention'''
#
#         self.fa_1 = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
#                                 nn.BatchNorm1d(1024),
#                                 nn.LeakyReLU(0.2))
#         # self.fa_2 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, bias=False),
#         #                         self.bn2_c,
#         #                         nn.LeakyReLU(0.2),
#         #                         nn.Conv1d(64, 64, kernel_size=1, bias=False))
#
#         ''' feature fusion '''
#         self.SE = SE_Block(256, 16)
#         self.CFM = CFM()
#         self.pred1 = nn.Sequential(nn.Linear(1024, 512, bias=False),
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.pred2 = nn.Sequential(nn.Linear(512, 256, bias=False),
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.pred3 = nn.Sequential(nn.Linear(256, 128, bias=False),
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.pred4 = nn.Sequential(nn.Linear(128, output_channels, bias=False))
#         self.dp1 = nn.Dropout(p=0.6)
#         self.dp2 = nn.Dropout(p=0.6)
#         self.dp3 = nn.Dropout(p=0.6)
#
#
#     def forward(self, x, index_face):
#         adj = Adj_matrix_gen(index_face)
#         adj = adj @ adj
#         coor = x[:, :12, :]
#         nor = x[:, 12:, :]
#
#         coor = self.conv1_c(coor)
#         coor = self.graphblock_c(coor)
#         coor = self.conv2_c(coor)  # 128
#
#         nor = self.conv1_n(nor) # 64
#
#         nor1_1= self.gcn_nor_1_1(nor, adj)
#
#         nor2_1 = self.gcn_nor_2_1(nor1_1, adj)
#
#         nor3_1 = self.gcn_nor_3_1(nor2_1, adj)
#
#         nor4_1 = self.gcn_nor_4_1(nor3_1, adj)
#
#         x1 = self.reduce_1(nor4_1)
#         x1 = self.reduce_2(x1)
#
#         # x = torch.cat((coor, x1), dim=1).unsqueeze(-1)
#
#         x = self.CFM(coor, x1).squeeze(-1)
#         x = x.transpose(-1, -2)
#
#         # 256到128，pred4输出任务结果
#         x = self.pred3(x)
#
#         score = self.pred4(x)
#         score = F.log_softmax(score, dim=2)
#         return score
