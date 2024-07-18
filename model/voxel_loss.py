import torch
import torch.nn as nn
import torch.nn.functional as F
import torch

# class Voxel_loss(nn.Module):
#     def __init__(self, margin, device='cuda'):
#         super(Voxel_loss, self).__init__()
#         self.margin = margin

#         self.device = device

#     def forward(self, cell_feture,latent_st):
#         cell_feture = F.normalize(cell_feture, dim=1)
#         st_features = F.normalize(latent_st, dim=1)
#         matrix = torch.matmul(cell_feture, st_features.t())
#         # print(cell_feture.shape, st_features.shape, matrix.shape)

#         loss_rows = F.cross_entropy(matrix, torch.arange(matrix.size(0)).to(self.device))
#         loss_columns = F.cross_entropy(matrix.t(), torch.arange(matrix.size(0)).to(self.device))
#         loss = (loss_rows + loss_columns)/2
#         return loss
    
class Voxel_loss(nn.Module):
    def __init__(self, margin):
        super(Voxel_loss, self).__init__()
        self.margin = margin

    def forward(self, cell_feature, latent_st):
        device = cell_feature.device  # 自动获取设备


        cell_feature = F.normalize(cell_feature, dim=1)
        latent_st = F.normalize(latent_st, dim=1)
        matrix = torch.matmul(cell_feature, latent_st.t())

        # 计算交叉熵损失
        loss_rows = F.cross_entropy(matrix, torch.arange(matrix.size(0), device=device))
        loss_columns = F.cross_entropy(matrix.t(), torch.arange(matrix.size(0), device=device))
        loss = (loss_rows + loss_columns) / 2

        return loss
