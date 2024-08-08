import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin=0.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def _pairwise_distances(self, embeddings, squared=False):
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = (
            square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
        )
        distances = torch.clamp(distances, min=0.0)

        if not squared:
            mask = (distances == 0.0).float()
            distances = distances + mask * 1e-16
            distances = torch.sqrt(distances)
            distances = distances * (1.0 - mask)

        return distances

    def _get_anchor_positive_triplet_mask(self, labels):
        # 创建一个对角线为True的索引矩阵
        indices_equal = torch.eye(labels.size(0), dtype=torch.bool).to(labels.device)

        # 创建一个索引矩阵，其中标签相等的元素为True
        labels_equal = labels.view(-1, 1) == labels.view(1, -1)

        labels_equal = labels_equal.to(labels.device)
        # 创建一个掩码矩阵，排除与锚样本相同的样本以及不是正例的样本
        mask = ~indices_equal & labels_equal

        return mask

    def _get_anchor_negative_triplet_mask(self, labels):
        labels_equal = labels.view(-1, 1) == labels.view(1, -1)
        mask = ~labels_equal
        return mask

    def _get_triplet_mask(self, labels):
        batch_size = labels.size(0)
        indices_equal = torch.eye(batch_size, dtype=torch.bool)
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.view(batch_size, 1, batch_size)
        i_not_equal_k = indices_not_equal.view(batch_size, batch_size, 1)
        j_not_equal_k = indices_not_equal.view(1, batch_size, batch_size)
        distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k
        labels_equal = labels.view(batch_size, 1) == labels.view(1, batch_size)
        i_equal_j = labels_equal.view(batch_size, 1, batch_size)
        i_equal_k = labels_equal.view(batch_size, batch_size, 1)
        valid_labels = i_equal_j & ~i_equal_k
        mask = distinct_indices & valid_labels
        return mask

    def forward(self, embeddings, labels):
        pairwise_dist = self._pairwise_distances(embeddings, squared=False)

        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels)
        mask_anchor_positive = mask_anchor_positive.float()
        anchor_positive_dist = pairwise_dist * mask_anchor_positive
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)

        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)
        mask_anchor_negative = mask_anchor_negative.float()
        max_anchor_negative_dist, _ = pairwise_dist.max(dim=1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
            1.0 - mask_anchor_negative
        )
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)

        triplet_loss = torch.clamp(
            hardest_positive_dist - hardest_negative_dist + self.margin, min=0.0
        )
        triplet_loss = triplet_loss.mean()

        return triplet_loss
