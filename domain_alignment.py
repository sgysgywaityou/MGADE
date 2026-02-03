import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(x, y, sigma=1.0):
    """
    高斯核函数
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1)  # [x_size, 1, dim]
    y = y.unsqueeze(0)  # [1, y_size, dim]

    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)

    kernel_output = torch.exp(-torch.sum((tiled_x - tiled_y) ** 2, dim=2) / (2 * sigma ** 2))
    return kernel_output


class MMMD(nn.Module):
    def __init__(self, config):
        super(MMMD, self).__init__()
        self.config = config

    def forward(self, source_features_list, target_features):
        """
        计算多源均值MMD损失

        Args:
            source_features_list: 源域特征列表，每个元素为[batch_size, feature_dim]
            target_features: 目标域特征，[batch_size, feature_dim]
        """
        num_sources = len(source_features_list)
        total_mmd_loss = 0

        for source_features in source_features_list:
            # 计算单个源域与目标域的MMD
            source_kernel = gaussian_kernel(source_features, source_features)
            target_kernel = gaussian_kernel(target_features, target_features)
            cross_kernel = gaussian_kernel(source_features, target_features)

            mmd_loss = torch.mean(source_kernel) + torch.mean(target_kernel) - 2 * torch.mean(cross_kernel)
            total_mmd_loss += mmd_loss

        # 均值MMD
        mean_mmd_loss = total_mmd_loss / num_sources

        # 添加类别中心余弦相似度惩罚项
        cosine_penalty = self.compute_cosine_penalty(source_features_list, target_features)

        return mean_mmd_loss - cosine_penalty

    def compute_cosine_penalty(self, source_features_list, target_features):
        """
        计算类别中心余弦相似度惩罚项
        """
        num_classes = self.config.num_classes
        num_sources = len(source_features_list)

        total_cosine_sim = 0

        # 注意：这里简化了实现，实际需要根据样本标签计算类别中心
        # 这里假设所有样本属于同一类别进行计算
        for c in range(num_classes):
            for k in range(num_sources):
                # 计算源域类别c的中心
                source_center = source_features_list[k].mean(dim=0, keepdim=True)
                # 计算目标域类别c的中心
                target_center = target_features.mean(dim=0, keepdim=True)

                # 计算余弦相似度
                cos_sim = F.cosine_similarity(source_center, target_center, dim=-1)
                total_cosine_sim += cos_sim

        return total_cosine_sim / (num_classes * num_sources)


class MLMMD(nn.Module):
    def __init__(self, config):
        super(MLMMD, self).__init__()
        self.config = config

    def forward(self, source_features_list, source_labels_list, target_features, target_pseudo_labels):
        """
        计算多源均值LMMD损失

        Args:
            source_features_list: 源域特征列表
            source_labels_list: 源域标签列表
            target_features: 目标域特征
            target_pseudo_labels: 目标域伪标签（模型预测）
        """
        num_sources = len(source_features_list)
        num_classes = self.config.num_classes
        total_lmmd_loss = 0

        for k in range(num_sources):
            source_features = source_features_list[k]
            source_labels = source_labels_list[k]

            source_weights = self.compute_sample_weights(source_labels, num_classes)
            target_weights = self.compute_sample_weights(target_pseudo_labels, num_classes)

            # 计算加权LMMD
            lmmd_loss = 0
            for c in range(num_classes):
                source_mask = (source_labels == c).float()
                target_mask = (target_pseudo_labels == c).float()

                if source_mask.sum() > 0 and target_mask.sum() > 0:
                    source_features_c = source_features[source_mask.bool()]
                    target_features_c = target_features[target_mask.bool()]

                    source_weights_c = source_weights[source_mask.bool(), c]
                    target_weights_c = target_weights[target_mask.bool(), c]

                    # 计算类别c的LMMD
                    kernel_ss = gaussian_kernel(source_features_c, source_features_c)
                    kernel_tt = gaussian_kernel(target_features_c, target_features_c)
                    kernel_st = gaussian_kernel(source_features_c, target_features_c)

                    weighted_kernel_ss = torch.matmul(source_weights_c.unsqueeze(1),
                                                      source_weights_c.unsqueeze(0)) * kernel_ss
                    weighted_kernel_tt = torch.matmul(target_weights_c.unsqueeze(1),
                                                      target_weights_c.unsqueeze(0)) * kernel_tt
                    weighted_kernel_st = torch.matmul(source_weights_c.unsqueeze(1),
                                                      target_weights_c.unsqueeze(0)) * kernel_st

                    lmmd_c = weighted_kernel_ss.sum() + weighted_kernel_tt.sum() - 2 * weighted_kernel_st.sum()
                    lmmd_loss += lmmd_c

            total_lmmd_loss += lmmd_loss / num_classes

        # 均值LMMD
        mean_lmmd_loss = total_lmmd_loss / num_sources

        # 添加余弦相似度惩罚项（与MMMD共享）
        cosine_penalty = MMMD.compute_cosine_penalty(self, source_features_list, target_features)

        return mean_lmmd_loss - cosine_penalty

    def compute_sample_weights(self, labels, num_classes):
        """
        计算样本权重
        """
        batch_size = labels.size(0)
        if labels.dim() == 1:  # 硬标签
            weights = torch.zeros(batch_size, num_classes, device=labels.device)
            weights.scatter_(1, labels.unsqueeze(1), 1)
        else:  # 软标签/伪标签
            weights = labels

        # 归一化
        class_counts = weights.sum(dim=0, keepdim=True)
        class_counts = torch.where(class_counts > 0, class_counts, torch.ones_like(class_counts))
        weights = weights / class_counts

        return weights