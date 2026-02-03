import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_model)
        self.embedding_dim = config.embedding_dim

        # 冻结BERT前几层
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-4:].parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用[CLS] token的特征
        features = outputs.last_hidden_state[:, 0, :]  # [batch_size, embedding_dim]
        return features

    def calculate_cosine_similarity(self, features, label_embeddings, label_idx=None):
        """
        计算特征与标签嵌入的余弦相似度

        Args:
            features: [batch_size, embedding_dim]
            label_embeddings: [num_classes, embedding_dim] 或 [embedding_dim] (对于源域)
            label_idx: 标签索引（源域使用）
        """
        if label_idx is not None:  # 源域：单个标签
            # 获取对应标签的嵌入
            label_emb = label_embeddings[label_idx].unsqueeze(0)  # [1, embedding_dim]
            label_emb = label_emb.expand(features.size(0), -1)  # [batch_size, embedding_dim]
        else:  # 目标域：所有标签
            label_emb = label_embeddings.unsqueeze(0)  # [1, num_classes, embedding_dim]
            label_emb = label_emb.expand(features.size(0), -1, -1)  # [batch_size, num_classes, embedding_dim]

        # 计算余弦相似度
        features_norm = F.normalize(features, p=2, dim=-1)
        if label_idx is not None:
            label_norm = F.normalize(label_emb, p=2, dim=-1)
            similarity = F.cosine_similarity(features_norm, label_norm, dim=-1)
        else:
            label_norm = F.normalize(label_emb, p=2, dim=-1)
            features_norm = features_norm.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            similarity = torch.bmm(features_norm, label_norm.transpose(1, 2)).squeeze(1)

        return similarity

    def enhance_source_features(self, features, label_embeddings, label_idx):
        """
        源域特征增强：基于余弦相似度
        """
        # 计算余弦相似度
        similarity = self.calculate_cosine_similarity(features, label_embeddings, label_idx)

        # 计算平均相似度
        mean_similarity = similarity.mean(dim=-1, keepdim=True)

        # 根据相似度缩放特征
        scale_factor = torch.where(
            similarity >= mean_similarity,
            1 + torch.abs(similarity),
            1 - torch.abs(similarity)
        )

        enhanced_features = features * scale_factor.unsqueeze(-1)
        return enhanced_features

    def enhance_target_features(self, features, label_embeddings):
        """
        目标域特征增强：基于全标签注意力
        """
        batch_size, embed_dim = features.shape
        num_classes = label_embeddings.shape[0]

        # 计算与所有标签的相似度
        similarity_matrix = self.calculate_cosine_similarity(features, label_embeddings)
        # similarity_matrix: [batch_size, num_classes]

        # 计算每个标签的平均相似度
        mean_similarity = similarity_matrix.mean(dim=0, keepdim=True)  # [1, num_classes]

        # 更新相似度矩阵
        scale_matrix = torch.where(
            similarity_matrix >= mean_similarity,
            1 + torch.abs(similarity_matrix),
            torch.ones_like(similarity_matrix)
        )

        # 逐行相乘
        row_product = torch.prod(scale_matrix, dim=-1, keepdim=True)  # [batch_size, 1]

        # 增强特征
        enhanced_features = features * row_product.expand(-1, embed_dim)
        return enhanced_features