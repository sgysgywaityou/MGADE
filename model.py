import torch
import torch.nn as nn
from .feature_extractor import FeatureExtractor
from .classifiers import MainClassifier, GABC
from .domain_alignment import MMMD, MLMMD
from .loss_functions import AdaptiveLossFusion


class MGADE(nn.Module):
    def __init__(self, config):
        super(MGADE, self).__init__()
        self.config = config

        # 特征提取器
        self.feature_extractor = FeatureExtractor(config)

        # 标签嵌入（可学习）
        self.label_embeddings = nn.Parameter(
            torch.randn(config.num_classes, config.embedding_dim)
        )

        # 分类器
        self.main_classifier = MainClassifier(config)
        self.gabc = GABC(config)

        # 域对齐模块
        self.mmmd = MMMD(config)
        self.mlmmd = MLMMD(config)

        # 损失融合
        self.loss_fusion = AdaptiveLossFusion(config)

    def forward(self, source_inputs_list, source_labels_list, target_inputs, epoch=None):
        """
        前向传播

        Args:
            source_inputs_list: 源域输入列表，每个元素为字典{input_ids, attention_mask}
            source_labels_list: 源域标签列表
            target_inputs: 目标域输入
            epoch: 当前epoch（用于某些策略）
        """
        batch_size = source_inputs_list[0]['input_ids'].size(0)

        # 提取源域特征
        source_features_list = []
        source_enhanced_features_list = []

        for i, (source_inputs, source_labels) in enumerate(zip(source_inputs_list, source_labels_list)):
            # 提取基础特征
            features = self.feature_extractor(
                source_inputs['input_ids'],
                source_inputs['attention_mask']
            )

            # 源域特征增强
            enhanced_features = self.feature_extractor.enhance_source_features(
                features,
                self.label_embeddings,
                source_labels
            )

            source_features_list.append(features)
            source_enhanced_features_list.append(enhanced_features)

        # 提取目标域特征
        target_features = self.feature_extractor(
            target_inputs['input_ids'],
            target_inputs['attention_mask']
        )

        # 目标域特征增强
        target_enhanced_features = self.feature_extractor.enhance_target_features(
            target_features,
            self.label_embeddings
        )

        # 分类
        losses = []
        text_losses = []

        for i, enhanced_features in enumerate(source_enhanced_features_list):
            source_labels = source_labels_list[i]

            # 主分类器
            main_logits = self.main_classifier(enhanced_features)
            main_loss = self.main_classifier.compute_loss(main_logits, source_labels)

            # GABC
            gabc_loss = self.gabc.compute_loss(
                enhanced_features, main_logits, main_loss, source_labels
            )

            # 文本分类损失融合
            text_loss = self.loss_fusion(main_loss, gabc_loss)
            text_losses.append(text_loss)

        # 平均文本分类损失
        avg_text_loss = torch.stack(text_losses).mean()

        # 域对齐损失
        # 获取目标域伪标签
        with torch.no_grad():
            target_logits = self.main_classifier(target_enhanced_features)
            target_pseudo_labels = torch.argmax(target_logits, dim=-1)

        mmd_loss = self.mmmd(source_features_list, target_features)
        mlmmd_loss = self.mlmmd(
            source_features_list, source_labels_list,
            target_features, target_pseudo_labels
        )

        # 总损失
        total_loss = self.loss_fusion(
            main_loss=avg_text_loss,
            mmd_loss=mmd_loss,
            mlmmd_loss=mlmmd_loss
        )

        # 返回结果
        return {
            'total_loss': total_loss,
            'text_loss': avg_text_loss,
            'mmd_loss': mmd_loss,
            'mlmmd_loss': mlmmd_loss,
            'target_logits': target_logits,
            'fusion_params': self.loss_fusion.get_fusion_params()
        }

    def predict(self, inputs):
        """预测目标域样本"""
        with torch.no_grad():
            # 提取特征
            features = self.feature_extractor(
                inputs['input_ids'],
                inputs['attention_mask']
            )

            # 增强特征
            enhanced_features = self.feature_extractor.enhance_target_features(
                features,
                self.label_embeddings
            )

            # 分类
            logits = self.main_classifier(enhanced_features)
            predictions = torch.argmax(logits, dim=-1)

        return predictions, logits