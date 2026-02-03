import torch
import torch.nn as nn
import torch.nn.functional as F


class MainClassifier(nn.Module):
    def __init__(self, config):
        super(MainClassifier, self).__init__()
        self.config = config

        self.fc_layers = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )

        # 偏置项
        self.bias = nn.Parameter(torch.zeros(config.num_classes))

    def forward(self, x):
        logits = self.fc_layers(x) + self.bias
        return logits

    def compute_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)


class GABC(nn.Module):
    def __init__(self, config):
        super(GABC, self).__init__()
        self.config = config
        self.main_classifier = MainClassifier(config)

    def compute_perturbed_bias(self, main_logits, main_loss, labels):
        """
        计算对抗性扰动偏置
        """
        # 计算主分类器损失对偏置的梯度
        if self.main_classifier.bias.grad is not None:
            self.main_classifier.bias.grad.zero_()

        main_loss.backward(retain_graph=True)
        bias_grad = self.main_classifier.bias.grad.detach()

        # 归一化梯度
        grad_norm = torch.norm(bias_grad, p=2)
        if grad_norm > 0:
            normalized_grad = bias_grad / grad_norm
        else:
            normalized_grad = bias_grad

        # 生成扰动偏置
        perturbed_bias = self.config.perturbation_strength * normalized_grad
        return perturbed_bias

    def forward(self, x, main_logits, main_loss, labels):
        """
        前向传播，生成两个扰动版本的预测
        """
        # 获取扰动偏置
        perturbed_bias = self.compute_perturbed_bias(main_logits, main_loss, labels)

        # 使用原始偏置+扰动偏置
        logits_positive = self.main_classifier.fc_layers(x) + self.main_classifier.bias + perturbed_bias

        # 使用原始偏置-扰动偏置
        logits_negative = self.main_classifier.fc_layers(x) + self.main_classifier.bias - perturbed_bias

        return logits_positive, logits_negative

    def compute_loss(self, x, main_logits, main_loss, labels):
        logits_pos, logits_neg = self.forward(x, main_logits, main_loss, labels)

        loss_pos = F.cross_entropy(logits_pos, labels)
        loss_neg = F.cross_entropy(logits_neg, labels)

        return loss_pos + loss_neg