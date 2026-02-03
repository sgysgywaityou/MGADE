import torch
import torch.nn as nn


class AdaptiveLossFusion(nn.Module):
    def __init__(self, config):
        super(AdaptiveLossFusion, self).__init__()
        self.config = config

        # 可学习的融合参数
        self.eta = nn.Parameter(torch.tensor(config.eta))
        self.lambda_ = nn.Parameter(torch.tensor(config.lambda_))
        self.sigma = nn.Parameter(torch.tensor(config.sigma))
        self.nu = nn.Parameter(torch.tensor(config.nu))

    def forward(self, main_loss, gabc_loss=None, mmd_loss=None, mlmmd_loss=None):
        """
        自适应损失融合
        """
        total_loss = 0

        # 文本分类损失融合
        if gabc_loss is not None:
            # 归一化参数
            eta_norm = torch.sigmoid(self.eta)
            lambda_norm = torch.sigmoid(self.lambda_)
            sum_params = eta_norm + lambda_norm
            eta = eta_norm / sum_params
            lambda_val = lambda_norm / sum_params

            text_loss = eta * main_loss + lambda_val * gabc_loss
            total_loss += text_loss

        # 域对齐损失融合
        if mmd_loss is not None and mlmmd_loss is not None:
            # 归一化参数
            sigma_norm = torch.sigmoid(self.sigma)
            nu_norm = torch.sigmoid(self.nu)
            sum_params = sigma_norm + nu_norm
            sigma = sigma_norm / sum_params
            nu = nu_norm / sum_params

            domain_loss = sigma * mmd_loss + nu * mlmmd_loss
            total_loss += self.config.psi * domain_loss

        return total_loss

    def get_fusion_params(self):
        """获取当前的融合参数值"""
        with torch.no_grad():
            eta_norm = torch.sigmoid(self.eta)
            lambda_norm = torch.sigmoid(self.lambda_)
            sigma_norm = torch.sigmoid(self.sigma)
            nu_norm = torch.sigmoid(self.nu)

            sum_text = eta_norm + lambda_norm
            sum_domain = sigma_norm + nu_norm

            return {
                'eta': (eta_norm / sum_text).item(),
                'lambda': (lambda_norm / sum_text).item(),
                'sigma': (sigma_norm / sum_domain).item(),
                'nu': (nu_norm / sum_domain).item()
            }