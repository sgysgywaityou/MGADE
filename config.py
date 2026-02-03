import torch


class Config:
    def __init__(self):
        # 模型参数
        self.embedding_dim = 768  # BERT维度
        self.hidden_dim = 512
        self.num_classes = 2
        self.num_source_domains = 3

        # 训练参数
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5

        # GABC参数
        self.perturbation_strength = 0.1  # χ值
        self.grad_clip_value = 1.0

        # 自适应融合参数
        self.eta = 0.7  # 主分类器权重
        self.lambda_ = 0.3  # GABC权重
        self.sigma = 0.6  # MMMD权重
        self.nu = 0.4  # MLMMD权重
        self.psi = 1.0  # 域对齐损失权重

        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 路径设置
        self.bert_model = 'bert-base-uncased'


config = Config()