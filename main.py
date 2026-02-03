import argparse
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from config import config
from data_loader import create_dataloaders
from model import MGADE
from trainer import Trainer


def load_data():
    """
    加载数据
    这里需要根据实际数据集实现
    """
    # 示例数据结构
    # source_datasets: [(texts1, labels1, 0), (texts2, labels2, 1), ...]
    # target_dataset: (texts_target, None, None)

    # 这里返回示例数据
    num_samples = 1000
    num_source_domains = 3

    source_datasets = []
    for i in range(num_source_domains):
        texts = [f"Sample {j} from source {i}" for j in range(num_samples)]
        labels = np.random.randint(0, config.num_classes, num_samples)
        source_datasets.append((texts, labels, i))

    target_texts = [f"Sample {j} from target" for j in range(num_samples)]

    return source_datasets, target_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model_path', type=str, default='best_model.pth')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载数据
    print("Loading data...")
    source_datasets, target_texts = load_data()

    # 划分训练集和验证集
    from sklearn.model_selection import train_test_split

    # 划分目标域数据为训练和验证
    target_train_texts, target_val_texts = train_test_split(
        target_texts, test_size=0.2, random_state=42
    )
    target_val_labels = np.random.randint(0, config.num_classes, len(target_val_texts))

    # 创建数据加载器
    train_loaders, target_train_loader = create_dataloaders(
        source_datasets, target_train_texts, config
    )

    _, target_val_loader = create_dataloaders(
        source_datasets, target_val_texts, config
    )

    # 创建模型
    print("Creating model...")
    model = MGADE(config)

    if args.mode == 'train':
        # 训练模型
        print("Training model...")
        trainer = Trainer(model, config)

        best_accuracy = trainer.train(
            train_loaders, target_train_loader,
            target_val_loader, target_val_labels
        )

        print(f"\nTraining completed. Best validation accuracy: {best_accuracy:.4f}")

        # 保存模型
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")

        # 保存配置
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
        with open('config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

    else:  # test mode
        # 加载模型
        print(f"Loading model from {args.model_path}...")
        model.load_state_dict(torch.load(args.model_path, map_location=config.device))
        model.to(config.device)

        # 测试
        trainer = Trainer(model, config)
        accuracy, predictions, logits = trainer.validate(target_val_loader, target_val_labels)

        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Logits shape: {logits.shape}")


if __name__ == "__main__":
    main()