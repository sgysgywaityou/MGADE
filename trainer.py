import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device

        # 优化器
        self.optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 学习率调度器
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)

        # 将模型移到设备
        self.model.to(self.device)

    def train_epoch(self, source_loaders, target_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        # 创建迭代器
        source_iters = [iter(loader) for loader in source_loaders]

        # 获取最小批次数量
        min_batches = min(len(loader) for loader in source_loaders)

        pbar = tqdm(range(min_batches), desc="Training")
        for batch_idx in pbar:
            # 获取源域批次数据
            source_batches = []
            source_labels_batches = []

            for i, source_iter in enumerate(source_iters):
                try:
                    batch = next(source_iter)
                except StopIteration:
                    # 重新初始化迭代器
                    source_iters[i] = iter(source_loaders[i])
                    batch = next(source_iters[i])

                # 移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                source_batches.append(batch)
                source_labels_batches.append(batch['label'])

            # 获取目标域批次数据
            try:
                target_batch = next(target_iter)
            except (NameError, StopIteration):
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            target_batch = {k: v.to(self.device) for k, v in target_batch.items()}

            # 前向传播
            outputs = self.model(source_batches, source_labels_batches, target_batch)
            loss = outputs['total_loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_value)

            self.optimizer.step()

            # 更新统计信息
            total_loss += loss.item()
            num_batches += 1

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'text': f"{outputs['text_loss'].item():.4f}",
                'mmd': f"{outputs['mmd_loss'].item():.4f}",
                'mlmmd': f"{outputs['mlmmd_loss'].item():.4f}"
            })

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss, outputs['fusion_params']

    def validate(self, target_loader, target_labels=None):
        """验证模型"""
        self.model.eval()
        all_predictions = []
        all_logits = []

        with torch.no_grad():
            for batch in tqdm(target_loader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                predictions, logits = self.model.predict(batch)
                all_predictions.extend(predictions.cpu().numpy())
                all_logits.append(logits.cpu())

        all_predictions = np.array(all_predictions)
        all_logits = torch.cat(all_logits, dim=0)

        if target_labels is not None:
            # 计算准确率
            accuracy = (all_predictions == target_labels).mean()
            return accuracy, all_predictions, all_logits
        else:
            return None, all_predictions, all_logits

    def train(self, source_loaders, target_loader, val_loader=None, val_labels=None):
        """完整训练过程"""
        best_accuracy = 0
        best_model_state = None

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # 训练
            train_loss, fusion_params = self.train_epoch(source_loaders, target_loader)
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Fusion Params: {fusion_params}")

            # 验证
            if val_loader is not None:
                accuracy, _, _ = self.validate(val_loader, val_labels)
                print(f"Validation Accuracy: {accuracy:.4f}")

                # 保存最佳模型
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_state = self.model.state_dict().copy()
                    print(f"New best model! Accuracy: {best_accuracy:.4f}")

            # 更新学习率
            self.scheduler.step()

        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return best_accuracy