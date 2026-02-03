import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class MSCDTCDataset(Dataset):
    def __init__(self, texts, labels=None, domain_id=None, is_target=False):
        self.texts = texts
        self.labels = labels
        self.domain_id = domain_id
        self.is_target = is_target
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

        if not self.is_target and self.labels is not None:
            item['label'] = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.domain_id is not None:
            item['domain_id'] = torch.tensor(self.domain_id, dtype=torch.long)

        return item


def create_dataloaders(source_datasets, target_dataset, config):
    """
    source_datasets: 源域数据集列表，每个元素为(texts, labels, domain_id)
    target_dataset: 目标域数据集(texts, None, None)
    """
    train_loaders = []

    for i, (texts, labels, domain_id) in enumerate(source_datasets):
        dataset = MSCDTCDataset(texts, labels, domain_id, is_target=False)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        train_loaders.append(loader)

    target_dataset = MSCDTCDataset(target_dataset, None, None, is_target=True)
    target_loader = DataLoader(target_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loaders, target_loader