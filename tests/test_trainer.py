from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from visionsearch_fg.engine import train_one_epoch, validate


class TinyImageDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict:
        return {
            "image": torch.full((3, 16, 16), fill_value=float(index)),
            "label": torch.tensor(index % 2, dtype=torch.long),
        }


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, 4), nn.ReLU())
        self.classifier = nn.Linear(4, 2)

    def forward(self, images: torch.Tensor):
        embedding = self.features(images)
        logits = self.classifier(embedding)
        return type("ModelOutput", (), {"logits": logits, "embedding": embedding})()


def test_train_one_epoch_and_validate_run() -> None:
    dataloader = DataLoader(TinyImageDataset(), batch_size=2)
    model = TinyClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    train_stats = train_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_batches=1,
    )
    eval_stats = validate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        max_batches=1,
    )

    assert train_stats.num_samples == 2
    assert eval_stats.num_samples == 2
    assert train_stats.loss > 0
    assert eval_stats.loss > 0
