"""PyTorch 入门训练示例。

任务：构造一个二维二分类玩具数据集，并使用一个小型神经网络完成训练。

运行方式：
    python 03_pytorch_intro_training.py

说明：
    - CPU 即可运行。
    - 不下载数据。
    - 不安装依赖。
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "当前环境未安装 PyTorch。请先在已配置好的环境中运行本脚本。"
    ) from exc


def make_dataset(n_samples: int = 600, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)

    class0 = torch.randn(n_samples // 2, 2, generator=generator) * 0.7 + torch.tensor([-1.5, -1.0])
    class1 = torch.randn(n_samples // 2, 2, generator=generator) * 0.7 + torch.tensor([1.5, 1.0])

    x = torch.cat([class0, class1], dim=0)
    y = torch.cat(
        [
            torch.zeros(n_samples // 2, dtype=torch.long),
            torch.ones(n_samples // 2, dtype=torch.long),
        ],
        dim=0,
    )

    indices = torch.randperm(n_samples, generator=generator)
    return x[indices], y[indices]


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    x, y = make_dataset()
    train_x, test_x = x[:500], x[500:]
    train_y, test_y = y[:500], y[500:]

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=32,
        shuffle=True,
    )

    model = TinyClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        if epoch % 10 == 0:
            avg_loss = total_loss / len(train_loader.dataset)
            print(f"epoch={epoch:02d}, loss={avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(test_x.to(device))
        pred = logits.argmax(dim=1).cpu()
        acc = (pred == test_y).float().mean().item()

    print("test accuracy:", round(acc, 4))

    print("\n核心训练流程回顾：")
    print("1. forward: logits = model(x)")
    print("2. loss: loss_fn(logits, y)")
    print("3. zero_grad: optimizer.zero_grad()")
    print("4. backward: loss.backward()")
    print("5. update: optimizer.step()")


if __name__ == "__main__":
    train()
