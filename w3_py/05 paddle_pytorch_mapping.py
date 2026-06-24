"""PyTorch 与 PaddlePaddle 基础写法对照。

本脚本优先运行 PyTorch 示例；如果当前环境安装了 PaddlePaddle，则同时运行 Paddle 示例。
不会安装任何软件包。
"""

from __future__ import annotations


def run_pytorch_demo() -> None:
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError:
        print("PyTorch not installed, skip PyTorch demo.")
        return

    class TorchNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    torch.manual_seed(42)
    x = torch.randn(16, 2)
    y = (x[:, 0] + x[:, 1] > 0).long()

    model = TorchNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    logits = model(x)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("[PyTorch]")
    print("logits shape:", tuple(logits.shape))
    print("loss:", round(loss.item(), 4))
    print("clear gradients API: optimizer.zero_grad()")


def run_paddle_demo() -> None:
    try:
        import paddle
        import paddle.nn as nn
    except ModuleNotFoundError:
        print("\nPaddlePaddle not installed, skip Paddle demo.")
        return

    class PaddleNet(nn.Layer):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
            )

        def forward(self, x: "paddle.Tensor") -> "paddle.Tensor":
            return self.net(x)

    paddle.seed(42)
    x = paddle.randn([16, 2])
    y = paddle.cast(x[:, 0] + x[:, 1] > 0, dtype="int64")

    model = PaddleNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.01)

    logits = model(x)
    loss = loss_fn(logits, y)
    optimizer.clear_grad()
    loss.backward()
    optimizer.step()

    print("\n[PaddlePaddle]")
    print("logits shape:", tuple(logits.shape))
    print("loss:", round(float(loss.numpy()), 4))
    print("clear gradients API: optimizer.clear_grad()")


def print_mapping_table() -> None:
    rows = [
        ("Tensor", "torch.Tensor", "paddle.Tensor"),
        ("Model base class", "torch.nn.Module", "paddle.nn.Layer"),
        ("Linear layer", "torch.nn.Linear", "paddle.nn.Linear"),
        ("Sequential", "torch.nn.Sequential", "paddle.nn.Sequential"),
        ("Cross entropy", "torch.nn.CrossEntropyLoss", "paddle.nn.CrossEntropyLoss"),
        ("Adam optimizer", "torch.optim.Adam", "paddle.optimizer.Adam"),
        ("Backward", "loss.backward()", "loss.backward()"),
        ("Clear grad", "optimizer.zero_grad()", "optimizer.clear_grad()"),
    ]

    print("\nPyTorch vs PaddlePaddle mapping")
    print("-" * 86)
    print(f"{'Concept':<18} | {'PyTorch':<30} | {'PaddlePaddle':<30}")
    print("-" * 86)
    for concept, torch_api, paddle_api in rows:
        print(f"{concept:<18} | {torch_api:<30} | {paddle_api:<30}")


if __name__ == "__main__":
    print_mapping_table()
    run_pytorch_demo()
    run_paddle_demo()
