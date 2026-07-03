"""Homework 4：使用 Transformer 完成序列反转任务。

输入是一串离散 token，输出是逆序 token 并以 EOS 结尾。该任务演示
Encoder、Decoder、causal mask、teacher forcing 和自回归生成。
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "当前环境未安装 PyTorch；本脚本不会自动安装依赖。"
    ) from exc


BOS_TOKEN = 0
EOS_TOKEN = 1
FIRST_DATA_TOKEN = 2
VOCAB_SIZE = 24
SOURCE_LENGTH = 6



def parse_args() -> argparse.Namespace:
    """读取训练参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """固定随机状态，使作业结果可复现。"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ReverseSequenceDataset(Dataset):
    """提供源序列、Decoder 输入和监督目标。"""

    def __init__(self, sample_count: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.sources = torch.randint(
            FIRST_DATA_TOKEN,
            VOCAB_SIZE,
            (sample_count, SOURCE_LENGTH),
            generator=generator,
        )

        # Decoder 输入从 BOS 开始，之后接逆序源序列。
        reversed_sources = self.sources.flip(dims=(1,))
        bos_column = torch.full((sample_count, 1), BOS_TOKEN)
        eos_column = torch.full((sample_count, 1), EOS_TOKEN)
        self.decoder_inputs = torch.cat([bos_column, reversed_sources], dim=1)
        self.targets = torch.cat([reversed_sources, eos_column], dim=1)

    def __len__(self) -> int:
        """返回数据集大小。"""

        return self.sources.size(0)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """读取一条序列到序列训练样本。"""

        return (
            self.sources[index],
            self.decoder_inputs[index],
            self.targets[index],
        )


class ReverseTransformer(nn.Module):
    """用于固定长度序列反转的 Encoder-Decoder Transformer。"""

    def __init__(self, hidden_size: int = 48) -> None:
        super().__init__()
        self.source_embedding = nn.Embedding(VOCAB_SIZE, hidden_size)
        self.target_embedding = nn.Embedding(VOCAB_SIZE, hidden_size)
        self.source_positions = nn.Embedding(SOURCE_LENGTH, hidden_size)
        self.target_positions = nn.Embedding(SOURCE_LENGTH + 1, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=96,
            dropout=0.1,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, VOCAB_SIZE)

    def encode_source(self, source: torch.Tensor) -> torch.Tensor:
        """将源 token 与位置编码相加。"""

        batch_size, sequence_length = source.shape
        positions = torch.arange(sequence_length, device=source.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        return self.source_embedding(source) + self.source_positions(positions)

    def encode_target(self, target: torch.Tensor) -> torch.Tensor:
        """将目标 token 与位置编码相加。"""

        batch_size, sequence_length = target.shape
        positions = torch.arange(sequence_length, device=target.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        return self.target_embedding(target) + self.target_positions(positions)

    def forward(
        self,
        source: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """使用 teacher forcing 返回每个位置的 token logits。"""

        source_hidden = self.encode_source(source)
        target_hidden = self.encode_target(decoder_input)
        target_length = decoder_input.size(1)

        # True 表示该位置被遮挡，防止 Decoder 读取未来答案。
        causal_mask = torch.triu(
            torch.ones(
                target_length,
                target_length,
                dtype=torch.bool,
                device=source.device,
            ),
            diagonal=1,
        )
        hidden = self.transformer(
            source_hidden,
            target_hidden,
            tgt_mask=causal_mask,
        )
        return self.output_layer(hidden)

    @torch.no_grad()
    def generate(self, source: torch.Tensor) -> torch.Tensor:
        """对一个 batch 执行贪心自回归生成。"""

        self.eval()
        batch_size = source.size(0)
        generated = torch.full(
            (batch_size, 1),
            BOS_TOKEN,
            dtype=torch.long,
            device=source.device,
        )

        # 每次只追加最后一个位置预测的 token。
        for _ in range(SOURCE_LENGTH + 1):
            logits = self(source, generated)
            next_token = logits[:, -1].argmax(dim=1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

        # 返回时移除仅用于启动 Decoder 的 BOS。
        return generated[:, 1:]


@dataclass(frozen=True)
class SequenceMetrics:
    """保存 token 准确率与整句准确率。"""

    loss: float
    token_accuracy: float
    sequence_accuracy: float


def run_epoch(
    model: ReverseTransformer,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> SequenceMetrics:
    """执行一轮 teacher-forcing 训练或验证。"""

    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct_tokens = 0
    total_tokens = 0
    correct_sequences = 0
    total_sequences = 0

    with torch.set_grad_enabled(is_training):
        for source, decoder_input, targets in loader:
            source = source.to(device)
            decoder_input = decoder_input.to(device)
            targets = targets.to(device)

            if optimizer is not None:
                optimizer.zero_grad()

            logits = model(source, decoder_input)
            loss = loss_function(
                logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
            )

            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            predictions = logits.argmax(dim=-1)
            batch_size = source.size(0)
            total_loss += loss.item() * batch_size
            correct_tokens += (predictions == targets).sum().item()
            total_tokens += targets.numel()
            correct_sequences += (
                (predictions == targets).all(dim=1).sum().item()
            )
            total_sequences += batch_size

    return SequenceMetrics(
        loss=total_loss / total_sequences,
        token_accuracy=correct_tokens / total_tokens,
        sequence_accuracy=correct_sequences / total_sequences,
    )


def main() -> None:
    """训练 Transformer，并展示自回归预测样例。"""

    args = parse_args()
    set_random_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 验证集由独立随机序列构成，用于检查模型泛化。
    train_dataset = ReverseSequenceDataset(2200, seed=42)
    validation_dataset = ReverseSequenceDataset(400, seed=2026)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
    )

    model = ReverseTransformer().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    checkpoint_path = args.output_dir / "reverse_transformer.pt"
    # 使用负初值保证即使首轮整句准确率为零也会保存 checkpoint。
    best_sequence_accuracy = -1.0

    print(f"device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
        )

        # 整句准确率比 token 准确率更严格，适合作为保存依据。
        if validation_metrics.sequence_accuracy > best_sequence_accuracy:
            best_sequence_accuracy = validation_metrics.sequence_accuracy
            torch.save(model.state_dict(), checkpoint_path)

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics.loss:.4f} "
            f"val_token_acc={validation_metrics.token_accuracy:.3f} "
            f"val_sequence_acc={validation_metrics.sequence_accuracy:.3f}"
        )

    # weights_only 是新版安全选项，TypeError 回退兼容旧版教程环境。
    try:
        saved_state = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        saved_state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(saved_state)
    examples = validation_dataset.sources[:5].to(device)
    predictions = model.generate(examples).cpu()
    expected = validation_dataset.targets[:5]

    print("\nautoregressive examples:")
    for source, prediction, target in zip(
        examples.cpu(),
        predictions,
        expected,
    ):
        print(
            f"source={source.tolist()} "
            f"prediction={prediction.tolist()} "
            f"target={target.tolist()}"
        )

    print(f"checkpoint: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
