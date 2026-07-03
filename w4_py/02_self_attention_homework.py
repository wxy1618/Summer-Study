"""
Homework 2：使用 Self-attention 完成合成序列分类。

标签由两个特殊 token 的先后顺序决定，因此模型必须同时利用 token 内容和
位置信息。脚本不下载数据，CPU 即可完成训练。

本模块包含：
- 合成数据生成器（随机序列 + 两个标记符）
- 基于多头自注意力的分类器
- 训练与验证流程
- 命令行参数解析

"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

# 尝试导入 PyTorch，若未安装则给出友好提示
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "当前环境未安装 PyTorch；本脚本不会自动安装依赖。"
    ) from exc


# ---------- 全局常量 ----------
PAD_TOKEN = 0          # padding 使用的 token ID
FIRST_MARKER = 1       # 第一个特殊标记（标签决定其位置）
SECOND_MARKER = 2      # 第二个特殊标记
VOCAB_SIZE = 32        # 词汇表大小（包括上述特殊 token 和随机 token）
MAX_LENGTH = 14        # 所有序列统一填充后的最大长度


# ---------- 辅助函数 ----------
def parse_args() -> argparse.Namespace:
    """
    读取训练轮数和输出目录。

    使用 argparse 解析命令行参数，允许用户自定义训练轮数、批大小和输出目录。

    返回：
        argparse.Namespace: 包含 epochs, batch_size, output_dir 三个属性
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8,
                        help="训练轮数（默认：8）")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批大小（默认：64）")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="保存模型检查点的目录（默认：outputs）")
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """
    固定 Python 与 PyTorch 随机状态，确保实验可复现。

    参数：
        seed (int): 随机种子
    """
    random.seed(seed)
    torch.manual_seed(seed)
    # 如果使用 CUDA，也固定 GPU 随机状态
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sample(
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    创建一个长度可变的序列及其二分类标签。

    生成过程：
        1. 随机选择序列长度（6 到 MAX_LENGTH 之间）
        2. 随机生成该长度的 token（ID 范围 3 到 VOCAB_SIZE-1）
        3. 随机选择两个互不相同的位置，分别放入 FIRST_MARKER 和 SECOND_MARKER
        4. 标签 = 1 如果 FIRST_MARKER 在 SECOND_MARKER 之前，否则 0
        5. 将序列右侧填充 PAD_TOKEN 至 MAX_LENGTH

    参数：
        generator (torch.Generator): PyTorch 随机数生成器，用于可复现

    返回：
        tuple[torch.Tensor, torch.Tensor]:
            - 填充后的序列张量，形状 (MAX_LENGTH,)
            - 标签张量，标量 (0 或 1)
    """
    # 随机生成长度（6 ~ MAX_LENGTH，包含两端）
    length = int(torch.randint(6, MAX_LENGTH + 1, (1,), generator=generator))
    # 生成普通 token（ID 从 3 开始，避免与特殊标记冲突）
    sequence = torch.randint(
        3,
        VOCAB_SIZE,
        (length,),
        generator=generator,
    )

    # 从所有位置中随机选出两个不同位置，用于放置标记
    positions = torch.randperm(length, generator=generator)[:2]
    first_position = int(positions[0])
    second_position = int(positions[1])
    # 放置两个特殊标记
    sequence[first_position] = FIRST_MARKER
    sequence[second_position] = SECOND_MARKER
    # 标签：第一个标记是否在第二个标记前面
    label = int(first_position < second_position)

    # 右侧填充 PAD_TOKEN 到固定长度，方便批次处理
    padded = torch.full((MAX_LENGTH,), PAD_TOKEN, dtype=torch.long)
    padded[:length] = sequence
    return padded, torch.tensor(label, dtype=torch.long)


def make_dataset(
    sample_count: int,
    seed: int,
) -> TensorDataset:
    """
    创建可复现的合成序列数据集。

    使用指定种子生成 sample_count 个样本，并打包成 PyTorch 的 TensorDataset。

    参数：
        sample_count (int): 样本数量
        seed (int): 随机种子（用于生成器）

    返回：
        TensorDataset: 包含所有序列和标签的数据集
    """
    generator = torch.Generator().manual_seed(seed)
    # 生成所有样本
    samples = [make_sample(generator) for _ in range(sample_count)]
    # 分别堆叠序列和标签
    sequences = torch.stack([sample[0] for sample in samples])
    labels = torch.stack([sample[1] for sample in samples])
    return TensorDataset(sequences, labels)


# ---------- 模型定义 ----------
class SelfAttentionClassifier(nn.Module):
    """
    使用多头注意力编码序列并执行二分类。

    模型结构：
        1. Token 嵌入 + 位置嵌入（可学习）
        2. 多头自注意力（batch_first=True）
        3. 残差连接 + LayerNorm
        4. 平均池化（忽略 padding）
        5. 两层 MLP 分类器输出二分类 logits

    属性：
        token_embedding (nn.Embedding): 词嵌入层，padding_idx 使得 PAD_TOKEN 嵌入为零
        position_embedding (nn.Embedding): 位置嵌入，可学习
        attention (nn.MultiheadAttention): 多头自注意力
        normalization (nn.LayerNorm): 层归一化
        classifier (nn.Sequential): 分类头
    """

    def __init__(self, hidden_size: int = 48, head_count: int = 4) -> None:
        """
        初始化分类器。

        参数：
            hidden_size (int): 嵌入维度和注意力隐藏维度（默认 48）
            head_count (int): 多头注意力的头数（默认 4）
        """
        super().__init__()

        # 词嵌入层，padding_idx 使填充 token 的嵌入向量恒为零向量
        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            hidden_size,
            padding_idx=PAD_TOKEN,
        )
        # 位置嵌入，每个位置对应一个可学习的嵌入向量
        self.position_embedding = nn.Embedding(MAX_LENGTH, hidden_size)
        # 多头自注意力，batch_first=True 使得输入形状为 (batch, seq_len, hidden)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            head_count,
            dropout=0.1,
            batch_first=True,
        )
        # LayerNorm 用于残差连接后归一化
        self.normalization = nn.LayerNorm(hidden_size)
        # 分类头：两层线性 + ReLU + Dropout，输出二分类 logits
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回每个序列的二分类 logits。

        参数：
            tokens (torch.Tensor): 形状 (batch_size, MAX_LENGTH) 的 token 序列

        返回：
            torch.Tensor: 形状 (batch_size, 2) 的 logits
        """
        batch_size, sequence_length = tokens.shape
        # 生成位置索引 (0, 1, ..., sequence_length-1) 并扩展到 batch 维度
        positions = torch.arange(sequence_length, device=tokens.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)

        # 获取 token 嵌入并加上位置嵌入，使注意力能够感知位置顺序
        hidden = self.token_embedding(tokens)          # (B, L, H)
        hidden = hidden + self.position_embedding(positions)  # (B, L, H)

        # 创建 padding 掩码，True 表示需要被忽略的位置
        padding_mask = tokens.eq(PAD_TOKEN)

        # 多头自注意力：使用相同输入作为 Q, K, V（自注意力）
        # key_padding_mask 使得 padding 位置不被关注
        attended, _ = self.attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=padding_mask,
            need_weights=False,          # 不返回注意力权重，节省内存
        )
        # 残差连接 + LayerNorm
        hidden = self.normalization(hidden + attended)

        # 池化：对真实 token（非 padding）取平均
        # 构建有效掩码，用于加权平均
        valid_mask = (~padding_mask).unsqueeze(-1)      # (B, L, 1)
        # 求和有效位置的特征
        pooled = (hidden * valid_mask).sum(dim=1)       # (B, H)
        # 除以有效数量（至少为 1，避免除零）
        pooled = pooled / valid_mask.sum(dim=1).clamp_min(1)  # (B, H)

        # 通过分类头得到 logits
        return self.classifier(pooled)                  # (B, 2)


# ---------- 训练/验证循环 ----------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """
    运行一轮训练或验证，并返回平均损失与准确率。

    若 optimizer 不为 None，则进入训练模式并执行反向传播；
    否则进入验证模式，不更新参数。

    参数：
        model (nn.Module): 模型
        loader (DataLoader): 数据加载器
        loss_function (nn.Module): 损失函数（如 CrossEntropyLoss）
        device (torch.device): 设备（CPU/GPU）
        optimizer (torch.optim.Optimizer | None): 若提供则执行训练

    返回：
        tuple[float, float]: (平均损失, 准确率)
    """
    is_training = optimizer is not None
    model.train(is_training)   # 根据训练/验证切换 dropout 等层的行为

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    # 在验证阶段禁用梯度计算，节省内存和加速
    with torch.set_grad_enabled(is_training):
        for tokens, labels in loader:
            # 将数据移至指定设备
            tokens = tokens.to(device)
            labels = labels.to(device)

            # 如果是训练，梯度清零
            if optimizer is not None:
                optimizer.zero_grad()

            # 前向传播
            logits = model(tokens)
            loss = loss_function(logits, labels)

            # 如果是训练，反向传播并更新参数
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            # 统计损失和准确率
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct_count += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    # 返回平均损失和准确率
    return total_loss / sample_count, correct_count / sample_count


# ---------- 主程序 ----------
def main() -> None:
    """
    训练模型，验证效果并保存最佳参数。

    流程：
        1. 解析命令行参数
        2. 固定随机种子
        3. 创建设备（CUDA 优先）
        4. 创建训练集和验证集（使用不同种子）
        5. 初始化模型、损失函数、优化器
        6. 循环训练，每轮验证并保存最佳模型
        7. 输出最佳验证准确率及检查点路径
    """
    args = parse_args()
    set_random_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建输出目录（如果不存在）
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 训练集和验证集使用不同种子，避免样本完全重复（增加泛化评估可信度）
    train_dataset = make_dataset(sample_count=1800, seed=42)
    validation_dataset = make_dataset(sample_count=400, seed=2026)

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,          # 训练集打乱
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,         # 验证集不打乱
    )

    # 初始化模型并移至设备
    model = SelfAttentionClassifier().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    checkpoint_path = args.output_dir / "self_attention_classifier.pt"
    best_accuracy = 0.0

    print(f"device: {device}")
    for epoch in range(1, args.epochs + 1):
        # 训练一轮
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
        )
        # 验证一轮
        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            optimizer=None,    # 验证时不更新参数
        )

        # 保存验证准确率最高的模型（避免最后一轮过拟合导致的退化）
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), checkpoint_path)

        # 打印当前轮次性能指标
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.3f} "
            f"val_loss={validation_loss:.4f} "
            f"val_acc={validation_accuracy:.3f}"
        )

    # 输出最终最佳结果
    print(f"best validation accuracy: {best_accuracy:.3f}")
    print(f"checkpoint: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()