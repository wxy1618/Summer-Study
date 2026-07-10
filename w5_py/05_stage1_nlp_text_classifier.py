# -*- coding: utf-8 -*-
"""Stage-1 NLP model-building pipeline.

本脚本对应第五周 fastbook NLP 实战。它用小型文本分类任务串起：
分词、词表、数值化、padding、DataLoader、模型训练、评估和结果记录。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    # 分类模型依赖 PyTorch；导入失败时脚本仍能说明环境状态。
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None

    class _MissingNN:
        Module = object

    nn = _MissingNN()
    F = None
    Dataset = object
    DataLoader = None


TOKEN_RE = re.compile(r"[a-zA-Z]+|[0-9]+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

# PAD 与 UNK 分开建模：PAD 表示批处理补齐，UNK 表示词表外词。
# 二者语义完全不同，混用会污染文本表示。

# 这个小数据集不是为了替代 IMDb，而是为了让实验在普通笔记本上几秒内跑通。
# 标签含义：1 表示“实验记录质量较好”，0 表示“实验记录质量不足”。
DATASET = [
    ("the experiment reports data split metrics and error cases", 1),
    ("the notebook records assumptions random seed and validation loss", 1),
    ("the model comparison includes baselines and reproducible settings", 1),
    ("training curves and qualitative examples are both preserved", 1),
    ("the report explains why the transformer objective is selected", 1),
    ("the pipeline checks tokenization vocabulary padding and accuracy", 1),
    ("results are summarized with loss accuracy and observed limitations", 1),
    ("the dependency parser trace is saved for later inspection", 1),
    ("the script only says success without any metric", 0),
    ("the result file misses data description and seed information", 0),
    ("the model is trained but no validation set is created", 0),
    ("the code changes many parameters without recording the reason", 0),
    ("the report copies definitions but does not analyze errors", 0),
    ("the experiment depends on hidden files and cannot be reproduced", 0),
    ("the training log contains only final accuracy and no context", 0),
    ("the conclusion ignores failed predictions and unstable outputs", 0),
]


def require_torch() -> None:
    # 课程要求不自动配置环境，因此依赖缺失时只提示，不执行安装命令。
    if torch is None:
        print("当前环境未安装 PyTorch；本脚本不会自动安装依赖。")
        raise SystemExit(0)


def tokenize(text: str) -> list[str]:
    # 这里选择透明的正则分词，便于在报告中逐词解释分类器输入。
    # 若迁移到中文或真实英文语料，应替换为更合适的分词/子词工具。
    return TOKEN_RE.findall(text.lower())


def split_dataset(
    examples: list[tuple[str, int]],
    valid_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Shuffle and split data while keeping the experiment reproducible."""

    # 使用局部 Random 对象，避免影响其他模块的随机状态。
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    # 验证集模拟未见数据，用于估计模型是否只记住训练样本。
    split_index = max(1, int(len(shuffled) * (1 - valid_ratio)))
    return shuffled[:split_index], shuffled[split_index:]


def build_vocab(
    examples: Iterable[tuple[str, int]],
    min_freq: int,
) -> tuple[dict[str, int], list[str]]:
    # 只用训练集构建词表，是监督学习中避免验证集泄漏的基本规范。
    counter: Counter[str] = Counter()
    for text, _ in examples:
        counter.update(tokenize(text))

    # 低频过滤由 min_freq 控制；小数据默认保留全部词以便观察每个 token。
    words = [
        word
        for word, count in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if count >= min_freq
    ]
    # id_to_token 支持从预测或调试结果反查词面，便于做错误分析。
    id_to_token = [PAD_TOKEN, UNK_TOKEN] + words
    token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
    return token_to_id, id_to_token


def encode_text(text: str, token_to_id: dict[str, int]) -> list[int]:
    # 验证集中的新词映射为 UNK，使模型处理开放词表输入时行为确定。
    unk_id = token_to_id[UNK_TOKEN]
    return [token_to_id.get(token, unk_id) for token in tokenize(text)]


class TextClassificationDataset(Dataset):
    """Dataset that stores encoded text and integer labels."""

    def __init__(
        self,
        examples: list[tuple[str, int]],
        token_to_id: dict[str, int],
    ) -> None:
        # 同时保存原始 text，是为了在验证阶段输出可读错误样例。
        self.rows = [
            (encode_text(text, token_to_id), label, text)
            for text, label in examples
        ]

    def __len__(self) -> int:
        # Dataset 长度用于 DataLoader 计算 batch 数，也用于训练日志解释数据规模。
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[list[int], int, str]:
        # 返回编码文本、标签和原文，兼顾训练输入和实验记录。
        return self.rows[index]


def collate_batch(
    batch: list[tuple[list[int], int, str]],
) -> tuple["torch.Tensor", "torch.Tensor", list[str]]:
    """Pad variable-length token ids to the longest sequence in a batch."""

    # 动态 padding 到 batch 内最长句子，减少不必要的 PAD 计算。
    max_length = max(len(token_ids) for token_ids, _, _ in batch)
    padded_rows = []
    labels = []
    texts = []
    for token_ids, label, text in batch:
        # PAD id 为 0，与 embedding 的 padding_idx 保持一致。
        padding = [0] * (max_length - len(token_ids))
        padded_rows.append(token_ids + padding)
        labels.append(label)
        texts.append(text)
    return (
        torch.tensor(padded_rows, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        texts,
    )


class MeanPoolingClassifier(nn.Module):
    """Embedding classifier with mask-aware mean pooling."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        # padding_idx=0 会让 PAD 向量不参与有效学习，降低 padding 噪声。
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 这个分类头故意保持简单，用于突出 NLP 基础流程而非复杂结构调参。
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        # mask 用来区分真实 token 和 PAD，保证平均池化只统计有效词。
        mask = input_ids.ne(0).float()
        embedded = self.embedding(input_ids)
        # mean pooling 是一个强基线：它忽略词序，但能快速验证词表和标签信号。
        summed = (embedded * mask.unsqueeze(-1)).sum(dim=1)
        lengths = mask.sum(dim=1).clamp_min(1.0)
        pooled = summed / lengths.unsqueeze(-1)
        return self.classifier(pooled)


def evaluate(
    model: MeanPoolingClassifier,
    loader: "DataLoader",
) -> tuple[float, float, list[dict[str, object]]]:
    """Return validation loss, accuracy and mistake examples."""

    # eval 模式关闭 dropout，保证验证指标不受训练随机性的影响。
    model.eval()
    total_loss = 0.0
    total_items = 0
    total_correct = 0
    mistakes: list[dict[str, object]] = []

    with torch.no_grad():
        for input_ids, labels, texts in loader:
            # 验证阶段不反向传播，只计算 logits、loss 和预测标签。
            logits = model(input_ids)
            loss = F.cross_entropy(logits, labels)
            predictions = logits.argmax(dim=1)

            # 累计总样本数，支持最后一个 batch 小于 batch_size 的情况。
            total_loss += float(loss.item()) * labels.size(0)
            total_items += labels.size(0)
            total_correct += int(predictions.eq(labels).sum().item())

            # 保存错误样例比单一准确率更有研究价值，可用于分析模型偏差。
            for text, gold, pred in zip(texts, labels.tolist(), predictions.tolist()):
                if gold != pred:
                    mistakes.append({"text": text, "gold": gold, "predicted": pred})

    return total_loss / total_items, total_correct / total_items, mistakes


def train(args: argparse.Namespace) -> dict[str, object]:
    # 固定随机种子，使数据划分、模型初始化和训练顺序可复现。
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 先划分再建词表，严格模拟真实监督学习的训练/验证边界。
    train_examples, valid_examples = split_dataset(DATASET, args.valid_ratio, args.seed)
    token_to_id, id_to_token = build_vocab(train_examples, min_freq=args.min_freq)
    # train_loader 打乱训练样本，valid_loader 保持顺序方便对照错误样例。
    train_loader = DataLoader(
        TextClassificationDataset(train_examples, token_to_id),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    valid_loader = DataLoader(
        TextClassificationDataset(valid_examples, token_to_id),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    # 平均池化分类器是阶段一实验的轻量基线，适合作为后续 RNN/Transformer 对照。
    model = MeanPoolingClassifier(
        vocab_size=len(id_to_token),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        # 每个 epoch 重新累计训练损失，记录模型拟合训练集的速度。
        train_loss = 0.0
        train_items = 0
        for input_ids, labels, _ in train_loader:
            # logits 形状为 [batch, 2]，对应两个实验记录质量标签。
            logits = model(input_ids)
            loss = F.cross_entropy(logits, labels)

            # 典型监督学习优化步骤；这里不加复杂 scheduler，便于看清主流程。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item()) * labels.size(0)
            train_items += labels.size(0)

        # 每个 epoch 后在验证集评估，避免只报告训练集表现。
        valid_loss, valid_accuracy, mistakes = evaluate(model, valid_loader)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss / train_items, 4),
                "valid_loss": round(valid_loss, 4),
                "valid_accuracy": round(valid_accuracy, 4),
            }
        )

    # 输出目录包含模型权重和 JSON 指标，满足“记录实验结果”的任务要求。
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "stage1_text_classifier.pt")
    result = {
        # 结果结构刻意保留数据规模、词表规模、曲线和错误样例。
        # 这些信息能支撑周报中“阶段一涵盖 NLP 基本步骤”的说明。
        "task": "stage1_nlp_text_classification",
        "train_size": len(train_examples),
        "valid_size": len(valid_examples),
        "vocab_size": len(id_to_token),
        "final_valid_accuracy": history[-1]["valid_accuracy"],
        "history": history,
        "mistakes": mistakes,
        "label_description": {
            "1": "实验记录较完整",
            "0": "实验记录不完整",
        },
    }
    (output_dir / "stage1_text_classifier_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    # 参数化训练配置，便于后续对 embedding 维度、学习率和验证比例做对比。
    parser = argparse.ArgumentParser(description="Stage-1 NLP classification pipeline")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--valid-ratio", type=float, default=0.25)
    parser.add_argument("--min-freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    # main 函数不写训练细节，便于 notebook 或其他脚本直接复用 train。
    require_torch()
    result = train(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"阶段一文本分类最终验证准确率：{result['final_valid_accuracy']}")


if __name__ == "__main__":
    main()
