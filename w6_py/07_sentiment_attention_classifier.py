# -*- coding: utf-8 -*-
"""Attention-based sentiment classifier with compact tuning.

本脚本对应第六周情感分析模型实战。它使用注意力池化从句子中提取关键信息，
并记录学习率、隐藏维度、验证准确率和错误样例。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from itertools import product
from pathlib import Path

try:
    # PyTorch 用于训练注意力分类器；依赖缺失时不自动安装。
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None

    class _MissingNN:
        Module = object

    nn = _MissingNN()
    F = None


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
PAD = "<pad>"
UNK = "<unk>"


# 标签 1 表示正向评价，0 表示负向评价；样本围绕实验和模型文本构造。
DATASET = [
    ("the model gives clear and reliable answers", 1),
    ("attention highlights useful evidence words", 1),
    ("the experiment records stable validation results", 1),
    ("the generated sentence is fluent and helpful", 1),
    ("the relation extractor finds correct facts", 1),
    ("the model ignores evidence and fails badly", 0),
    ("the output repeats noisy irrelevant words", 0),
    ("the experiment lacks metrics and is unstable", 0),
    ("the generated answer is vague and wrong", 0),
    ("the classifier misses important negative clues", 0),
]


def require_torch() -> None:
    """Exit gracefully when PyTorch is missing."""

    # 不安装依赖，保持作业代码对当前环境无侵入。
    if torch is None:
        print("当前环境未安装 PyTorch；本脚本不会自动安装依赖。")
        raise SystemExit(0)


def tokenize(text: str) -> list[str]:
    """Tokenize a sentence."""

    # 简单分词保证注意力权重可以直接映射回原始词。
    return TOKEN_RE.findall(text.lower())


def split_data(seed: int) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Create a deterministic train/validation split."""

    # 固定划分让每次调参结果可复现，也便于保留错误样例。
    rng = random.Random(seed)
    rows = list(DATASET)
    rng.shuffle(rows)
    return rows[:8], rows[8:]


def build_vocab(rows: list[tuple[str, int]]) -> tuple[dict[str, int], list[str]]:
    """Build vocabulary from training rows."""

    # 只用训练集建词表，避免验证集词汇信息泄漏到模型中。
    counter: Counter[str] = Counter()
    for text, _ in rows:
        counter.update(tokenize(text))
    words = sorted(counter, key=lambda word: (-counter[word], word))
    id_to_token = [PAD, UNK] + words
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    return token_to_id, id_to_token


def encode_rows(
    rows: list[tuple[str, int]],
    token_to_id: dict[str, int],
) -> tuple["torch.Tensor", "torch.Tensor", list[str]]:
    """Encode and pad text rows."""

    # 手写 padding 能直观看到 batch 输入如何从变长文本变为矩阵。
    encoded = []
    texts = []
    for text, _ in rows:
        ids = [token_to_id.get(token, token_to_id[UNK]) for token in tokenize(text)]
        encoded.append(ids)
        texts.append(text)
    max_len = max(len(ids) for ids in encoded)
    padded = [ids + [0] * (max_len - len(ids)) for ids in encoded]
    labels = [label for _, label in rows]
    return torch.tensor(padded), torch.tensor(labels), texts


class AttentionClassifier(nn.Module):
    """Embedding classifier with additive attention pooling."""

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        # embedding 层负责学习词级情感线索，如 reliable、wrong、unstable。
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        input_ids: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Return logits and token attention weights."""

        # mask 确保 PAD 不参与注意力归一化和句向量聚合。
        mask = input_ids.ne(0)
        hidden = torch.tanh(self.projection(self.embedding(input_ids)))
        scores = self.attention(hidden).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return self.output(pooled), weights


def evaluate(
    model: AttentionClassifier,
    inputs: "torch.Tensor",
    labels: "torch.Tensor",
    texts: list[str],
) -> tuple[float, list[dict[str, object]]]:
    """Evaluate accuracy and collect mistakes."""

    # 保存错误样例可帮助判断 attention 是否关注到了错误线索。
    model.eval()
    mistakes = []
    with torch.no_grad():
        logits, _ = model(inputs)
        predictions = logits.argmax(dim=1)
        accuracy = float(predictions.eq(labels).float().mean().item())
        for text, gold, pred in zip(texts, labels.tolist(), predictions.tolist()):
            if gold != pred:
                mistakes.append({"text": text, "gold": gold, "predicted": pred})
    return accuracy, mistakes


def train_config(
    args: argparse.Namespace,
    config: dict[str, float | int],
) -> dict[str, object]:
    """Train one sentiment model configuration."""

    # 每组超参数重新初始化模型，保证 trial 之间互不污染。
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows, valid_rows = split_data(args.seed)
    token_to_id, _ = build_vocab(train_rows)
    train_x, train_y, _ = encode_rows(train_rows, token_to_id)
    valid_x, valid_y, valid_texts = encode_rows(valid_rows, token_to_id)

    model = AttentionClassifier(
        vocab_size=len(token_to_id),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    history = []

    # 小数据全量 batch 训练，重点记录调参趋势和验证结果。
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits, _ = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 3, 1) == 0:
            valid_accuracy, _ = evaluate(model, valid_x, valid_y, valid_texts)
            history.append(
                {
                    "epoch": epoch,
                    "loss": round(float(loss.item()), 4),
                    "valid_accuracy": round(valid_accuracy, 4),
                }
            )

    valid_accuracy, mistakes = evaluate(model, valid_x, valid_y, valid_texts)
    return {
        "config": config,
        "valid_accuracy": round(valid_accuracy, 4),
        "history": history,
        "mistakes": mistakes,
    }


def run_tuning(args: argparse.Namespace) -> dict[str, object]:
    """Run a small hyperparameter search."""

    # 搜索空间覆盖表示维度、注意力隐层和学习率。
    search_space = {
        "embedding_dim": [24],
        "hidden_dim": [16, 24],
        "learning_rate": [0.01, 0.03],
    }
    trials = []
    for values in product(*search_space.values()):
        config = dict(zip(search_space.keys(), values))
        trials.append(train_config(args, config))
    best = max(trials, key=lambda row: row["valid_accuracy"])
    return {
        "task": "sentiment_attention_classifier",
        "num_examples": len(DATASET),
        "best_result": best,
        "trials": trials,
    }


def parse_args() -> argparse.Namespace:
    """Parse training configuration."""

    # epochs 与 seed 参数让实验预算和复现性可控。
    parser = argparse.ArgumentParser(description="Toy attention sentiment classifier")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=53)
    return parser.parse_args()


def main() -> None:
    """Run sentiment tuning and save results."""

    # 输出 JSON 记录调参过程和错误样例。
    require_torch()
    result = run_tuning(parse_args())
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sentiment_attention_results.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
