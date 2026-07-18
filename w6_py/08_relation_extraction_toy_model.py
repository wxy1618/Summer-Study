# -*- coding: utf-8 -*-
"""Toy relation extraction classifier with entity markers and tuning.

本脚本对应第六周关系抽取模型实战。它用实体标记后的句子训练轻量分类器，
记录 macro-F1、调参结果和错误样例。
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
    # PyTorch 用于训练关系分类模型；脚本不负责安装依赖。
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None

    class _MissingNN:
        Module = object

    nn = _MissingNN()
    F = None


TOKEN_RE = re.compile(r"[A-Za-z0-9_<>/]+")
PAD = "<pad>"
UNK = "<unk>"


# 句子中用 <e1>、</e1>、<e2>、</e2> 显式标记两个实体。
DATASET = [
    ("<e1> Marie_Curie </e1> discovered <e2> radium </e2>", "discover"),
    ("<e1> Researchers </e1> discovered <e2> new_facts </e2>", "discover"),
    ("<e1> Alan_Turing </e1> designed <e2> Turing_machine </e2>", "create"),
    ("<e1> Engineers </e1> designed <e2> sequence_model </e2>", "create"),
    ("<e1> Ada_Lovelace </e1> wrote about <e2> analytical_engine </e2>", "write_about"),
    ("<e1> Students </e1> wrote about <e2> language_models </e2>", "write_about"),
    ("<e1> Transformer </e1> uses <e2> attention </e2>", "use"),
    ("<e1> GloVe </e1> learns from <e2> cooccurrence </e2>", "use"),
    ("<e1> Bob </e1> works in <e2> Cambridge </e2>", "work_in"),
    ("<e1> Alice </e1> works in <e2> Paris </e2>", "work_in"),
    ("<e1> Alice </e1> studies <e2> question_answering </e2>", "study"),
    ("<e1> Students </e1> study <e2> text_generation </e2>", "study"),
    ("<e1> TransE </e1> models <e2> knowledge_graph </e2>", "model"),
    ("<e1> Coreference </e1> models <e2> mention_links </e2>", "model"),
]


def require_torch() -> None:
    """Exit when PyTorch is not available."""

    # 遵守“不配置环境”要求，缺包时只提示。
    if torch is None:
        print("当前环境未安装 PyTorch；本脚本不会自动安装依赖。")
        raise SystemExit(0)


def tokenize(text: str) -> list[str]:
    """Tokenize text while preserving entity markers."""

    # 保留实体标记可以让模型知道关系判断的两个对象在哪里。
    return TOKEN_RE.findall(text.lower())


def split_data(seed: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Deterministically split relation examples."""

    # 按关系标签分层切分，避免验证集中出现训练阶段完全未见过的关系。
    rng = random.Random(seed)
    by_label: dict[str, list[tuple[str, str]]] = {}
    for row in DATASET:
        by_label.setdefault(row[1], []).append(row)

    # 每个标签留一条做验证，其余进入训练；这是小数据关系抽取的常见折中。
    train_rows = []
    valid_rows = []
    for rows in by_label.values():
        rows = list(rows)
        rng.shuffle(rows)
        valid_rows.append(rows[0])
        train_rows.extend(rows[1:])
    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)
    return train_rows, valid_rows


def build_vocab_and_labels(
    rows: list[tuple[str, str]],
) -> tuple[dict[str, int], dict[str, int], list[str]]:
    """Build token and label mappings from training data."""

    # 词表从训练集构建，标签表用全数据构建以避免验证标签未知。
    counter: Counter[str] = Counter()
    for text, _ in rows:
        counter.update(tokenize(text))
    words = sorted(counter, key=lambda word: (-counter[word], word))
    token_to_id = {token: idx for idx, token in enumerate([PAD, UNK] + words)}
    labels = sorted({label for _, label in DATASET})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    return token_to_id, label_to_id, labels


def encode_rows(
    rows: list[tuple[str, str]],
    token_to_id: dict[str, int],
    label_to_id: dict[str, int],
) -> tuple["torch.Tensor", "torch.Tensor", list[str]]:
    """Encode examples as padded tensors."""

    # 关系抽取输入需要保留整句上下文，因为关系触发词可能在实体之间或实体外。
    encoded = []
    texts = []
    for text, _ in rows:
        ids = [token_to_id.get(token, token_to_id[UNK]) for token in tokenize(text)]
        encoded.append(ids)
        texts.append(text)
    max_len = max(len(ids) for ids in encoded)
    padded = [ids + [0] * (max_len - len(ids)) for ids in encoded]
    labels = [label_to_id[label] for _, label in rows]
    return torch.tensor(padded), torch.tensor(labels), texts


class RelationClassifier(nn.Module):
    """Mean-pooling relation classifier with entity marker embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_labels: int,
    ) -> None:
        super().__init__()
        # 轻量模型用于观察数据处理和标签预测，不追求大规模 OpenNRE 性能。
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        """Compute relation logits."""

        # mask-aware mean pooling 汇总句子上下文，同时排除 padding。
        mask = input_ids.ne(0).float()
        embedded = self.embedding(input_ids)
        pooled = (embedded * mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        return self.classifier(pooled)


def macro_f1(predictions: list[int], labels: list[int], num_labels: int) -> float:
    """Compute macro-F1 over relation labels."""

    # macro-F1 对每个类别同等看待，适合类别较不均衡的关系抽取任务。
    scores = []
    for label_id in range(num_labels):
        tp = sum(p == label_id and y == label_id for p, y in zip(predictions, labels))
        fp = sum(p == label_id and y != label_id for p, y in zip(predictions, labels))
        fn = sum(p != label_id and y == label_id for p, y in zip(predictions, labels))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
    return sum(scores) / len(scores)


def evaluate(
    model: RelationClassifier,
    inputs: "torch.Tensor",
    labels: "torch.Tensor",
    texts: list[str],
    id_to_label: list[str],
) -> tuple[float, list[dict[str, object]]]:
    """Evaluate macro-F1 and collect mistakes."""

    # 错误样例记录有助于判断模型是否混淆 create/use/model 等相近关系。
    model.eval()
    mistakes = []
    with torch.no_grad():
        logits = model(inputs)
        preds = logits.argmax(dim=1).tolist()
        golds = labels.tolist()
        score = macro_f1(preds, golds, len(id_to_label))
        for text, gold, pred in zip(texts, golds, preds):
            if gold != pred:
                mistakes.append(
                    {
                        "text": text,
                        "gold": id_to_label[gold],
                        "predicted": id_to_label[pred],
                    }
                )
    return score, mistakes


def train_config(
    args: argparse.Namespace,
    config: dict[str, float | int],
) -> dict[str, object]:
    """Train one relation extraction configuration."""

    # 每组配置重新划分和初始化，使调参记录可复查。
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rows, valid_rows = split_data(args.seed)
    token_to_id, label_to_id, id_to_label = build_vocab_and_labels(train_rows)
    train_x, train_y, _ = encode_rows(train_rows, token_to_id, label_to_id)
    valid_x, valid_y, valid_texts = encode_rows(valid_rows, token_to_id, label_to_id)

    model = RelationClassifier(
        vocab_size=len(token_to_id),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_labels=len(id_to_label),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    history = []

    # 全量 batch 训练适合小数据；真实 OpenNRE 任务需要 DataLoader 和大语料。
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 3, 1) == 0:
            valid_f1, _ = evaluate(model, valid_x, valid_y, valid_texts, id_to_label)
            history.append(
                {
                    "epoch": epoch,
                    "loss": round(float(loss.item()), 4),
                    "valid_macro_f1": round(valid_f1, 4),
                }
            )

    valid_f1, mistakes = evaluate(model, valid_x, valid_y, valid_texts, id_to_label)
    return {
        "config": config,
        "valid_macro_f1": round(valid_f1, 4),
        "history": history,
        "mistakes": mistakes,
    }


def run_tuning(args: argparse.Namespace) -> dict[str, object]:
    """Tune the relation extraction baseline."""

    # 搜索基础容量和学习率，满足阶段二“模型调优、记录结果”的要求。
    search_space = {
        "embedding_dim": [24],
        "hidden_dim": [16, 24],
        "learning_rate": [0.01, 0.03],
    }
    trials = []
    for values in product(*search_space.values()):
        config = dict(zip(search_space.keys(), values))
        trials.append(train_config(args, config))
    best = max(trials, key=lambda row: row["valid_macro_f1"])
    return {
        "task": "relation_extraction_toy_model",
        "num_examples": len(DATASET),
        "best_result": best,
        "trials": trials,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    # epochs 和 seed 让训练预算与随机性控制显式化。
    parser = argparse.ArgumentParser(description="Toy relation extraction model")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--seed", type=int, default=67)
    return parser.parse_args()


def main() -> None:
    """Run relation extraction tuning and save results."""

    # 生成 outputs JSON，便于直接整理进实验记录。
    require_torch()
    result = run_tuning(parse_args())
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "relation_extraction_results.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
