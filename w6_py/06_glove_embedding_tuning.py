# -*- coding: utf-8 -*-
"""Toy GloVe-style word embedding training with window tuning.

本脚本对应第六周 GloVe 词嵌入实战。它从小语料构建词共现矩阵，
训练 GloVe 风格目标函数，并记录不同窗口大小下的 loss 和近邻词。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

try:
    # PyTorch 用于优化词向量；脚本不会尝试安装或配置环境。
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


# 语料覆盖第六周核心任务词，便于观察近邻词是否具有主题相关性。
CORPUS = [
    "translation models use attention for alignment",
    "question answering selects evidence from context",
    "generation models decode fluent natural language",
    "coreference resolution links mentions to entities",
    "transe learns relation embeddings in knowledge graphs",
    "glove learns word embeddings from cooccurrence statistics",
    "sentiment analysis attends to opinion words",
    "relation extraction classifies entity pairs in text",
]


def require_torch() -> None:
    """Stop if PyTorch is unavailable."""

    # 保持“不配置环境”的要求，缺失依赖时只打印说明。
    if torch is None:
        print("当前环境未安装 PyTorch；本脚本不会自动安装依赖。")
        raise SystemExit(0)


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words."""

    # GloVe 基于共现统计，分词粒度会直接影响共现矩阵质量。
    return TOKEN_RE.findall(text.lower())


def build_vocab(corpus: list[str]) -> tuple[dict[str, int], list[str]]:
    """Build deterministic vocabulary from the corpus."""

    # 词频排序让高频词 id 更靠前，方便后续观察词表和共现矩阵。
    counts: Counter[str] = Counter()
    for sentence in corpus:
        counts.update(tokenize(sentence))
    words = sorted(counts, key=lambda word: (-counts[word], word))
    token_to_id = {word: index for index, word in enumerate(words)}
    return token_to_id, words


def build_cooccurrence(
    corpus: list[str],
    token_to_id: dict[str, int],
    window_size: int,
) -> list[tuple[int, int, float]]:
    """Construct weighted word cooccurrence pairs."""

    # 使用距离倒数作为权重，近邻词对比远邻词贡献更大。
    matrix: dict[tuple[int, int], float] = defaultdict(float)
    for sentence in corpus:
        ids = [token_to_id[token] for token in tokenize(sentence)]
        for center_index, center_id in enumerate(ids):
            left = max(0, center_index - window_size)
            right = min(len(ids), center_index + window_size + 1)
            for context_index in range(left, right):
                if center_index == context_index:
                    continue
                distance = abs(center_index - context_index)
                matrix[(center_id, ids[context_index])] += 1.0 / distance
    return [(i, j, value) for (i, j), value in matrix.items()]


class GloVeModel(nn.Module):
    """GloVe objective with word/context embeddings and biases."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        # GloVe 使用两套词向量和两个 bias 项拟合 log 共现次数。
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_bias = nn.Embedding(vocab_size, 1)
        self.context_bias = nn.Embedding(vocab_size, 1)

        # 小范围初始化让初期点积接近 0，避免 loss 一开始过大。
        nn.init.uniform_(self.word_embeddings.weight, -0.1, 0.1)
        nn.init.uniform_(self.context_embeddings.weight, -0.1, 0.1)
        nn.init.zeros_(self.word_bias.weight)
        nn.init.zeros_(self.context_bias.weight)

    def forward(
        self,
        word_ids: "torch.Tensor",
        context_ids: "torch.Tensor",
        counts: "torch.Tensor",
        x_max: float,
        alpha: float,
    ) -> "torch.Tensor":
        """Compute weighted GloVe loss."""

        # 权重函数降低超高频共现对目标函数的支配，同时保留稳定统计。
        weights = torch.clamp((counts / x_max).pow(alpha), max=1.0)
        dots = (
            self.word_embeddings(word_ids)
            * self.context_embeddings(context_ids)
        ).sum(dim=1)
        bias = (
            self.word_bias(word_ids).squeeze(1)
            + self.context_bias(context_ids).squeeze(1)
        )
        residual = dots + bias - torch.log(counts)
        return (weights * residual.pow(2)).mean()

    def combined_embeddings(self) -> "torch.Tensor":
        """Return final word vectors for similarity analysis."""

        # 常见做法是把 word/context 两套向量相加或平均作为最终词向量。
        return (
            self.word_embeddings.weight.detach()
            + self.context_embeddings.weight.detach()
        )


def nearest_words(
    embeddings: "torch.Tensor",
    id_to_token: list[str],
    query: str,
    top_k: int,
) -> list[dict[str, float | str]]:
    """Find nearest neighbors with cosine similarity."""

    # 近邻词用于定性检查共现训练是否学到主题相关结构。
    if query not in id_to_token:
        return []
    query_id = id_to_token.index(query)
    normalized = F.normalize(embeddings, dim=1)
    scores = normalized @ normalized[query_id]
    best_ids = torch.topk(scores, k=min(top_k + 1, len(id_to_token))).indices.tolist()

    # 去掉查询词自身，返回最接近的其他词。
    rows = []
    for token_id in best_ids:
        if token_id == query_id:
            continue
        rows.append(
            {
                "word": id_to_token[token_id],
                "score": round(float(scores[token_id]), 4),
            }
        )
        if len(rows) == top_k:
            break
    return rows


def train_config(
    cooccurrence: list[tuple[int, int, float]],
    vocab_size: int,
    id_to_token: list[str],
    args: argparse.Namespace,
    config: dict[str, float | int],
) -> dict[str, object]:
    """Train one GloVe configuration and return metrics."""

    # 同一 seed 让不同窗口大小和维度的比较更公平。
    torch.manual_seed(args.seed)
    model = GloVeModel(vocab_size, int(config["embedding_dim"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    # 共现数据一次性转为张量，小语料无需 DataLoader。
    word_ids = torch.tensor([row[0] for row in cooccurrence], dtype=torch.long)
    context_ids = torch.tensor([row[1] for row in cooccurrence], dtype=torch.long)
    counts = torch.tensor([row[2] for row in cooccurrence], dtype=torch.float32)
    history = []

    # 每个 epoch 对全量共现矩阵优化一次，记录若干关键点的 loss。
    for epoch in range(1, args.epochs + 1):
        loss = model(word_ids, context_ids, counts, args.x_max, args.alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 4, 1) == 0:
            history.append({"epoch": epoch, "loss": round(float(loss.item()), 4)})

    # 最终近邻词帮助判断 embedding 是否按任务主题聚合。
    embeddings = model.combined_embeddings()
    neighbors = {
        query: nearest_words(embeddings, id_to_token, query, top_k=4)
        for query in ["attention", "relation", "generation"]
    }
    return {
        "config": config,
        "num_cooccurrence_pairs": len(cooccurrence),
        "final_loss": history[-1]["loss"],
        "history": history,
        "neighbors": neighbors,
    }


def run_tuning(args: argparse.Namespace) -> dict[str, object]:
    """Tune window size and embedding dimension for the GloVe toy corpus."""

    # 词表固定后，只改变共现窗口和模型超参数，便于比较影响来源。
    token_to_id, id_to_token = build_vocab(CORPUS)
    trials = []
    search_space = {
        "window_size": [2, 3],
        "embedding_dim": [16, 24],
        "learning_rate": [0.03, 0.05],
    }

    # 每组配置都重新构建对应窗口的共现矩阵。
    for values in product(*search_space.values()):
        config = dict(zip(search_space.keys(), values))
        cooccurrence = build_cooccurrence(
            CORPUS,
            token_to_id,
            int(config["window_size"]),
        )
        trials.append(
            train_config(
                cooccurrence,
                len(id_to_token),
                id_to_token,
                args,
                config,
            )
        )

    # GloVe toy 实验以 final_loss 作为主要调参指标，并结合近邻词人工观察。
    best = min(trials, key=lambda row: row["final_loss"])
    return {
        "task": "glove_embedding_tuning",
        "vocab_size": len(id_to_token),
        "best_result": best,
        "trials": trials,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line configuration."""

    # x_max 和 alpha 是 GloVe 权重函数的关键超参数。
    parser = argparse.ArgumentParser(description="Toy GloVe tuning experiment")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--x-max", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def main() -> None:
    """Run GloVe tuning and save results."""

    # 结果落盘，方便在周报中引用最优配置和近邻词现象。
    require_torch()
    result = run_tuning(parse_args())
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "glove_tuning_results.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
