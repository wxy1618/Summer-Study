# -*- coding: utf-8 -*-
"""Skip-gram with Negative Sampling for a small NLP corpus.

本脚本对应第五周“词向量 / Word2vec”部分。实验目标不是追求大语料效果，
而是把 Word2vec 的训练样本构造、负采样、embedding 更新和近邻检索流程写清楚。
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
    # PyTorch 只在实际训练阶段需要；把导入包裹起来，能让脚本在未配置环境时
    # 以“可读说明”的方式退出，而不是在导入阶段直接报错。
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - 仅在缺少 PyTorch 的环境触发
    torch = None

    class _MissingNN:
        Module = object

    nn = _MissingNN()
    Dataset = object
    DataLoader = None
    F = None


TOKEN_RE = re.compile(r"[a-zA-Z]+|[0-9]+")

# 这里的正则分词是受控简化：它牺牲了缩写、标点和大小写细节，
# 但能把注意力集中到 Word2vec 的分布式表示学习机制上。

# 语料故意写得小而集中：它覆盖 NLP、语言模型、依存分析、Transformer 等本周概念。
# 小语料更适合教学，因为可以直接检查“哪些上下文让两个词被拉近”。
CORPUS = [
    "natural language processing studies words syntax and meaning",
    "word vectors represent words in a continuous embedding space",
    "skip gram predicts context words from a center word",
    "negative sampling contrasts observed context with noise words",
    "dependency parsing connects each token to its syntactic head",
    "language models predict the next token from previous context",
    "recurrent networks remember context through hidden states",
    "transformers use attention to compare every token with others",
    "pretrained models transfer language knowledge to downstream tasks",
    "multimodal learning aligns text representation with image features",
    "graduate experiments should record assumptions metrics and errors",
]


def require_torch() -> None:
    """Fail softly when the environment has not installed PyTorch."""

    # 课程任务要求“不配置环境，只生成代码”；因此依赖缺失时只提示，
    # 不在脚本里执行 pip 或 conda，避免破坏已有实验环境。
    if torch is None:
        print("当前环境未安装 PyTorch；本脚本不会自动安装依赖。")
        raise SystemExit(0)


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into simple English tokens."""

    # Word2vec 原论文不绑定具体分词器；这里用简单正则是为了突出训练目标本身。
    # 在真实研究中，分词策略会影响低频词、复合词和领域词的向量质量。
    return TOKEN_RE.findall(text.lower())


def build_vocab(
    sentences: Iterable[str],
    min_count: int,
) -> tuple[dict[str, int], list[str]]:
    """Create word-to-id and id-to-word mappings."""

    # 先统计全语料词频，是为了给 min_count 过滤和负采样分布提供基础。
    # 教学语料很小，所以没有划分训练/验证词表；大规模实验应避免信息泄漏。
    counter: Counter[str] = Counter()
    for sentence in sentences:
        counter.update(tokenize(sentence))

    # 先按词频降序、再按字母序排序，保证每次实验得到相同词表。
    tokens = [
        word
        for word, count in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if count >= min_count
    ]
    # Word2vec 的输入层本质是 token id 查 embedding 表，
    # 因此词到整数的映射是所有后续训练样本的公共索引空间。
    word_to_id = {word: idx for idx, word in enumerate(tokens)}
    return word_to_id, tokens


def encode_corpus(
    sentences: Iterable[str],
    word_to_id: dict[str, int],
) -> list[list[int]]:
    """Turn tokenized sentences into integer ids."""

    # 保留句子边界是为了生成局部上下文窗口时不跨句取样，
    # 否则相邻句子的末尾和开头会被错误地视为上下文关系。
    encoded: list[list[int]] = []
    for sentence in sentences:
        ids = [
            word_to_id[word]
            for word in tokenize(sentence)
            if word in word_to_id
        ]
        # Skip-gram 至少需要一个中心词和一个上下文词；
        # 长度不足的句子不会贡献有效训练对。
        if len(ids) >= 2:
            encoded.append(ids)
    return encoded


def generate_skipgram_pairs(
    encoded: list[list[int]],
    window_size: int,
) -> list[tuple[int, int]]:
    """Generate ``(center_id, context_id)`` pairs for Skip-gram training."""

    # Skip-gram 把“一个中心词预测周围词”拆成多个二元训练样本。
    # 这种拆分让低频中心词也能在多个上下文位置上更新表示。
    pairs: list[tuple[int, int]] = []
    for sentence_ids in encoded:
        for center_index, center_id in enumerate(sentence_ids):
            # 窗口在句子边界处自动截断，避免访问不存在的上下文位置。
            left = max(0, center_index - window_size)
            right = min(len(sentence_ids), center_index + window_size + 1)
            for context_index in range(left, right):
                # 中心词本身不作为自己的上下文，否则模型会学习到退化的自复制关系。
                if context_index == center_index:
                    continue
                pairs.append((center_id, sentence_ids[context_index]))
    return pairs


class SkipGramDataset(Dataset):
    """Small dataset wrapper around precomputed center-context pairs."""

    def __init__(self, pairs: list[tuple[int, int]]) -> None:
        # 预先构造 pair 可以让 Dataset 很轻量，也便于检查样本总数是否合理。
        self.pairs = pairs

    def __len__(self) -> int:
        # DataLoader 依赖长度估计 epoch 内 batch 数，这也是 loss 加权平均的分母。
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[int, int]:
        # 返回 Python 整数即可，DataLoader 会在默认 collate 中组装为张量。
        return self.pairs[index]


class NegativeSampler:
    """Draw negative words from the smoothed unigram distribution."""

    def __init__(
        self,
        encoded: list[list[int]],
        vocab_size: int,
        power: float = 0.75,
    ) -> None:
        # 从 1 开始计数相当于轻微平滑，避免极小语料中某些词概率为零。
        counts = torch.ones(vocab_size, dtype=torch.float32)
        for sentence_ids in encoded:
            for token_id in sentence_ids:
                counts[token_id] += 1.0

        # 3/4 次幂平滑来自 Word2vec 经验做法：高频词仍更常被采样，但不会过度垄断。
        probabilities = counts.pow(power)
        self.probabilities = probabilities / probabilities.sum()

    def sample(self, batch_size: int, num_negative: int) -> "torch.Tensor":
        # replacement=True 符合负采样假设：每个负例独立来自噪声分布。
        # 返回形状 [batch_size, num_negative]，便于后续批量点积计算。
        draws = torch.multinomial(
            self.probabilities,
            batch_size * num_negative,
            replacement=True,
        )
        return draws.view(batch_size, num_negative)


class SkipGramNegSampling(nn.Module):
    """Two embedding tables used by Skip-gram negative sampling."""

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        # Word2vec SGNS 通常使用两套向量：中心词向量和上下文词向量。
        # 训练后常取中心词表征做相似度检索，也可合并两套向量做分析。
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 较小的初始化范围能让训练初期的 sigmoid 不至于过早饱和。
        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.center_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.context_embeddings.weight, -bound, bound)

    def forward(
        self,
        center_ids: "torch.Tensor",
        positive_ids: "torch.Tensor",
        negative_ids: "torch.Tensor",
    ) -> "torch.Tensor":
        # 中心词向量形状为 [batch, dim]，正样本上下文向量形状相同。
        # 负样本向量形状为 [batch, negative, dim]，用于一次性计算多个噪声词。
        center_vectors = self.center_embeddings(center_ids)
        positive_vectors = self.context_embeddings(positive_ids)
        negative_vectors = self.context_embeddings(negative_ids)

        # 点积越大，表示中心词和上下文词在当前嵌入空间中越相容。
        # bmm 用批矩阵乘法避免 Python 循环，是向量化训练的基本写法。
        positive_scores = torch.sum(center_vectors * positive_vectors, dim=1)
        negative_scores = torch.bmm(
            negative_vectors,
            center_vectors.unsqueeze(2),
        ).squeeze(2)

        # 正样本希望 sigmoid(score) 接近 1；负样本希望 sigmoid(score) 接近 0。
        positive_loss = F.logsigmoid(positive_scores)
        negative_loss = F.logsigmoid(-negative_scores).sum(dim=1)
        return -(positive_loss + negative_loss).mean()


def cosine_neighbors(
    model: SkipGramNegSampling,
    id_to_word: list[str],
    query: str,
    top_k: int,
) -> list[dict[str, float | str]]:
    """Return nearest words for a query token."""

    # 对教学脚本而言，未知查询直接返回空列表比抛异常更利于结果汇总。
    if query not in id_to_word:
        return []

    # 余弦相似度只比较方向，不比较向量长度，更符合词义邻近检索的常见做法。
    query_id = id_to_word.index(query)
    weights = model.center_embeddings.weight.detach()
    normalized = F.normalize(weights, dim=1)
    scores = torch.mv(normalized, normalized[query_id])
    best_ids = torch.topk(scores, k=min(top_k + 1, len(id_to_word))).indices.tolist()

    # top_k 结果中会包含查询词自己，因此需要显式跳过 self-match。
    neighbors: list[dict[str, float | str]] = []
    for word_id in best_ids:
        if word_id == query_id:
            continue
        neighbors.append(
            {
                "word": id_to_word[word_id],
                "score": round(float(scores[word_id]), 4),
            }
        )
        if len(neighbors) == top_k:
            break
    return neighbors


def train(args: argparse.Namespace) -> dict[str, object]:
    """Train the toy Word2vec model and collect experiment artifacts."""

    # 固定随机种子是实验记录的最低要求：负采样、shuffle 和初始化都会引入随机性。
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 数据预处理阶段依次完成词表、整数编码和 Skip-gram 训练对构造。
    # 这三个对象也是复现实验时最需要检查的中间状态。
    word_to_id, id_to_word = build_vocab(CORPUS, min_count=args.min_count)
    encoded = encode_corpus(CORPUS, word_to_id)
    pairs = generate_skipgram_pairs(encoded, window_size=args.window_size)

    # sampler 与模型分开，是为了强调“负例来自数据分布”，
    # 而不是模型结构本身的一部分。
    dataset = SkipGramDataset(pairs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    sampler = NegativeSampler(encoded, vocab_size=len(id_to_word))
    model = SkipGramNegSampling(len(id_to_word), args.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for center_ids, context_ids in loader:
            # 每个正样本配多个负样本，提高二分类信号的判别性。
            negative_ids = sampler.sample(center_ids.size(0), args.num_negative)
            loss = model(center_ids, context_ids, negative_ids)

            # 典型 PyTorch 三步：清梯度、反向传播、参数更新。
            # 这里没有梯度裁剪，因为 SGNS 小模型通常稳定。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * center_ids.size(0)

        # 按样本数加权平均，比简单平均 batch loss 更不受最后一个小 batch 影响。
        average_loss = total_loss / len(dataset)
        history.append({"epoch": epoch, "loss": round(average_loss, 4)})

    # 查询词覆盖课程关键词，用于观察学到的局部语义邻近关系是否合理。
    queries = ["language", "word", "context", "models", "token"]
    neighbors = {
        query: cosine_neighbors(model, id_to_word, query, top_k=args.top_k)
        for query in queries
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # JSON 记录面向周报和笔记，包含超参数、loss 曲线和可解释的近邻词。
    result = {
        "task": "skipgram_negative_sampling",
        "vocab_size": len(id_to_word),
        "num_pairs": len(pairs),
        "embedding_dim": args.embedding_dim,
        "final_loss": history[-1]["loss"],
        "loss_history": history,
        "nearest_neighbors": neighbors,
    }
    (output_dir / "word2vec_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 保存张量文件方便后续复现实验，但课堂笔记中主要使用 JSON 结果即可。
    torch.save(model.state_dict(), output_dir / "word2vec_skipgram.pt")
    return result


def parse_args() -> argparse.Namespace:
    # 命令行参数把实验配置显式化，便于后续对窗口大小、负采样数等做消融。
    parser = argparse.ArgumentParser(description="Toy Word2vec Skip-gram experiment")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--num-negative", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    # main 函数只负责依赖检查、训练调用和结果打印，使核心实验逻辑保持可测试。
    require_torch()
    result = train(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"实验结束，最终 loss = {result['final_loss']}")


if __name__ == "__main__":
    main()
