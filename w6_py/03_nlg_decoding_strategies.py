# -*- coding: utf-8 -*-
"""Natural language generation with trigram modeling and decoding strategies.

本脚本对应第六周自然语言生成任务。它用标准库实现一个可解释的 trigram
语言模型，对比 greedy、beam search、top-k sampling 和 top-p sampling。
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


# 分词规则保持简洁，便于把概率表、生成路径和 distinct 指标逐项检查。
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
BOS = "<bos>"
EOS = "<eos>"


# 语料围绕第六周主题组织，让生成样例能体现课程语义而不是随机英文句子。
CORPUS = [
    "translation models align source tokens with target tokens",
    "question answering systems select evidence from a context",
    "natural language generation requires decoding and evaluation",
    "coreference resolution links mentions that refer to the same entity",
    "glove embeddings learn from global word cooccurrence statistics",
    "transe embeddings model relations as translations in vector space",
    "sentiment analysis benefits from attention over opinion words",
    "relation extraction predicts semantic links between marked entities",
    "large language model evaluation should record prompts and failures",
]


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words."""

    # 统一小写减少稀疏性；小语料若保留大小写会让 n-gram 统计更碎片化。
    return TOKEN_RE.findall(text.lower())


class TrigramLanguageModel:
    """A smoothed trigram language model for decoding experiments."""

    def __init__(self, alpha: float = 0.2) -> None:
        # alpha 是加性平滑系数，防止未见 trigram 概率为零。
        self.alpha = alpha
        self.context_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def fit(self, corpus: list[str]) -> None:
        """Estimate trigram counts from a small corpus."""

        # 每句前加两个 BOS，使模型在开头也有长度为 2 的上下文。
        for sentence in corpus:
            tokens = [BOS, BOS] + tokenize(sentence) + [EOS]
            self.vocab.update(tokens)
            for index in range(2, len(tokens)):
                context = (tokens[index - 2], tokens[index - 1])
                self.context_counts[context][tokens[index]] += 1

        # 解码时不应生成 BOS，因此从可生成词表中排除它。
        self.vocab.discard(BOS)

    def distribution(self, context: tuple[str, str]) -> list[tuple[str, float]]:
        """Return a smoothed next-token distribution."""

        # 若上下文未见过，仍使用全词表均匀平滑概率，保证生成不会中断。
        counts = self.context_counts.get(context, Counter())
        vocab = sorted(self.vocab)
        denominator = sum(counts.values()) + self.alpha * len(vocab)
        rows = []

        # 每个候选词都获得 alpha 平滑后的概率。
        for token in vocab:
            probability = (counts[token] + self.alpha) / denominator
            rows.append((token, probability))

        # 降序排序便于 greedy、beam、top-k 等解码策略复用。
        return sorted(rows, key=lambda item: item[1], reverse=True)

    def sequence_log_prob(self, tokens: list[str]) -> float:
        """Compute log probability of a complete token sequence."""

        # 评价困惑度时把句子重新补上 BOS/EOS，与训练统计口径一致。
        sequence = [BOS, BOS] + tokens + [EOS]
        log_prob = 0.0
        for index in range(2, len(sequence)):
            context = (sequence[index - 2], sequence[index - 1])
            dist = dict(self.distribution(context))
            log_prob += math.log(dist.get(sequence[index], 1e-12))
        return log_prob


def greedy_decode(model: TrigramLanguageModel, max_len: int) -> list[str]:
    """Generate by always choosing the most probable next token."""

    # greedy 的优点是稳定，缺点是容易陷入高频模板和重复表达。
    tokens = [BOS, BOS]
    for _ in range(max_len):
        next_token = model.distribution((tokens[-2], tokens[-1]))[0][0]
        if next_token == EOS:
            break
        tokens.append(next_token)
    return tokens[2:]


def beam_decode(model: TrigramLanguageModel, beam_size: int, max_len: int) -> list[str]:
    """Generate with beam search."""

    # beam search 保留多个候选序列，减少单步贪心造成的早期错误。
    beams = [([BOS, BOS], 0.0)]
    for _ in range(max_len):
        candidates = []
        for prefix, score in beams:
            if prefix[-1] == EOS:
                candidates.append((prefix, score))
                continue
            for token, prob in model.distribution((prefix[-2], prefix[-1]))[:beam_size]:
                candidates.append((prefix + [token], score + math.log(prob)))

        # 用平均 log probability 近似长度归一化，避免短句天然占优。
        beams = sorted(
            candidates,
            key=lambda item: item[1] / max(len(item[0]) - 2, 1),
            reverse=True,
        )[:beam_size]

    best = beams[0][0]
    return [token for token in best[2:] if token != EOS]


def sample_decode(
    model: TrigramLanguageModel,
    mode: str,
    max_len: int,
    top_k: int = 4,
    top_p: float = 0.8,
    seed: int = 42,
) -> list[str]:
    """Generate with top-k or top-p sampling."""

    # sampling 用固定随机种子，既能展示多样性，也能保持实验复现。
    rng = random.Random(seed)
    tokens = [BOS, BOS]

    # 每一步根据 mode 截断候选集合，然后按归一化概率采样。
    for _ in range(max_len):
        dist = model.distribution((tokens[-2], tokens[-1]))
        if mode == "top_k":
            candidates = dist[:top_k]
        elif mode == "top_p":
            candidates = []
            cumulative = 0.0
            for token, prob in dist:
                candidates.append((token, prob))
                cumulative += prob
                if cumulative >= top_p:
                    break
        else:
            raise ValueError(f"Unsupported sampling mode: {mode}")

        total = sum(prob for _, prob in candidates)
        threshold = rng.random()
        cumulative = 0.0
        next_token = candidates[-1][0]
        for token, prob in candidates:
            cumulative += prob / total
            if cumulative >= threshold:
                next_token = token
                break

        if next_token == EOS:
            break
        tokens.append(next_token)
    return tokens[2:]


def distinct_n(sentences: list[list[str]], n: int) -> float:
    """Compute distinct-n diversity for generated outputs."""

    # distinct-n 衡量生成文本中不同 n-gram 的比例，常用于观察多样性。
    all_ngrams = []
    for tokens in sentences:
        all_ngrams.extend(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def corpus_perplexity(model: TrigramLanguageModel, corpus: list[str]) -> float:
    """Compute a small-corpus perplexity estimate."""

    # 困惑度用于衡量模型对训练语料的平均不确定性。
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in corpus:
        tokens = tokenize(sentence)
        total_log_prob += model.sequence_log_prob(tokens)
        total_tokens += len(tokens) + 1
    return math.exp(-total_log_prob / max(total_tokens, 1))


def run_experiment() -> dict[str, object]:
    """Train the trigram model and compare decoding strategies."""

    # alpha 是可调超参数；这里保留单值，也在结果中显式记录。
    model = TrigramLanguageModel(alpha=0.2)
    model.fit(CORPUS)

    # 每种解码策略都生成一个样例，便于在报告中做定性对比。
    generations = {
        "greedy": greedy_decode(model, max_len=10),
        "beam": beam_decode(model, beam_size=3, max_len=10),
        "top_k": sample_decode(model, "top_k", max_len=10, top_k=4, seed=7),
        "top_p": sample_decode(model, "top_p", max_len=10, top_p=0.75, seed=11),
    }
    generated_lists = list(generations.values())

    # 结果同时包含概率指标和多样性指标，避免只凭单个生成样例下结论。
    return {
        "task": "natural_language_generation_decoding",
        "alpha": model.alpha,
        "train_perplexity": round(corpus_perplexity(model, CORPUS), 4),
        "distinct_1": round(distinct_n(generated_lists, 1), 4),
        "distinct_2": round(distinct_n(generated_lists, 2), 4),
        "generations": {
            name: " ".join(tokens)
            for name, tokens in generations.items()
        },
    }


def main() -> None:
    """Run the NLG experiment and save results."""

    # 输出 JSON 便于把解码策略对比直接整理到学习笔记或周报中。
    result = run_experiment()
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "nlg_decoding_results.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
