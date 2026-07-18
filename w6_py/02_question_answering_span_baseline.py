# -*- coding: utf-8 -*-
"""Extractive question answering span baseline with tuning records.

本脚本对应第六周问答任务。它不依赖深度学习框架，而是构造一个可解释的
span 打分基线，用于理解 QA 中“问题-上下文匹配、答案类型约束、评价指标”
这些比模型结构更基础的实验环节。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path


# 正则分词保持实验透明：所有候选 span、关键词重合和答案类型判断都可复查。
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# 停用词只用于降低功能词权重，避免问题中的 who/what/the 等词主导 span 得分。
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "in",
    "is",
    "of",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
}


# 轻量实体词典是小型 QA baseline 的领域知识来源，类似真实系统中的 gazetteer。
TYPE_LEXICON = {
    "person": {"marie curie", "alan turing", "ada lovelace"},
    "place": {"paris", "cambridge"},
    "object": {"radium", "turing machine", "analytical engine", "attention"},
}


@dataclass(frozen=True)
class QAExample:
    """A small extractive QA example with one gold answer string."""

    # context 是证据文本，question 是查询，answer 必须能在 context 中找到。
    context: str
    question: str
    answer: str
    answer_type: str


# 数据集刻意覆盖 person/place/date/object 四种答案类型，方便观察类型约束作用。
DATASET = [
    QAExample(
        "Marie Curie discovered radium in Paris in 1898.",
        "Who discovered radium?",
        "Marie Curie",
        "person",
    ),
    QAExample(
        "Marie Curie discovered radium in Paris in 1898.",
        "Where did Marie Curie discover radium?",
        "Paris",
        "place",
    ),
    QAExample(
        "Marie Curie discovered radium in Paris in 1898.",
        "When did Marie Curie discover radium?",
        "1898",
        "date",
    ),
    QAExample(
        "Alan Turing designed the Turing machine in Cambridge.",
        "Who designed the Turing machine?",
        "Alan Turing",
        "person",
    ),
    QAExample(
        "Ada Lovelace wrote notes about the analytical engine.",
        "What did Ada Lovelace write about?",
        "analytical engine",
        "object",
    ),
    QAExample(
        "The transformer model uses attention for contextual representation.",
        "What does the transformer model use?",
        "attention",
        "object",
    ),
]


def tokenize(text: str) -> list[str]:
    """Convert text to lowercase word tokens."""

    # 小写化让关键词匹配不受句首大写影响，便于教学基线稳定复现。
    return TOKEN_RE.findall(text.lower())


def content_words(question: str) -> set[str]:
    """Return informative words from a question."""

    # 问题关键词是抽取式 QA 的主要匹配信号，但需要去除功能词噪声。
    return {token for token in tokenize(question) if token not in STOP_WORDS}


def candidate_spans(context: str, max_len: int) -> list[tuple[str, int, int]]:
    """Enumerate short contiguous spans from the context."""

    # 真实抽取式 QA 通常预测 start/end 位置；这里直接枚举短 span 模拟该搜索空间。
    raw_tokens = TOKEN_RE.findall(context)
    spans: list[tuple[str, int, int]] = []

    # max_len 控制候选答案长度，过大容易包含无关词，过小会漏掉多词实体。
    for start in range(len(raw_tokens)):
        for end in range(start + 1, min(len(raw_tokens), start + max_len) + 1):
            spans.append((" ".join(raw_tokens[start:end]), start, end))
    return spans


def answer_type_score(span: str, answer_type: str) -> float:
    """Score whether a candidate span matches the expected answer type."""

    # 类型约束来自问题词：who 偏 person，where 偏 place，when 偏 date。
    tokens = span.split()
    normalized = " ".join(token.lower() for token in tokens)
    if normalized in TYPE_LEXICON.get(answer_type, set()):
        return 2.0

    if answer_type == "date":
        return 1.0 if any(token.isdigit() for token in tokens) else 0.0

    # 人名在英文小实验中常由连续首字母大写词组成，这是一个简单但可解释的线索。
    if answer_type == "person":
        looks_like_name = (
            len(tokens) >= 2
            and all(token[:1].isupper() for token in tokens)
        )
        return 0.8 if looks_like_name else 0.0

    # 地点词通常是单个首字母大写实体；这里避免把两词人名误判为地点。
    if answer_type == "place":
        return 0.6 if len(tokens) == 1 and tokens[0][:1].isupper() else 0.0

    # object 类型更开放，不做强约束，避免过度规则化导致召回下降。
    return 0.3


def score_span(
    span: str,
    start: int,
    question_words: set[str],
    answer_type: str,
    weights: dict[str, float],
) -> float:
    """Assign a heuristic score to one answer span."""

    # span 内词与问题关键词重合，通常说明候选周围包含同一主题。
    span_tokens = set(tokenize(span))
    overlap = len(span_tokens & question_words)

    # 长度惩罚防止模型总是选择很长的片段；位置惩罚略偏向前部证据。
    length_penalty = len(span_tokens)
    type_bonus = answer_type_score(span, answer_type)
    position_penalty = math.log1p(start)

    # 可调权重让这个基线具备“模型调优”性质，而不是固定规则。
    return (
        weights["overlap"] * overlap
        + weights["type"] * type_bonus
        - weights["length"] * length_penalty
        - weights["position"] * position_penalty
    )


def predict_answer(example: QAExample, weights: dict[str, float]) -> str:
    """Select the highest-scoring span for one QA example."""

    # 每个问题都独立生成候选 span，再用同一套权重打分，模拟参数共享。
    q_words = content_words(example.question)
    scored = []
    for span, start, _ in candidate_spans(example.context, max_len=3):
        score = score_span(span, start, q_words, example.answer_type, weights)
        scored.append((score, span))

    # 分数相同时偏向较短输出，减少“答案外多带一个词”的常见错误。
    scored.sort(key=lambda item: (item[0], -len(item[1].split())), reverse=True)
    return scored[0][1]


def token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 for extractive QA."""

    # QA 评价中 exact match 很严格，F1 能部分认可“包含正确核心词”的预测。
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    common = set(pred_tokens) & set(gold_tokens)

    # 没有重合词时 precision 和 recall 都为零。
    if not common:
        return 0.0

    precision = len(common) / max(len(pred_tokens), 1)
    recall = len(common) / max(len(gold_tokens), 1)
    return 2 * precision * recall / (precision + recall)


def evaluate(weights: dict[str, float]) -> dict[str, object]:
    """Evaluate one weight configuration on the toy QA dataset."""

    # 同时记录整体指标和错误样例，避免只凭平均分判断模型是否可靠。
    exact_matches = 0
    f1_scores = []
    predictions = []

    # 每个样本都保留问题、预测、答案和指标，便于后续写实验分析。
    for example in DATASET:
        prediction = predict_answer(example, weights)
        em = tokenize(prediction) == tokenize(example.answer)
        f1 = token_f1(prediction, example.answer)
        exact_matches += int(em)
        f1_scores.append(f1)
        predictions.append(
            {
                "question": example.question,
                "prediction": prediction,
                "gold": example.answer,
                "exact_match": em,
                "f1": round(f1, 4),
            }
        )

    # EM 强调完全正确，F1 反映部分重合，两者结合更适合抽取式 QA 小实验。
    return {
        "weights": weights,
        "exact_match": round(exact_matches / len(DATASET), 4),
        "f1": round(sum(f1_scores) / len(f1_scores), 4),
        "predictions": predictions,
    }


def tune_weights() -> dict[str, object]:
    """Run a small grid search over interpretable QA scoring weights."""

    # 调参范围保持很小，是为了让结果可解释，并能在 CPU 上瞬间完成。
    grid = {
        "overlap": [0.5, 1.0, 1.5],
        "type": [0.5, 1.0, 2.0],
        "length": [0.05, 0.1, 0.2],
        "position": [0.0, 0.05],
    }
    trials = []

    # product 枚举所有组合，形成可复查的超参数搜索记录。
    for values in product(*grid.values()):
        weights = dict(zip(grid.keys(), values))
        result = evaluate(weights)
        trials.append(result)

    # 首先按 F1，其次按 EM 选择最优配置，兼顾片段级和完全匹配表现。
    best = max(trials, key=lambda row: (row["f1"], row["exact_match"]))
    return {
        "task": "extractive_question_answering_baseline",
        "num_examples": len(DATASET),
        "best_result": best,
        "top_trials": sorted(
            trials,
            key=lambda row: (row["f1"], row["exact_match"]),
            reverse=True,
        )[:5],
    }


def main() -> None:
    """Run tuning and save a reproducible QA experiment report."""

    # outputs 目录由脚本运行时创建，不要求用户提前配置任何文件结构。
    result = tune_weights()
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 文件比纯终端输出更适合写入周报，也便于后续比较不同版本。
    output_path = output_dir / "qa_span_baseline_results.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
