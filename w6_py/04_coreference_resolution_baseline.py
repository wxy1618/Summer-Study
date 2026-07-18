# -*- coding: utf-8 -*-
"""Coreference resolution baseline with interpretable tuning.

本脚本对应第六周共指消解任务。它构造 mention、实体类型和代词约束，
用可调打分函数完成聚类，并记录 pairwise F1 和错误样例。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path


# mention 抽取在真实系统中通常由神经模型完成；本脚本用标注 mention 聚焦消解。
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Mention:
    """A candidate mention that may refer to an entity."""

    # mention_id 用于评价聚类，text 是词面，entity_type 提供语义兼容约束。
    mention_id: str
    text: str
    sentence_id: int
    entity_type: str
    gold_cluster: str


# 数据集覆盖人、论文、模型三类实体，以及 he/she/it/they 等常见代词。
MENTIONS = [
    Mention("m1", "Alice", 0, "person_female", "c1"),
    Mention("m2", "a paper", 0, "object", "c2"),
    Mention("m3", "She", 1, "person_female", "c1"),
    Mention("m4", "it", 1, "object", "c2"),
    Mention("m5", "Bob", 2, "person_male", "c3"),
    Mention("m6", "the transformer model", 2, "model", "c4"),
    Mention("m7", "He", 3, "person_male", "c3"),
    Mention("m8", "the model", 3, "model", "c4"),
    Mention("m9", "Researchers", 4, "group", "c5"),
    Mention("m10", "they", 5, "group", "c5"),
]


PRONOUN_TYPES = {
    "she": "person_female",
    "he": "person_male",
    "it": "object_or_model",
    "they": "group",
}


def normalize(text: str) -> str:
    """Normalize mention text for lexical matching."""

    # 去掉冠词和大小写差异，让 “the model” 与 “model” 可以共享核心词。
    tokens = [token.lower() for token in TOKEN_RE.findall(text)]
    tokens = [token for token in tokens if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def type_compatible(left: Mention, right: Mention) -> bool:
    """Check coarse semantic compatibility between two mentions."""

    # 代词 it 可以指 object 或 model，其他代词使用更严格的类型约束。
    left_type = PRONOUN_TYPES.get(left.text.lower(), left.entity_type)
    right_type = PRONOUN_TYPES.get(right.text.lower(), right.entity_type)
    if "object_or_model" in {left_type, right_type}:
        other = right_type if left_type == "object_or_model" else left_type
        return other in {"object", "model"}
    return left_type == right_type


def mention_score(
    antecedent: Mention,
    current: Mention,
    weights: dict[str, float],
) -> float:
    """Score whether current should link to antecedent."""

    # 共指消解通常只向前寻找 antecedent，因此距离越远可信度越低。
    distance = current.sentence_id - antecedent.sentence_id
    lexical_match = normalize(antecedent.text) == normalize(current.text)
    compatible = type_compatible(antecedent, current)
    pronoun = current.text.lower() in PRONOUN_TYPES

    # 权重对应不同研究假设：词面匹配、类型兼容、代词回指、距离惩罚。
    return (
        weights["lexical"] * float(lexical_match)
        + weights["type"] * float(compatible)
        + weights["pronoun"] * float(pronoun and compatible)
        - weights["distance"] * distance
    )


def cluster_mentions(weights: dict[str, float]) -> dict[str, list[str]]:
    """Cluster mentions using greedy antecedent linking."""

    # 每个 mention 初始自成一类，后续若找到高分 antecedent 则并入其 cluster。
    mention_to_cluster: dict[str, str] = {}
    clusters: dict[str, list[str]] = {}

    for index, mention in enumerate(MENTIONS):
        best_score = -999.0
        best_antecedent: Mention | None = None

        # 只考虑当前 mention 之前的候选，符合在线共指消解的方向约束。
        for antecedent in MENTIONS[:index]:
            score = mention_score(antecedent, mention, weights)
            if score > best_score:
                best_score = score
                best_antecedent = antecedent

        # 阈值也可调；这里固定为 0.5，使链接必须有明确证据。
        if best_antecedent is not None and best_score >= 0.5:
            cluster_id = mention_to_cluster[best_antecedent.mention_id]
        else:
            cluster_id = mention.mention_id

        mention_to_cluster[mention.mention_id] = cluster_id
        clusters.setdefault(cluster_id, []).append(mention.mention_id)
    return clusters


def pair_set(clusters: dict[str, list[str]]) -> set[tuple[str, str]]:
    """Convert clusters into mention pairs for pairwise evaluation."""

    # Pairwise F1 简单直观：同一 cluster 内任意两个 mention 构成一个正例。
    pairs = set()
    for members in clusters.values():
        for left_index, left in enumerate(members):
            for right in members[left_index + 1:]:
                pairs.add(tuple(sorted((left, right))))
    return pairs


def gold_clusters() -> dict[str, list[str]]:
    """Build gold clusters from annotated cluster ids."""

    # 金标准聚类由人工 cluster id 得到，用于和模型输出比较。
    clusters: dict[str, list[str]] = {}
    for mention in MENTIONS:
        clusters.setdefault(mention.gold_cluster, []).append(mention.mention_id)
    return clusters


def evaluate(weights: dict[str, float]) -> dict[str, object]:
    """Evaluate one coreference scoring configuration."""

    # 将预测聚类和金标准聚类都转换为 pair 集合，计算 precision/recall/F1。
    predicted_clusters = cluster_mentions(weights)
    predicted_pairs = pair_set(predicted_clusters)
    target_pairs = pair_set(gold_clusters())
    true_positive = len(predicted_pairs & target_pairs)
    precision = true_positive / max(len(predicted_pairs), 1)
    recall = true_positive / max(len(target_pairs), 1)
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )

    # 错误 pair 便于分析规则是否过度链接或漏掉远距离回指。
    return {
        "weights": weights,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "predicted_clusters": predicted_clusters,
        "false_positive_pairs": sorted(predicted_pairs - target_pairs),
        "false_negative_pairs": sorted(target_pairs - predicted_pairs),
    }


def tune() -> dict[str, object]:
    """Tune the interpretable coreference scoring weights."""

    # 网格只覆盖几个合理值，目的是观察权重变化对聚类结果的影响。
    grid = {
        "lexical": [0.5, 1.0],
        "type": [0.5, 1.0, 1.5],
        "pronoun": [0.5, 1.0, 1.5],
        "distance": [0.1, 0.3],
    }
    trials = []

    # 完整保存 top trials，可以在报告中解释为什么某组参数更合理。
    for values in product(*grid.values()):
        weights = dict(zip(grid.keys(), values))
        trials.append(evaluate(weights))

    best = max(trials, key=lambda row: (row["f1"], row["precision"], row["recall"]))
    return {
        "task": "coreference_resolution_baseline",
        "num_mentions": len(MENTIONS),
        "best_result": best,
        "top_trials": sorted(
            trials,
            key=lambda row: (row["f1"], row["precision"], row["recall"]),
            reverse=True,
        )[:5],
    }


def main() -> None:
    """Run coreference tuning and persist the result."""

    # 结果落盘而不仅打印，是为了保留调参证据和错误 pair。
    result = tune()
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "coreference_baseline_results.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
