# -*- coding: utf-8 -*-
"""Offline prompt evaluation framework for large language model tasks.

本脚本对应第六周“大模型”资料学习。它不调用任何外部 API，而是建立一个
可复用的 prompt 评估记录框架，用于后续接入真实大模型输出。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptCase:
    """One evaluation case for an LLM-style NLP task."""

    # task 标明任务类型，prompt 是模型输入，reference 是人工期望输出要点。
    task: str
    prompt: str
    reference_keywords: tuple[str, ...]
    simulated_output: str


# 这里使用模拟输出而非真实 API，是为了满足“不配置环境、不调用服务”的要求。
CASES = [
    PromptCase(
        "translation",
        "Translate into Chinese: question answering requires evidence.",
        ("问答", "证据"),
        "问答任务需要证据。",
    ),
    PromptCase(
        "question_answering",
        "Context: Alice wrote a paper. She submitted it. Q: What did Alice write?",
        ("paper",),
        "Alice wrote a paper.",
    ),
    PromptCase(
        "natural_language_generation",
        "Write one sentence about GloVe embeddings.",
        ("cooccurrence", "word"),
        "GloVe learns word vectors from cooccurrence statistics.",
    ),
    PromptCase(
        "coreference",
        "Resolve: Bob built a model. He evaluated it.",
        ("Bob=He", "model=it"),
        "Bob and He refer to the same person; model and it refer to the same object.",
    ),
    PromptCase(
        "relation_extraction",
        "Extract relation: Marie Curie discovered radium.",
        ("discover", "Marie Curie", "radium"),
        "discoverer(Marie Curie, radium)",
    ),
]


def keyword_score(output: str, keywords: tuple[str, ...]) -> dict[str, object]:
    """Score an output by reference keyword coverage."""

    # 关键词覆盖不是最终人工评价，但能作为离线自动检查的最低门槛。
    normalized = output.lower()
    hits = [keyword for keyword in keywords if keyword.lower() in normalized]
    return {
        "covered": hits,
        "missing": [keyword for keyword in keywords if keyword not in hits],
        "coverage": round(len(hits) / max(len(keywords), 1), 4),
    }


def classify_failure(score: dict[str, object]) -> str:
    """Assign a coarse failure type for later error analysis."""

    # 错误类型标签帮助我们区分“完全答非所问”和“部分缺少关键事实”。
    coverage = float(score["coverage"])
    if coverage == 1.0:
        return "pass"
    if coverage >= 0.5:
        return "partial_fact_missing"
    return "major_mismatch"


def evaluate_cases() -> dict[str, object]:
    """Evaluate all offline prompt cases."""

    # 每条样例保存 prompt、输出、得分和失败类型，方便后续替换真实模型输出。
    rows = []
    for case in CASES:
        score = keyword_score(case.simulated_output, case.reference_keywords)
        rows.append(
            {
                "task": case.task,
                "prompt": case.prompt,
                "output": case.simulated_output,
                "reference_keywords": list(case.reference_keywords),
                "score": score,
                "failure_type": classify_failure(score),
            }
        )

    # 汇总指标用于快速比较不同 prompt 模板或不同模型版本。
    average_coverage = sum(row["score"]["coverage"] for row in rows) / len(rows)
    pass_rate = sum(row["failure_type"] == "pass" for row in rows) / len(rows)
    return {
        "task": "offline_llm_prompt_evaluation",
        "num_cases": len(rows),
        "average_keyword_coverage": round(average_coverage, 4),
        "pass_rate": round(pass_rate, 4),
        "cases": rows,
    }


def main() -> None:
    """Run offline LLM prompt evaluation and save JSON results."""

    # 这个脚本的价值在于框架而不是模型调用，后续只需替换 simulated_output。
    result = evaluate_cases()
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "llm_prompt_evaluation_offline.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
