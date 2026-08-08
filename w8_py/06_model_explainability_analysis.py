"""Lightweight model analysis and explanation experiment.

This script analyzes a transparent sentiment baseline with word attribution,
deletion tests, and counterfactual replacements.  It is intentionally small so
that the explanation process itself is visible and auditable.
"""

# 解释性实验使用标准库实现，避免把关键逻辑隐藏在复杂框架中。
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


# 输出目录与阶段四其他实验统一。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class ExplanationExample:
    """An example to analyze."""

    # text 是待解释输入。
    text: str

    # gold_label 供错误分析使用。
    gold_label: str


@dataclass(frozen=True)
class ExplanationRecord:
    """A prediction and its diagnostic explanation."""

    # text 保存原始样本。
    text: str

    # prediction 是基线模型输出。
    prediction: str

    # score 是正负词典得分差。
    score: int

    # token_contributions 记录每个触发词的贡献方向。
    token_contributions: dict[str, int]

    # deletion_effects 记录删除某个触发词后分数如何变化。
    deletion_effects: dict[str, int]

    # counterfactual_text 是替换关键词后的样本。
    counterfactual_text: str

    # counterfactual_prediction 用来观察最小修改是否改变输出。
    counterfactual_prediction: str


def build_examples() -> list[ExplanationExample]:
    """Create examples for explanation analysis."""

    # 样例与第八周情感分析任务保持一致，便于跨脚本比较。
    return [
        ExplanationExample(
            text="The semantic search demo is clear and useful.",
            gold_label="positive",
        ),
        ExplanationExample(
            text="The retrieval result is confusing and poorly ranked.",
            gold_label="negative",
        ),
        ExplanationExample(
            text="The ethical audit is helpful but the model ignores privacy.",
            gold_label="negative",
        ),
    ]


def lexicon() -> tuple[set[str], set[str]]:
    """Return the lexicon used by the transparent baseline."""

    # 词典与情感分析 dry-run 保持相近，支持可解释性复现。
    positive = {"clear", "useful", "reliable", "helpful", "concrete"}
    negative = {"confusing", "poorly", "ignores", "unstable", "fails", "privacy"}
    return positive, negative


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase lexical units."""

    # 稳定 token 化是删除实验和反事实实验的基础。
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())


def score_tokens(tokens: list[str]) -> tuple[int, dict[str, int]]:
    """Score tokens and return contribution dictionary."""

    # 贡献字典把每个触发词映射到 +1 或 -1，体现模型决策依据。
    positive, negative = lexicon()
    contributions: dict[str, int] = {}
    for token in tokens:
        if token in positive:
            contributions[token] = contributions.get(token, 0) + 1
        elif token in negative:
            contributions[token] = contributions.get(token, 0) - 1
    return sum(contributions.values()), contributions


def predict(text: str) -> tuple[str, int, dict[str, int]]:
    """Predict a label and expose lexical contributions."""

    # 基线模型把得分非负判断为 positive，这一规则完全透明。
    tokens = tokenize(text)
    score, contributions = score_tokens(tokens)
    label = "positive" if score >= 0 else "negative"
    return label, score, contributions


def deletion_test(text: str, contributions: dict[str, int]) -> dict[str, int]:
    """Measure score changes after deleting evidence tokens."""

    # 删除实验比单纯列关键词更接近“该词是否真的影响输出”的问题。
    original_tokens = tokenize(text)
    _, original_score, _ = predict(text)
    effects: dict[str, int] = {}
    for token in contributions:
        reduced_tokens = [item for item in original_tokens if item != token]
        _, reduced_score, _ = predict(" ".join(reduced_tokens))
        effects[token] = reduced_score - original_score
    return effects


def build_counterfactual(text: str) -> str:
    """Construct a simple counterfactual by replacing sentiment words."""

    # 反事实样本检验模型是否对关键属性变化作出合理响应。
    replacements = {
        "clear": "confusing",
        "useful": "poorly",
        "helpful": "ignores",
        "confusing": "clear",
        "poorly": "useful",
        "ignores": "helpful",
    }
    tokens = tokenize(text)
    changed = [replacements.get(token, token) for token in tokens]
    return " ".join(changed)


def analyze_example(example: ExplanationExample) -> ExplanationRecord:
    """Analyze one example with several explanation tools."""

    # 每个样本保存预测、贡献、删除实验和反事实结果。
    prediction, score, contributions = predict(example.text)
    deletion_effects = deletion_test(example.text, contributions)
    counterfactual_text = build_counterfactual(example.text)
    counterfactual_prediction, _, _ = predict(counterfactual_text)
    return ExplanationRecord(
        text=example.text,
        prediction=prediction,
        score=score,
        token_contributions=contributions,
        deletion_effects=deletion_effects,
        counterfactual_text=counterfactual_text,
        counterfactual_prediction=counterfactual_prediction,
    )


def write_markdown(records: list[ExplanationRecord], path: Path) -> None:
    """Write a readable explanation report."""

    # 报告强调解释是诊断工具，不是对模型因果机制的最终证明。
    lines = [
        "# Model Explanation Analysis",
        "",
        "说明：本报告使用透明词典基线展示解释流程。对于深度模型，"
        "attention 或 saliency 只能作为诊断线索，仍需 deletion、"
        "counterfactual 等忠实性检查。",
        "",
        "| Text | Prediction | Score | Contributions | Counterfactual |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for record in records:
        lines.append(
            "| "
            f"{record.text} | {record.prediction} | {record.score} | "
            f"{record.token_contributions} | "
            f"{record.counterfactual_prediction}: {record.counterfactual_text} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the explanation analysis and save artifacts."""

    # 分析每个样本并保存结构化记录，方便后续复盘错误。
    examples = build_examples()
    records = [analyze_example(example) for example in examples]

    # JSON 保存全部细节，Markdown 提供人类可读总结。
    OUTPUT_DIR.mkdir(exist_ok=True)
    json_path = OUTPUT_DIR / "week8_model_explainability_analysis.json"
    json_path.write_text(
        json.dumps([asdict(item) for item in records], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(records, OUTPUT_DIR / "week8_model_explainability_analysis.md")

    # 打印摘要，用于命令行检查反事实是否产生变化。
    changed = sum(
        item.prediction != item.counterfactual_prediction for item in records
    )
    print(
        json.dumps(
            {"examples": len(records), "counterfactual_changed": changed},
            ensure_ascii=False,
            indent=2,
        )
    )


# 主入口保持单一职责，便于独立运行和阶段汇总。
if __name__ == "__main__":
    main()
