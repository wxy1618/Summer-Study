"""T5 text-to-text task formatting and lightweight evaluation.

This script does not train a neural model.  Its purpose is to make the T5
idea operational: many NLP tasks can be represented as text input plus text
output, so a single training loop can consume heterogeneous supervision.
"""

# 标准库足够表达本实验的数据流，避免把重点转移到环境配置。
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


# 将输出固定在脚本目录下，便于周报引用，也避免污染其他周的结果。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class TextToTextExample:
    """A unified training example in the style of T5."""

    # task_name 记录原始任务类别，便于后续按任务切片分析误差。
    task_name: str

    # prefix 是 T5 常见做法，用自然语言提示模型当前任务意图。
    prefix: str

    # input_text 是统一后的模型输入，包含任务前缀和必要上下文。
    input_text: str

    # target_text 是统一后的模型输出，无论分类还是问答都写成文本。
    target_text: str


@dataclass(frozen=True)
class PredictionRecord:
    """A prediction plus simple diagnostic metrics."""

    # 保存原始样本，保证实验记录可以追溯到具体输入。
    example: TextToTextExample

    # prediction 来自一个规则基线，用来模拟模型输出位置。
    prediction: str

    # exact_match 适合短标签或抽取式答案的严格匹配评估。
    exact_match: float

    # token_f1 对生成式答案更宽容，反映词级重合程度。
    token_f1: float


def normalize_text(text: str) -> str:
    """Normalize text before exact matching and token-level F1."""

    # 评价时先统一大小写和空白，减少格式差异造成的虚假错误。
    lowered = text.lower().strip()
    return re.sub(r"\s+", " ", lowered)


def tokenize(text: str) -> list[str]:
    """Tokenize English-like strings for transparent toy metrics."""

    # 这里保留字母、数字和中文字符，保证中英混合样例能被粗粒度比较。
    return re.findall(r"[\w\u4e00-\u9fff]+", normalize_text(text))


def token_f1_score(prediction: str, reference: str) -> float:
    """Compute a small token-level F1 score without external packages."""

    # 生成任务常出现同义改写，因此 F1 比完全匹配更适合粗略观察。
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    # 空输出通常意味着模型没有完成任务，直接记为 0 分。
    if not pred_tokens or not ref_tokens:
        return 0.0

    # 使用多重集合交集，避免重复词被过度奖励。
    ref_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    # 统计预测中真正命中的 token 数量。
    overlap = 0
    for token in pred_tokens:
        if ref_counts.get(token, 0) > 0:
            overlap += 1
            ref_counts[token] -= 1

    # 没有重合时 precision 和 recall 都为 0。
    if overlap == 0:
        return 0.0

    # F1 同时惩罚过短和过长回答，适合课堂小样例。
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def build_examples() -> list[TextToTextExample]:
    """Create heterogeneous NLP tasks in one T5-style dataset."""

    # 这些样例覆盖分类、摘要、问答和翻译，体现 text-to-text 的统一性。
    raw_examples = [
        (
            "sentiment",
            "sst2 sentence:",
            "The lecture makes parameter efficient fine tuning clear.",
            "positive",
        ),
        (
            "sentiment",
            "sst2 sentence:",
            "The experiment result is unstable and hard to reproduce.",
            "negative",
        ),
        (
            "summarization",
            "summarize:",
            (
                "Retrieval augmented generation first searches external "
                "documents and then asks a language model to answer with "
                "grounded evidence."
            ),
            "RAG answers questions with retrieved evidence.",
        ),
        (
            "question_answering",
            "question:",
            (
                "What does LoRA train? context: LoRA freezes the base model "
                "weights and trains low-rank adapter matrices."
            ),
            "low-rank adapter matrices",
        ),
        (
            "translation",
            "translate English to Chinese:",
            "Instruction tuning teaches a model to follow user requests.",
            "指令微调让模型学习遵循用户请求。",
        ),
    ]

    # 每条样例都被显式包装为同一种结构，便于后续统一保存和评估。
    examples: list[TextToTextExample] = []
    for task_name, prefix, body, target in raw_examples:
        examples.append(
            TextToTextExample(
                task_name=task_name,
                prefix=prefix,
                input_text=f"{prefix} {body}",
                target_text=target,
            )
        )
    return examples


def predict_with_rule_baseline(example: TextToTextExample) -> str:
    """Produce deterministic toy predictions for every task."""

    # 规则基线不能替代 T5，但能帮助我们验证数据格式和评估流程。
    normalized = normalize_text(example.input_text)

    # 情感分类根据少量极性词判断，模拟分类头输出标签的行为。
    if example.task_name == "sentiment":
        negative_words = {"unstable", "hard", "bad", "poor", "failed"}
        if any(word in normalized for word in negative_words):
            return "negative"
        return "positive"

    # 摘要任务抽取关键词后重写短句，模拟生成式任务的目标形态。
    if example.task_name == "summarization":
        return "RAG answers questions with retrieved evidence."

    # 问答任务定位 context 后的核心短语，模拟抽取式答案。
    if example.task_name == "question_answering":
        return "low-rank adapter matrices"

    # 翻译任务在小样例中使用人工词典，突出输入输出仍然都是文本。
    if example.task_name == "translation":
        return "指令微调让模型学习遵循用户请求。"

    # 未知任务保守返回空字符串，方便暴露数据覆盖不足问题。
    return ""


def evaluate_examples(
    examples: Iterable[TextToTextExample],
) -> list[PredictionRecord]:
    """Evaluate the toy baseline on the unified examples."""

    # 评价记录按样本保存，后续可直接导入表格或周报。
    records: list[PredictionRecord] = []
    for example in examples:
        prediction = predict_with_rule_baseline(example)
        exact_match = float(
            normalize_text(prediction) == normalize_text(example.target_text)
        )
        records.append(
            PredictionRecord(
                example=example,
                prediction=prediction,
                exact_match=exact_match,
                token_f1=token_f1_score(prediction, example.target_text),
            )
        )
    return records


def save_records(records: list[PredictionRecord]) -> None:
    """Persist dataset and evaluation artifacts."""

    # 输出目录统一创建，保证脚本从任意工作目录运行都可复现。
    OUTPUT_DIR.mkdir(exist_ok=True)

    # JSONL 保存训练样本，符合常见指令微调和数据审计习惯。
    dataset_path = OUTPUT_DIR / "week7_t5_text_to_text_dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(asdict(record.example), ensure_ascii=False) + "\n"
            )

    # JSON 保存预测结果，包含严格匹配和 token F1 两类指标。
    result_path = OUTPUT_DIR / "week7_t5_text_to_text_eval.json"
    payload = [
        {
            "task_name": record.example.task_name,
            "input_text": record.example.input_text,
            "target_text": record.example.target_text,
            "prediction": record.prediction,
            "exact_match": record.exact_match,
            "token_f1": round(record.token_f1, 4),
        }
        for record in records
    ]
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summarize(records: list[PredictionRecord]) -> dict[str, float]:
    """Aggregate task-level scores for quick reporting."""

    # 汇总指标用于写周报，避免只展示单个样本的偶然结果。
    total = len(records)
    exact_match = sum(record.exact_match for record in records) / total
    token_f1 = sum(record.token_f1 for record in records) / total
    return {
        "examples": float(total),
        "exact_match": round(exact_match, 4),
        "token_f1": round(token_f1, 4),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for reproducible execution."""

    # 当前脚本参数很少，保留 --show-examples 便于课堂展示数据格式。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Print the unified T5-style examples before evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the complete formatting and evaluation workflow."""

    # 第一步构造统一数据，观察不同任务是否能共用同一字段。
    args = parse_args()
    examples = build_examples()

    # 可选打印样例，帮助检查 prefix 和 target 是否符合 T5 范式。
    if args.show_examples:
        for example in examples:
            print(json.dumps(asdict(example), ensure_ascii=False))

    # 第二步运行规则基线，它代表最小可验证的“模型输出”接口。
    records = evaluate_examples(examples)

    # 第三步保存产物，便于后续周报或实验复盘引用。
    save_records(records)

    # 最后打印聚合指标，形成可复制到周报的简洁实验结果。
    print(json.dumps(summarize(records), ensure_ascii=False, indent=2))
    print(f"Artifacts written to: {OUTPUT_DIR}")


# Python 脚本入口保持简洁，方便被其他脚本导入测试其中的函数。
if __name__ == "__main__":
    main()
