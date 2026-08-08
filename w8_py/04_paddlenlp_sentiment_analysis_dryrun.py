"""Dry-run sentiment analysis workflow for PaddleNLP fine-tuning practice.

The default implementation uses a transparent lexicon baseline.  It mirrors a
pretrained-model fine-tuning report by saving data, predictions, metrics, and a
PaddleNLP replacement plan without configuring the environment.
"""

# 标准库基线确保脚本可运行，PaddleNLP 部分只记录可替换接口。
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


# 输出目录与其他第八周作业脚本保持一致。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class SentimentExample:
    """A labeled sentiment example."""

    # text 是待分类文本。
    text: str

    # label 采用二分类标签，便于计算 precision/recall/F1。
    label: str


@dataclass(frozen=True)
class SentimentPrediction:
    """A prediction with explanation fields."""

    # text 保存原始输入，便于错误分析。
    text: str

    # gold_label 是人工标签。
    gold_label: str

    # predicted_label 是词典基线输出。
    predicted_label: str

    # score 是 positive 分数减 negative 分数后的差值。
    score: int

    # evidence_words 记录触发判断的词，服务于解释性分析。
    evidence_words: list[str]

    # correct 表示预测是否命中标签。
    correct: bool


def build_dataset() -> list[SentimentExample]:
    """Create a compact dataset around course-learning comments."""

    # 样例使用课程学习场景，便于和 PaddleNLP 情感分析任务衔接。
    return [
        SentimentExample(
            "The semantic search demo is clear and useful.",
            "positive",
        ),
        SentimentExample(
            "The DocVQA pipeline feels reliable after evidence checks.",
            "positive",
        ),
        SentimentExample(
            "The sentiment model is unstable on short comments.",
            "negative",
        ),
        SentimentExample(
            "The explanation report is helpful for error analysis.",
            "positive",
        ),
        SentimentExample(
            "The retrieval result is confusing and poorly ranked.",
            "negative",
        ),
        SentimentExample(
            "The ethical audit improves the deployment plan.",
            "positive",
        ),
        SentimentExample(
            "The model ignores privacy risks and fails the review.",
            "negative",
        ),
        SentimentExample(
            "The PaddleNLP practice makes fine-tuning steps concrete.",
            "positive",
        ),
    ]


def sentiment_lexicon() -> tuple[set[str], set[str]]:
    """Return positive and negative lexicons for the baseline."""

    # 词典基线简单但透明，适合在正式微调前建立可解释下限。
    positive = {
        "clear",
        "useful",
        "reliable",
        "helpful",
        "improves",
        "concrete",
    }
    negative = {
        "unstable",
        "confusing",
        "poorly",
        "ignores",
        "fails",
        "risk",
        "risks",
    }
    return positive, negative


def tokenize(text: str) -> list[str]:
    """Tokenize an English sentence for the baseline."""

    # 与真实 tokenizer 不同，词典基线只需要稳定的小写词。
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())


def predict(example: SentimentExample) -> SentimentPrediction:
    """Predict sentiment with a transparent lexicon baseline."""

    # 词典得分让我们能记录每个预测的可解释证据。
    positive, negative = sentiment_lexicon()
    tokens = tokenize(example.text)
    positive_hits = [token for token in tokens if token in positive]
    negative_hits = [token for token in tokens if token in negative]

    # 正负分差是最小可解释分类器的决策边界。
    score = len(positive_hits) - len(negative_hits)
    predicted_label = "positive" if score >= 0 else "negative"
    evidence_words = positive_hits + negative_hits
    return SentimentPrediction(
        text=example.text,
        gold_label=example.label,
        predicted_label=predicted_label,
        score=score,
        evidence_words=evidence_words,
        correct=predicted_label == example.label,
    )


def compute_binary_metrics(
    predictions: list[SentimentPrediction],
) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1 for positive class."""

    # 情感分析微调报告应至少包含混淆矩阵相关指标。
    true_positive = sum(
        item.gold_label == "positive" and item.predicted_label == "positive"
        for item in predictions
    )
    false_positive = sum(
        item.gold_label == "negative" and item.predicted_label == "positive"
        for item in predictions
    )
    false_negative = sum(
        item.gold_label == "positive" and item.predicted_label == "negative"
        for item in predictions
    )
    correct = sum(item.correct for item in predictions)

    # 除零保护让小样本边界情况也能稳定运行。
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    accuracy = correct / len(predictions)
    return {
        "accuracy": round(accuracy, 4),
        "precision_positive": round(precision, 4),
        "recall_positive": round(recall, 4),
        "f1_positive": round(f1, 4),
    }


def build_paddlenlp_plan() -> dict[str, object]:
    """Record how to replace the baseline with PaddleNLP."""

    # 该计划说明真实微调时的关键步骤，但不会执行环境配置。
    return {
        "dataset_fields": ["text", "label"],
        "suggested_pipeline": [
            "Load PaddleNLP tokenizer for a sentiment checkpoint.",
            "Map text examples into input_ids and token_type_ids.",
            "Use a pretrained sequence classification model.",
            "Train on train split and evaluate on validation split.",
            "Record accuracy, precision, recall, F1 and error examples.",
        ],
        "environment_policy": (
            "Run this replacement only in an existing PaddlePaddle/PaddleNLP "
            "environment. This script does not install or configure packages."
        ),
    }


def check_environment() -> dict[str, object]:
    """Check optional Paddle/PaddleNLP availability."""

    # 检查 import 状态可以写入实验记录，说明当前为何采用 dry-run。
    status = {"paddle_available": False, "paddlenlp_available": False}
    try:
        import paddle  # type: ignore

        status["paddle_available"] = True
        status["paddle_version"] = getattr(paddle, "__version__", "unknown")
    except ImportError as error:
        status["paddle_error"] = str(error)

    # PaddleNLP 独立记录，避免把框架和 NLP 套件混为一谈。
    try:
        import paddlenlp  # type: ignore

        status["paddlenlp_available"] = True
        status["paddlenlp_version"] = getattr(paddlenlp, "__version__", "unknown")
    except ImportError as error:
        status["paddlenlp_error"] = str(error)
    return status


def run_experiment() -> dict[str, object]:
    """Run the sentiment dry-run experiment."""

    # 先预测再评价，形成和真实微调实验一致的记录结构。
    dataset = build_dataset()
    predictions = [predict(example) for example in dataset]
    metrics = compute_binary_metrics(predictions)
    errors = [asdict(item) for item in predictions if not item.correct]
    return {
        "mode": "standard_library_lexicon_dryrun",
        "dataset": [asdict(item) for item in dataset],
        "predictions": [asdict(item) for item in predictions],
        "metrics": metrics,
        "errors": errors,
        "paddlenlp_plan": build_paddlenlp_plan(),
        "paddlenlp_environment": check_environment(),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    # 当前参数保留给课堂演示，默认执行完整 dry-run。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-errors",
        action="store_true",
        help="Print misclassified examples after saving results.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute and save the sentiment analysis dry-run."""

    # dry-run 结果可以作为阶段四“记录实验结果”的基础材料。
    args = parse_args()
    result = run_experiment()

    # 保存完整数据、预测、指标、错误样例和 PaddleNLP 迁移计划。
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "week8_sentiment_analysis_dryrun.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 可选打印错误样例，训练调参时这比单个 accuracy 更有价值。
    if args.print_errors:
        print(json.dumps(result["errors"], ensure_ascii=False, indent=2))
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Artifact written to: {output_path}")


# 主入口保持清晰，方便其他汇总脚本导入其中的函数。
if __name__ == "__main__":
    main()
