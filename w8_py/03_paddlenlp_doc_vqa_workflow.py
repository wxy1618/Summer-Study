"""Dry-run document VQA workflow for PaddleNLP-style document intelligence.

The script uses structured OCR tokens and layout boxes to simulate a document
question-answering pipeline.  It records the evidence tokens and answer spans
that a real PaddleNLP DocVQA experiment should inspect after prediction.
"""

# 默认不调用外部模型，保证作业可以在未配置环境时完成。
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


# 统一输出位置，便于阶段四结果汇总。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class OcrToken:
    """An OCR token with page-layout information."""

    # token_id 用于从答案追溯到文档证据。
    token_id: str

    # text 是 OCR 识别出的文字片段。
    text: str

    # bbox 使用 [x1, y1, x2, y2] 表示版面位置。
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class DocVqaQuestion:
    """A document question with an expected answer."""

    # question_id 让每个问答样例可独立记录。
    question_id: str

    # question 是用户针对文档提出的问题。
    question: str

    # expected_answer 用于计算 dry-run 是否命中。
    expected_answer: str


@dataclass(frozen=True)
class DocVqaPrediction:
    """A dry-run answer with evidence metadata."""

    # question_id 对应输入问题。
    question_id: str

    # answer 是规则系统返回的答案。
    answer: str

    # evidence_token_ids 保存支持答案的 OCR token。
    evidence_token_ids: list[str]

    # evidence_bbox 是证据 token 的合并区域。
    evidence_bbox: tuple[int, int, int, int] | None

    # confidence 是可解释的启发式置信度。
    confidence: float

    # correct 表示答案是否与人工期望一致。
    correct: bool


def build_ocr_tokens() -> list[OcrToken]:
    """Create a small synthetic invoice-like document."""

    # DocVQA 的关键在于文本与版面共同构成证据，而不只是普通字符串。
    return [
        OcrToken("t01", "Invoice", (40, 30, 120, 55)),
        OcrToken("t02", "No.", (40, 75, 70, 95)),
        OcrToken("t03", "A-2026-08", (85, 75, 170, 95)),
        OcrToken("t04", "Vendor", (40, 120, 90, 140)),
        OcrToken("t05", "PaddleNLP", (120, 120, 215, 140)),
        OcrToken("t06", "Due", (40, 165, 70, 185)),
        OcrToken("t07", "Date", (75, 165, 115, 185)),
        OcrToken("t08", "2026-08-31", (130, 165, 225, 185)),
        OcrToken("t09", "Total", (40, 220, 90, 240)),
        OcrToken("t10", "1280", (130, 220, 175, 240)),
        OcrToken("t11", "CNY", (180, 220, 215, 240)),
    ]


def build_questions() -> list[DocVqaQuestion]:
    """Create document questions with gold answers."""

    # 问题覆盖编号、供应方、日期和金额，模拟常见票据问答场景。
    return [
        DocVqaQuestion("q_invoice_id", "What is the invoice number?", "A-2026-08"),
        DocVqaQuestion("q_vendor", "Who is the vendor?", "PaddleNLP"),
        DocVqaQuestion("q_due_date", "What is the due date?", "2026-08-31"),
        DocVqaQuestion("q_total", "What is the total amount?", "1280 CNY"),
    ]


def merge_bboxes(tokens: list[OcrToken]) -> tuple[int, int, int, int] | None:
    """Merge bounding boxes for evidence tokens."""

    # 证据区域能帮助检查模型是否看到了正确版面位置。
    if not tokens:
        return None
    x1 = min(token.bbox[0] for token in tokens)
    y1 = min(token.bbox[1] for token in tokens)
    x2 = max(token.bbox[2] for token in tokens)
    y2 = max(token.bbox[3] for token in tokens)
    return (x1, y1, x2, y2)


def answer_question(
    question: DocVqaQuestion,
    tokens: list[OcrToken],
) -> DocVqaPrediction:
    """Answer a document question using transparent rules."""

    # 规则基线用于模拟 DocVQA 输出，并暴露证据定位逻辑。
    text_by_id = {token.token_id: token.text for token in tokens}
    token_by_id = {token.token_id: token for token in tokens}
    normalized = question.question.lower()

    # 不同问题类型对应不同证据 token，真实模型会用多模态表示学习。
    evidence_ids: list[str]
    if "invoice number" in normalized:
        evidence_ids = ["t03"]
    elif "vendor" in normalized:
        evidence_ids = ["t05"]
    elif "due date" in normalized:
        evidence_ids = ["t08"]
    elif "total amount" in normalized:
        evidence_ids = ["t10", "t11"]
    else:
        evidence_ids = []

    # 答案由证据 token 拼接而来，便于从 answer 反查到 OCR 片段。
    answer = " ".join(text_by_id[token_id] for token_id in evidence_ids)
    evidence_tokens = [token_by_id[token_id] for token_id in evidence_ids]
    confidence = 0.92 if evidence_ids else 0.0
    return DocVqaPrediction(
        question_id=question.question_id,
        answer=answer or "无法根据当前 OCR 证据确定答案",
        evidence_token_ids=evidence_ids,
        evidence_bbox=merge_bboxes(evidence_tokens),
        confidence=confidence,
        correct=answer == question.expected_answer,
    )


def build_taskflow_plan() -> dict[str, object]:
    """Describe how this dry-run maps to PaddleNLP Taskflow."""

    # 计划记录真实环境中应替换的组件，但默认不实例化模型。
    return {
        "paddlenlp_mapping": [
            "Use OCR or document_intelligence pipeline to obtain tokens.",
            "Keep token text, bounding boxes, and page identifiers.",
            "Use DocVQA or document intelligence model to score answer spans.",
            "Store answer text, evidence region, confidence, and error cases.",
        ],
        "environment_policy": (
            "This script does not install PaddlePaddle or PaddleNLP. "
            "Real Taskflow calls should be run only in an existing environment."
        ),
    }


def run_optional_taskflow_check() -> dict[str, object]:
    """Check optional PaddleNLP availability without downloading models."""

    # 这里只检查 import 能否成功，不调用 Taskflow 构造器，避免下载模型。
    status = {"paddlenlp_available": False}
    try:
        import paddlenlp  # type: ignore

        status["paddlenlp_available"] = True
        status["paddlenlp_version"] = getattr(paddlenlp, "__version__", "unknown")
    except ImportError as error:
        status["paddlenlp_error"] = str(error)
    return status


def run_workflow() -> dict[str, object]:
    """Run the document VQA dry-run workflow."""

    # 构造 OCR token、问题、预测和指标，形成最小可复现实验闭环。
    tokens = build_ocr_tokens()
    questions = build_questions()
    predictions = [answer_question(question, tokens) for question in questions]

    # accuracy 用于快速判断规则基线是否正确定位证据。
    correct_count = sum(prediction.correct for prediction in predictions)
    accuracy = correct_count / len(predictions)
    return {
        "mode": "standard_library_doc_vqa_dryrun",
        "ocr_tokens": [asdict(token) for token in tokens],
        "questions": [asdict(question) for question in questions],
        "predictions": [asdict(prediction) for prediction in predictions],
        "metrics": {"accuracy": round(accuracy, 4)},
        "taskflow_plan": build_taskflow_plan(),
        "paddlenlp_environment": run_optional_taskflow_check(),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    # 预留参数用于课堂演示；当前脚本默认只做 dry-run。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-ocr-text",
        action="store_true",
        help="Print reconstructed OCR text before saving results.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the DocVQA dry-run and save artifacts."""

    # 运行 dry-run 时不会访问外部文件，也不会调用网络模型。
    args = parse_args()
    result = run_workflow()

    # 可选打印 OCR 重建文本，帮助人工检查文档内容。
    if args.show_ocr_text:
        joined = " ".join(token["text"] for token in result["ocr_tokens"])
        print(re.sub(r"\s+", " ", joined))

    # 保存完整预测、证据和平台映射计划。
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "week8_doc_vqa_workflow.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 命令行输出指标摘要，方便周报直接引用。
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Artifact written to: {output_path}")


# 主入口保持无副作用导入，便于其他脚本聚合结果。
if __name__ == "__main__":
    main()
