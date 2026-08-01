"""Build extractive question-answering features with token offsets.

This script mirrors the key preprocessing idea behind HuggingFace QA tasks:
the answer is supervised by start and end positions after tokenization.  It is
implemented with the standard library to make alignment details visible.
"""

# 不依赖 transformers，先把 span 对齐逻辑拆出来独立观察。
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


# 输出位置固定，便于和第七周其他实验一起归档。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class QaExample:
    """A raw extractive QA example."""

    # question 是用户问题，在模型输入中通常放在第一段。
    question: str

    # context 是答案所在文本，在模型输入中通常放在第二段。
    context: str

    # answer_text 是标准答案文本，应能在 context 中定位。
    answer_text: str


@dataclass(frozen=True)
class Token:
    """A token with character offsets."""

    # text 是 token 字符串，用于调试和人工检查。
    text: str

    # start 是 token 在原字符串中的起始字符位置。
    start: int

    # end 是 token 在原字符串中的结束字符位置，采用左闭右开。
    end: int


@dataclass(frozen=True)
class QaFeature:
    """A model-like QA feature with span labels."""

    # feature_id 标识滑动窗口生成的具体特征。
    feature_id: str

    # question 保存原始问题，便于错误分析时阅读。
    question: str

    # context_window 保存当前窗口内的上下文文本。
    context_window: str

    # tokens 是问题和上下文拼接后的简化 token 序列。
    tokens: list[str]

    # start_position 和 end_position 是答案 token 在序列中的位置。
    start_position: int

    # end_position 与 start_position 共同定义抽取式答案 span。
    end_position: int

    # answer_in_window 表示当前窗口是否覆盖标准答案。
    answer_in_window: bool


def build_examples() -> list[QaExample]:
    """Create small QA samples around week-seven topics."""

    # 样例刻意覆盖 RAG、LoRA 和指令微调三个核心概念。
    return [
        QaExample(
            question="What does LoRA train?",
            context=(
                "LoRA freezes the base model weights and trains low-rank "
                "adapter matrices for task adaptation."
            ),
            answer_text="low-rank adapter matrices",
        ),
        QaExample(
            question="When does RAG add external knowledge?",
            context=(
                "RAG retrieves external documents at inference time before "
                "the language model generates an answer."
            ),
            answer_text="at inference time",
        ),
        QaExample(
            question="What does instruction tuning teach?",
            context=(
                "Instruction tuning teaches a pretrained model to follow "
                "user requests and produce task-oriented answers."
            ),
            answer_text="follow user requests",
        ),
    ]


def tokenize_with_offsets(text: str) -> list[Token]:
    """Tokenize text while preserving character offsets."""

    # offset 是 QA 预处理的关键，因为标签首先存在于字符空间。
    tokens: list[Token] = []
    for match in re.finditer(r"[A-Za-z0-9_-]+|[\u4e00-\u9fff]", text):
        tokens.append(Token(text=match.group(0), start=match.start(), end=match.end()))
    return tokens


def find_answer_span(context: str, answer_text: str) -> tuple[int, int]:
    """Find the character span of the gold answer."""

    # 课堂样例要求答案唯一出现；真实数据中还要处理多答案和噪声。
    start = context.index(answer_text)
    end = start + len(answer_text)
    return start, end


def token_span_from_char_span(
    tokens: list[Token],
    answer_start: int,
    answer_end: int,
) -> tuple[int, int] | None:
    """Map a character answer span to token positions."""

    # token 起止要完全覆盖答案字符，不能只依靠字符串再次搜索。
    covered = [
        index
        for index, token in enumerate(tokens)
        if token.start < answer_end and token.end > answer_start
    ]
    if not covered:
        return None
    return covered[0], covered[-1]


def make_context_windows(
    tokens: list[Token],
    max_context_tokens: int,
    stride: int,
) -> list[tuple[int, list[Token]]]:
    """Create sliding windows over context tokens."""

    # 长文档超过模型长度时必须滑动切分，stride 保留跨窗口上下文。
    windows: list[tuple[int, list[Token]]] = []
    start = 0
    while start < len(tokens):
        window = tokens[start:start + max_context_tokens]
        windows.append((start, window))
        if start + max_context_tokens >= len(tokens):
            break
        start += stride
    return windows


def build_features(
    example: QaExample,
    example_index: int,
    max_context_tokens: int,
    stride: int,
) -> list[QaFeature]:
    """Build QA features for one raw example."""

    # 问题 token 在拼接序列前部，答案标签只允许落在 context 部分。
    question_tokens = tokenize_with_offsets(example.question)
    context_tokens = tokenize_with_offsets(example.context)
    answer_start, answer_end = find_answer_span(example.context, example.answer_text)

    # 窗口化后，每个窗口都独立判断是否覆盖标准答案。
    features: list[QaFeature] = []
    for window_id, (window_start, window_tokens) in enumerate(
        make_context_windows(context_tokens, max_context_tokens, stride)
    ):
        span = token_span_from_char_span(window_tokens, answer_start, answer_end)
        answer_in_window = span is not None

        # 拼接时用特殊标记模拟 BERT 类模型的输入格式。
        tokens = (
            ["[CLS]"]
            + [token.text for token in question_tokens]
            + ["[SEP]"]
            + [token.text for token in window_tokens]
            + ["[SEP]"]
        )

        # context 在拼接序列中的偏移需要加上 [CLS]、问题和 [SEP]。
        context_offset = 1 + len(question_tokens) + 1
        if span is None:
            start_position = 0
            end_position = 0
        else:
            start_position = context_offset + span[0]
            end_position = context_offset + span[1]

        # 保存窗口文本，帮助人工理解模型到底看到了哪段上下文。
        context_window = " ".join(token.text for token in window_tokens)
        features.append(
            QaFeature(
                feature_id=f"ex{example_index}_win{window_id}_start{window_start}",
                question=example.question,
                context_window=context_window,
                tokens=tokens,
                start_position=start_position,
                end_position=end_position,
                answer_in_window=answer_in_window,
            )
        )
    return features


def parse_args() -> argparse.Namespace:
    """Parse preprocessing parameters."""

    # max_context_tokens 和 stride 用于观察滑动窗口对答案覆盖的影响。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-context-tokens", type=int, default=12)
    parser.add_argument("--stride", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    """Build and persist QA features."""

    # 构造全部特征后保存为 JSON，便于检查 span 标签是否正确。
    args = parse_args()
    all_features: list[QaFeature] = []
    for index, example in enumerate(build_examples()):
        all_features.extend(
            build_features(
                example,
                example_index=index,
                max_context_tokens=args.max_context_tokens,
                stride=args.stride,
            )
        )

    # 输出中保留 token 序列和位置标签，正是 QA 微调最易出错的部分。
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "week7_qa_features.json"
    output_path.write_text(
        json.dumps(
            [asdict(item) for item in all_features],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 打印覆盖统计，确认每个原始样本至少有一个窗口包含答案。
    answer_windows = sum(feature.answer_in_window for feature in all_features)
    print(
        json.dumps(
            {
                "features": len(all_features),
                "answer_windows": answer_windows,
                "output_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


# 使用标准入口，避免导入时自动写文件。
if __name__ == "__main__":
    main()
