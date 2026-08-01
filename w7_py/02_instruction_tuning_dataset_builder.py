"""Build and audit a small instruction-tuning dataset.

The script focuses on the data side of supervised fine-tuning.  It validates
instruction/input/output triples, removes duplicates, creates train/validation
splits, and records quality diagnostics for later reporting.
"""

# 只使用标准库，让数据构造流程独立于任何大模型训练环境。
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


# 所有结果都写入第七周作业目录，保证文件结构清晰可追溯。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class InstructionExample:
    """A supervised fine-tuning example."""

    # instruction 描述用户希望模型完成的任务。
    instruction: str

    # input 保存任务所需上下文；无上下文任务也显式保留空字符串。
    input: str

    # output 是目标回答，应尽量完整、具体、可验证。
    output: str

    # category 用于分层统计，避免数据集只覆盖单一能力。
    category: str


@dataclass(frozen=True)
class AuditResult:
    """Quality diagnostics for one instruction example."""

    # index 对应原始数据位置，便于回溯问题样本。
    index: int

    # category 帮助分析某类任务是否质量偏低。
    category: str

    # is_valid 表示样本是否通过基础结构检查。
    is_valid: bool

    # quality_score 是教学用启发式分数，不替代人工标注审核。
    quality_score: float

    # issues 保存可解释的问题标签，方便后续数据清洗。
    issues: list[str]


def raw_examples() -> list[InstructionExample]:
    """Return a compact but diverse instruction-tuning dataset."""

    # 样例覆盖概念解释、步骤生成、判断纠错和格式化回答。
    return [
        InstructionExample(
            instruction="解释 LoRA 为什么属于参数高效微调方法。",
            input="要求用三句话回答，并说明低秩矩阵的作用。",
            output=(
                "LoRA 冻结基座模型的大部分权重，只训练额外加入的低秩矩阵。"
                "低秩矩阵用于近似任务相关的权重更新，因此显著减少可训练参数。"
                "这种方式降低显存与存储成本，同时保留基座模型的通用能力。"
            ),
            category="concept_explanation",
        ),
        InstructionExample(
            instruction="给出一个离线 RAG 系统的核心步骤。",
            input="面向课程资料问答场景。",
            output=(
                "先收集课程文档并按语义边界切分，再为每个片段建立检索索引。"
                "用户提问后召回相关片段，把证据与问题共同放入 prompt。"
                "最后由模型基于证据回答，并记录来源与检索得分。"
            ),
            category="workflow_generation",
        ),
        InstructionExample(
            instruction="判断下面说法是否准确，并给出理由。",
            input="说法：RAG 会修改大语言模型的全部参数。",
            output=(
                "不准确。RAG 通常在推理阶段检索外部知识并拼接上下文，"
                "它不需要更新模型参数；修改参数属于微调或继续训练范畴。"
            ),
            category="error_correction",
        ),
        InstructionExample(
            instruction="把文本分类任务改写为 T5 text-to-text 格式。",
            input="原任务：判断句子 The answer is grounded 的标签。",
            output=(
                "输入可以写为：sst2 sentence: The answer is grounded。"
                "输出写为文本标签：positive。"
            ),
            category="format_conversion",
        ),
        InstructionExample(
            instruction="说明指令微调和预训练的差异。",
            input="要求从训练目标和数据形式两个角度回答。",
            output=(
                "预训练主要使用大规模无标注文本学习下一个 token 预测。"
                "指令微调使用 instruction-input-output 样本，"
                "让模型学习按照用户意图完成具体任务。"
            ),
            category="concept_comparison",
        ),
        InstructionExample(
            instruction="列出 HuggingFace 文本分类微调的主要流程。",
            input="至少包含 tokenizer、模型和评价指标。",
            output=(
                "首先整理 text 与 label 字段，然后用 tokenizer 生成模型输入。"
                "接着加载 AutoModelForSequenceClassification 并设置标签数。"
                "训练时定义 accuracy 或 F1 等指标，最后保存模型与实验记录。"
            ),
            category="workflow_generation",
        ),
    ]


def normalize_for_duplicate(text: str) -> str:
    """Normalize text for duplicate detection."""

    # 重复检测只关注语义载体，弱化标点和空白差异。
    lowered = text.lower().strip()
    collapsed = re.sub(r"\s+", "", lowered)
    return re.sub(r"[，。,.!?！？：:；;]", "", collapsed)


def audit_example(
    example: InstructionExample,
    index: int,
    seen_keys: set[str],
) -> AuditResult:
    """Audit one sample using interpretable quality heuristics."""

    # 基础结构检查确保每条样本都能被监督微调管线读取。
    issues: list[str] = []
    if not example.instruction.strip():
        issues.append("missing_instruction")
    if not example.output.strip():
        issues.append("missing_output")
    if not example.category.strip():
        issues.append("missing_category")

    # 指令过短往往导致任务意图不清，实际训练中会增加标签噪声。
    if len(example.instruction) < 8:
        issues.append("instruction_too_short")

    # 输出过短通常无法承载解释性监督，特别不适合研究型问答数据。
    if len(example.output) < 25:
        issues.append("output_too_short")

    # 重复样本会让验证集泄漏训练信息，因此在构造阶段提前标记。
    duplicate_key = normalize_for_duplicate(
        example.instruction + example.input + example.output
    )
    if duplicate_key in seen_keys:
        issues.append("duplicate_example")
    seen_keys.add(duplicate_key)

    # 质量分数是可解释启发式：问题越少、输出越具体，分数越高。
    quality_score = 1.0
    quality_score -= 0.18 * len(issues)
    if 40 <= len(example.output) <= 180:
        quality_score += 0.08
    if "。" in example.output or "." in example.output:
        quality_score += 0.04
    quality_score = max(0.0, min(1.0, quality_score))

    # 结构有效性和质量分数分开记录，因为有效样本也可能质量一般。
    return AuditResult(
        index=index,
        category=example.category,
        is_valid=not any(issue.startswith("missing") for issue in issues),
        quality_score=round(quality_score, 3),
        issues=issues,
    )


def audit_dataset(
    examples: Iterable[InstructionExample],
) -> list[AuditResult]:
    """Audit the full dataset and preserve per-example diagnostics."""

    # seen_keys 在全局范围维护，确保重复检测不局限于单一类别。
    seen_keys: set[str] = set()
    results: list[AuditResult] = []
    for index, example in enumerate(examples):
        results.append(audit_example(example, index, seen_keys))
    return results


def split_dataset(
    examples: list[InstructionExample],
    seed: int,
    validation_ratio: float,
) -> tuple[list[InstructionExample], list[InstructionExample]]:
    """Create a deterministic train/validation split."""

    # 固定随机种子是实验复现的基本要求，尤其适合周报记录。
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    # 小样本至少保留一个验证样本，以便检查泛化和格式问题。
    validation_size = max(1, round(len(shuffled) * validation_ratio))
    validation = shuffled[:validation_size]
    train = shuffled[validation_size:]
    return train, validation


def write_jsonl(path: Path, rows: Iterable[InstructionExample]) -> None:
    """Write instruction examples in JSONL format."""

    # JSONL 是微调数据常见格式，支持按行流式读取和人工抽检。
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def build_summary(audits: list[AuditResult]) -> dict[str, object]:
    """Build a compact dataset-level summary."""

    # 汇总字段服务于周报，不需要打开全部 JSONL 也能看到质量概况。
    category_counts: dict[str, int] = {}
    for audit in audits:
        category_counts[audit.category] = category_counts.get(
            audit.category,
            0,
        ) + 1

    # 平均质量分数帮助观察数据是否需要进一步人工清洗。
    average_quality = sum(item.quality_score for item in audits) / len(audits)
    invalid_count = sum(not item.is_valid for item in audits)
    issue_count = sum(len(item.issues) for item in audits)

    return {
        "total_examples": len(audits),
        "invalid_examples": invalid_count,
        "issue_count": issue_count,
        "average_quality_score": round(average_quality, 3),
        "category_counts": category_counts,
    }


def parse_args() -> argparse.Namespace:
    """Parse runtime options for reproducible data generation."""

    # seed 和 validation_ratio 暴露为参数，便于复现实验划分。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    """Generate the audited instruction-tuning dataset."""

    # 构造原始样本后先审计，再做划分，防止问题样本被静默写入训练集。
    args = parse_args()
    examples = raw_examples()
    audits = audit_dataset(examples)

    # 数据划分只处理通过结构检查的样本，质量较低样本仍保留审计记录。
    valid_examples = [
        example
        for example, audit in zip(examples, audits)
        if audit.is_valid and not audit.issues
    ]
    train, validation = split_dataset(
        valid_examples,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
    )

    # 写入 JSONL 和审计 JSON，形成“数据 + 质量记录”的完整产物。
    OUTPUT_DIR.mkdir(exist_ok=True)
    write_jsonl(OUTPUT_DIR / "week7_instruction_train.jsonl", train)
    write_jsonl(OUTPUT_DIR / "week7_instruction_valid.jsonl", validation)
    (OUTPUT_DIR / "week7_instruction_audit.json").write_text(
        json.dumps([asdict(item) for item in audits], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 打印摘要，让命令行运行结果可直接写入个人周报。
    summary = build_summary(audits)
    summary.update({"train_size": len(train), "validation_size": len(validation)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artifacts written to: {OUTPUT_DIR}")


# 保持入口单一，便于 py_compile、单元测试或 notebook 复用函数。
if __name__ == "__main__":
    main()
