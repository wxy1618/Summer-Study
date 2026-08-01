"""HuggingFace-style extractive QA fine-tuning script.

The default mode is dry-run, so the file can be submitted and inspected without
requiring local model downloads.  Use ``--run-training`` only when transformers
and datasets are already available in the environment.
"""

# 可选依赖全部延迟导入，保证“生成代码文件”不等于“配置环境”。
from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path


# 与第七周其他脚本保持同一个输出目录。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class RawQaExample:
    """A minimal SQuAD-like example."""

    # identifier 用于从 tokenized feature 追溯到原始样本。
    identifier: str

    # question 是自然语言问题。
    question: str

    # context 是包含答案的证据文本。
    context: str

    # answer_text 是抽取式问答的标准答案。
    answer_text: str


def build_examples() -> list[RawQaExample]:
    """Create tiny QA examples for the week-seven topic."""

    # 这些样例用于展示流程，不代表完整 SQuAD 规模训练。
    return [
        RawQaExample(
            identifier="qa_lora",
            question="What does LoRA train?",
            context=(
                "LoRA freezes base model weights and trains low-rank adapter "
                "matrices instead of updating all original weights."
            ),
            answer_text="low-rank adapter matrices",
        ),
        RawQaExample(
            identifier="qa_rag",
            question="What does RAG retrieve before generation?",
            context=(
                "RAG retrieves external documents before generation so that "
                "the answer can be grounded in source evidence."
            ),
            answer_text="external documents",
        ),
        RawQaExample(
            identifier="qa_instruction",
            question="What does instruction tuning help a model follow?",
            context=(
                "Instruction tuning helps a pretrained model follow user "
                "requests and produce answers in the required format."
            ),
            answer_text="user requests",
        ),
    ]


def to_squad_rows(examples: list[RawQaExample]) -> list[dict[str, object]]:
    """Convert examples to a SQuAD-like dictionary format."""

    # HuggingFace QA tutorials commonly use answers with text and answer_start.
    rows: list[dict[str, object]] = []
    for item in examples:
        answer_start = item.context.index(item.answer_text)
        rows.append(
            {
                "id": item.identifier,
                "question": item.question,
                "context": item.context,
                "answers": {
                    "text": [item.answer_text],
                    "answer_start": [answer_start],
                },
            }
        )
    return rows


def build_dry_run_plan(model_name: str) -> dict[str, object]:
    """Build a QA fine-tuning plan without loading optional dependencies."""

    # dry-run 体现问答微调的关键步骤，适合没有配置环境时提交。
    rows = to_squad_rows(build_examples())
    return {
        "mode": "dry_run",
        "model_name": model_name,
        "task": "extractive_question_answering",
        "examples": len(rows),
        "pipeline": [
            "Load AutoTokenizer and AutoModelForQuestionAnswering.",
            "Tokenize question/context pairs with truncation='only_second'.",
            "Use offset_mapping to align answer_start and answer_end labels.",
            "Train with Trainer for a small number of epochs.",
            "Decode start/end logits into answer spans.",
            "Evaluate exact match and token-level overlap on validation data.",
        ],
        "sample": rows[0],
    }


def import_hf_dependencies() -> dict[str, object] | None:
    """Import optional libraries only when training is explicitly requested."""

    # 缺依赖时只报告原因，不安装包、不改环境，符合本周任务边界。
    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForQuestionAnswering,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        print(f"Optional dependency is missing: {error}")
        return None

    # 返回字典让后续代码显式使用依赖，避免模块级硬绑定。
    return {
        "Dataset": Dataset,
        "AutoModelForQuestionAnswering": AutoModelForQuestionAnswering,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def filtered_training_args(training_args_cls: object, **kwargs: object) -> object:
    """Create TrainingArguments with version-tolerant keyword filtering."""

    # Transformers 不同版本参数名可能变化，按签名过滤可以减少脆弱性。
    signature = inspect.signature(training_args_cls)
    accepted = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return training_args_cls(**accepted)


def run_training(
    model_name: str,
    output_dir: Path,
    max_length: int,
    doc_stride: int,
) -> dict[str, object]:
    """Run a tiny optional QA fine-tuning experiment."""

    # 环境可用性是训练前置条件；缺失时自动退回 dry-run 产物。
    deps = import_hf_dependencies()
    if deps is None:
        return build_dry_run_plan(model_name)

    # 本段取出运行时依赖，保持顶层模块仍可独立编译。
    Dataset = deps["Dataset"]
    AutoTokenizer = deps["AutoTokenizer"]
    AutoModelForQuestionAnswering = deps["AutoModelForQuestionAnswering"]
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]

    # 构造一个极小的 SQuAD-like 数据集，用于检验流程是否闭环。
    rows = to_squad_rows(build_examples())
    dataset = Dataset.from_list(rows).train_test_split(test_size=0.34, seed=42)

    # tokenizer 负责处理长上下文截断，并返回 offset_mapping。
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # 预处理函数是 QA 微调的核心：把字符答案映射到 token 位置。
    def preprocess(batch: dict[str, list[object]]) -> dict[str, object]:
        questions = [str(question).strip() for question in batch["question"]]
        tokenized = tokenizer(
            questions,
            batch["context"],
            max_length=max_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # overflowing 映射告诉我们每个窗口来自哪条原始样本。
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")
        start_positions: list[int] = []
        end_positions: list[int] = []

        # 逐个 feature 计算 span 标签，避免长文本窗口错位。
        for feature_index, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][feature_index]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized.sequence_ids(feature_index)
            sample_index = sample_mapping[feature_index]
            answers = batch["answers"][sample_index]
            answer_start = answers["answer_start"][0]
            answer_text = answers["text"][0]
            answer_end = answer_start + len(answer_text)

            # 找到 context 在拼接序列中的 token 边界。
            token_start = 0
            while sequence_ids[token_start] != 1:
                token_start += 1
            token_end = len(input_ids) - 1
            while sequence_ids[token_end] != 1:
                token_end -= 1

            # 如果当前窗口不含答案，把标签设为 CLS，表示不可回答窗口。
            if (
                offsets[token_start][0] > answer_start
                or offsets[token_end][1] < answer_end
            ):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                continue

            # 向内移动 token_start/token_end，精确包住答案字符 span。
            while (
                token_start < len(offsets)
                and offsets[token_start][0] <= answer_start
            ):
                token_start += 1
            start_positions.append(token_start - 1)

            while offsets[token_end][1] >= answer_end:
                token_end -= 1
            end_positions.append(token_end + 1)

        # 返回模型训练所需字段，Trainer 会读取 start/end positions。
        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    # 数据预处理去除原始文本列，减少 Trainer 批处理时的干扰。
    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # 训练参数保持轻量，目的在于熟悉 HuggingFace QA 微调骨架。
    training_args = filtered_training_args(
        TrainingArguments,
        output_dir=str(output_dir / "hf_question_answering_model"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="epoch",
        evaluation_strategy="epoch",
        report_to=[],
    )

    # Trainer 运行最小训练和评估流程，不在脚本中配置任何环境。
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
    )
    trainer.train()
    metrics = trainer.evaluate()

    # 保存模型和 tokenizer，形成可复现的实验输出目录。
    model_save_dir = output_dir / "hf_question_answering_model"
    trainer.save_model(str(model_save_dir))
    tokenizer.save_pretrained(str(model_save_dir))
    return {"mode": "trained", "model_name": model_name, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for QA fine-tuning."""

    # 默认不训练，避免未授权下载模型或占用计算资源。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--doc-stride", type=int, default=32)
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="Actually run HuggingFace Trainer if dependencies already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and save the plan without loading model dependencies.",
    )
    return parser.parse_args()


def main() -> None:
    """Run dry-run planning or optional QA training."""

    # dry-run 也会写入正式产物，便于周报记录当前完成度。
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 只有显式指定 --run-training 才会加载模型和训练依赖。
    if args.dry_run or not args.run_training:
        result = build_dry_run_plan(args.model_name)
    else:
        result = run_training(
            args.model_name,
            OUTPUT_DIR,
            max_length=args.max_length,
            doc_stride=args.doc_stride,
        )

    # 保存结果，便于从 dry-run 平滑过渡到真实训练实验。
    output_path = OUTPUT_DIR / "week7_hf_question_answering_plan.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Artifact written to: {output_path}")


# 主入口单独保留，方便导入其中的数据构造函数做测试。
if __name__ == "__main__":
    main()
