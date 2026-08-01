"""HuggingFace-style text classification fine-tuning script.

By default this script runs in dry-run mode and prints a reproducible plan.
Use ``--run-training`` only in an environment where transformers and datasets
are already installed.  The script never installs or configures dependencies.
"""

# 依赖导入放在训练函数内部，保证未安装环境也能进行语法检查和 dry-run。
from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path


# 输出目录与其他第七周脚本保持一致，方便统一整理实验结果。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class ClassificationExample:
    """A minimal text classification sample."""

    # text 是模型输入，来自第七周大模型学习主题。
    text: str

    # label 是整数类别，符合 HuggingFace 分类模型输入要求。
    label: int

    # label_name 保留可读标签，便于人工检查训练样本。
    label_name: str


def build_examples() -> list[ClassificationExample]:
    """Create a tiny supervised dataset for demonstration."""

    # 类别设计为 rag、fine_tuning、model_principle 三类课程主题。
    label_to_id = {"rag": 0, "fine_tuning": 1, "model_principle": 2}

    # 小样本不能代表真实泛化性能，但足够说明完整微调管线。
    rows = [
        (
            "RAG retrieves external documents before generation.",
            "rag",
        ),
        (
            "A vector store helps retrieve evidence for user questions.",
            "rag",
        ),
        (
            "LoRA trains low-rank adapters instead of all weights.",
            "fine_tuning",
        ),
        (
            "Instruction tuning teaches models to follow user requests.",
            "fine_tuning",
        ),
        (
            "A language model predicts the next token from context.",
            "model_principle",
        ),
        (
            "T5 converts classification and QA into text-to-text tasks.",
            "model_principle",
        ),
    ]

    # 同时保存 label_name 和 label_id，避免后续分析时失去语义信息。
    return [
        ClassificationExample(
            text=text,
            label=label_to_id[label_name],
            label_name=label_name,
        )
        for text, label_name in rows
    ]


def labels() -> tuple[dict[str, int], dict[int, str]]:
    """Return deterministic label mappings."""

    # label 映射必须固定，否则模型保存后预测类别会难以解释。
    label_to_id = {"rag": 0, "fine_tuning": 1, "model_principle": 2}
    id_to_label = {identifier: name for name, identifier in label_to_id.items()}
    return label_to_id, id_to_label


def build_dry_run_plan(model_name: str) -> dict[str, object]:
    """Build an executable experiment plan without loading dependencies."""

    # dry-run 记录任务、模型、数据和指标，适合作为无环境配置时的作业产物。
    examples = build_examples()
    label_to_id, id_to_label = labels()
    return {
        "mode": "dry_run",
        "model_name": model_name,
        "task": "sequence_classification",
        "train_examples": len(examples),
        "labels": label_to_id,
        "id_to_label": id_to_label,
        "pipeline": [
            "Load AutoTokenizer.",
            "Tokenize text with padding and truncation.",
            "Load AutoModelForSequenceClassification.",
            "Train with Trainer for a small number of epochs.",
            "Evaluate accuracy on a held-out split.",
            "Save model, tokenizer, and metrics.",
        ],
        "sample": [asdict(item) for item in examples[:2]],
    }


def import_hf_dependencies() -> dict[str, object] | None:
    """Import optional HuggingFace dependencies when training is requested."""

    # 训练依赖属于用户环境状态；缺失时只提示，不做安装或修改。
    try:
        import numpy as np
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        print(f"Optional dependency is missing: {error}")
        return None

    # 统一返回依赖对象，避免在模块顶层制造硬依赖。
    return {
        "np": np,
        "Dataset": Dataset,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def filtered_training_args(training_args_cls: object, **kwargs: object) -> object:
    """Create TrainingArguments while tolerating version differences."""

    # Transformers 版本会调整参数名，按签名过滤能提升脚本鲁棒性。
    signature = inspect.signature(training_args_cls)
    accepted = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return training_args_cls(**accepted)


def run_training(model_name: str, output_dir: Path) -> dict[str, object]:
    """Run a tiny text classification fine-tuning experiment."""

    # 如果依赖不存在，返回 dry-run 信息，不中断整个作业流程。
    deps = import_hf_dependencies()
    if deps is None:
        return build_dry_run_plan(model_name)

    # 本段把动态依赖取出，类型检查让位于运行时可用性。
    np = deps["np"]
    Dataset = deps["Dataset"]
    AutoTokenizer = deps["AutoTokenizer"]
    AutoModelForSequenceClassification = deps["AutoModelForSequenceClassification"]
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]

    # 数据集先转为 HuggingFace Dataset，再使用 train_test_split 划分。
    examples = build_examples()
    dataset = Dataset.from_list(
        [{"text": item.text, "label": item.label} for item in examples]
    )
    dataset = dataset.train_test_split(test_size=0.33, seed=42)

    # tokenizer 与模型来自同一 checkpoint，保证词表和 embedding 对齐。
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _, id_to_label = labels()

    # 文本分类模型需要显式传入标签数和标签映射。
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id_to_label),
        id2label=id_to_label,
        label2id={name: identifier for identifier, name in id_to_label.items()},
    )

    # tokenization 使用 batched map，符合官方任务教程的基本写法。
    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, object]:
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    tokenized = dataset.map(tokenize_batch, batched=True)

    # 简单 accuracy 足以检查小样例训练流程是否贯通。
    def compute_metrics(eval_prediction: object) -> dict[str, float]:
        logits, true_labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        accuracy = float((predictions == true_labels).mean())
        return {"accuracy": accuracy}

    # 参数设置尽量小，目的是验证流程而非追求最终分数。
    training_args = filtered_training_args(
        TrainingArguments,
        output_dir=str(output_dir / "hf_text_classification_model"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="epoch",
        evaluation_strategy="epoch",
        report_to=[],
    )

    # Trainer 封装训练循环、评估和保存，是 HuggingFace 入门常用接口。
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()

    # 保存 tokenizer 与模型，便于后续推理或复现实验。
    model_save_dir = output_dir / "hf_text_classification_model"
    trainer.save_model(str(model_save_dir))
    tokenizer.save_pretrained(str(model_save_dir))
    return {"mode": "trained", "model_name": model_name, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    # 默认 dry-run 是为了符合“不配置环境，仅生成代码文件”的任务要求。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
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
    """Run dry-run planning or optional HuggingFace training."""

    # 创建输出目录，让 dry-run 也留下正式实验记录。
    args = parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 未显式请求训练时保持 dry-run，避免误触发模型下载。
    if args.dry_run or not args.run_training:
        result = build_dry_run_plan(args.model_name)
    else:
        result = run_training(args.model_name, OUTPUT_DIR)

    # JSON 结果既可读又可被后续汇总脚本解析。
    output_path = OUTPUT_DIR / "week7_hf_text_classification_plan.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Artifact written to: {output_path}")


# 入口保持清晰，便于课堂中直接演示 dry-run 与真实训练的差别。
if __name__ == "__main__":
    main()
