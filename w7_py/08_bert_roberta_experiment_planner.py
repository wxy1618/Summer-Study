"""Generate an experiment matrix for BERT and RoBERTa fine-tuning.

The script records optional experiments in a structured way.  It is useful when
the environment is not configured yet, because the research plan can still be
checked for fairness, variables, metrics, and expected artifacts.
"""

# 实验规划只需要标准库，避免为了生成计划而依赖深度学习框架。
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path


# 本周所有结果集中到 outputs，便于提交和后续周报汇总。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class ExperimentConfig:
    """A fine-tuning experiment configuration."""

    # experiment_id 是可复现实验编号。
    experiment_id: str

    # model_family 表示 BERT 或 RoBERTa。
    model_family: str

    # checkpoint 是 HuggingFace 模型名称或本地路径。
    checkpoint: str

    # task 表示微调任务类型。
    task: str

    # tuning_strategy 记录全量、冻结或 LoRA 等策略。
    tuning_strategy: str

    # learning_rate 是本轮计划的核心超参数之一。
    learning_rate: float

    # epochs 控制训练轮数。
    epochs: int

    # batch_size 控制每步样本数量。
    batch_size: int

    # primary_metric 是选择模型时优先观察的指标。
    primary_metric: str

    # notes 记录实验假设，防止配置矩阵变成机械枚举。
    notes: str


def build_experiment_matrix() -> list[ExperimentConfig]:
    """Create a controlled BERT/RoBERTa comparison matrix."""

    # checkpoint 选择经典基座模型，便于在 Colab 或本地复现实验。
    models = [
        ("BERT", "bert-base-uncased"),
        ("RoBERTa", "roberta-base"),
    ]

    # 任务覆盖第七周要求的文本分类和问答。
    tasks = [
        ("text_classification", "macro_f1"),
        ("question_answering", "exact_match_f1"),
    ]

    # tuning_strategy 控制是否更新全部参数或只更新少量参数。
    strategies = [
        "full_finetune",
        "freeze_encoder_head_only",
        "lora_adapter_optional",
    ]

    # 选择两组学习率，不做过大网格，符合小样本课程作业范围。
    learning_rates = [2e-5, 5e-5]
    epochs = [2]
    batch_sizes = [8]

    # 矩阵生成时保留实验假设，体现变量控制和结果解释意识。
    configs: list[ExperimentConfig] = []
    counter = 1
    for (family, checkpoint), (task, metric), strategy, lr, ep, batch in product(
        models,
        tasks,
        strategies,
        learning_rates,
        epochs,
        batch_sizes,
    ):
        if strategy == "freeze_encoder_head_only":
            notes = "Tests whether task head alone is enough on small data."
        elif strategy == "lora_adapter_optional":
            notes = "Requires PEFT; compares adapter efficiency with full tuning."
        else:
            notes = "Reference setting that updates all trainable weights."
        configs.append(
            ExperimentConfig(
                experiment_id=f"w7_exp_{counter:02d}",
                model_family=family,
                checkpoint=checkpoint,
                task=task,
                tuning_strategy=strategy,
                learning_rate=lr,
                epochs=ep,
                batch_size=batch,
                primary_metric=metric,
                notes=notes,
            )
        )
        counter += 1
    return configs


def write_csv(configs: list[ExperimentConfig], path: Path) -> None:
    """Write the experiment matrix as CSV."""

    # CSV 方便导入 Excel，在周报中记录计划与实际结果。
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(configs[0]).keys()))
        writer.writeheader()
        for config in configs:
            writer.writerow(asdict(config))


def write_markdown(configs: list[ExperimentConfig], path: Path) -> None:
    """Write the experiment matrix as a readable Markdown table."""

    # Markdown 版本适合直接附在学习笔记或提交说明中。
    lines = [
        "# BERT/RoBERTa Fine-tuning Experiment Matrix",
        "",
        "| ID | Model | Task | Strategy | LR | Epochs | Batch | Metric |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for config in configs:
        lines.append(
            "| "
            f"{config.experiment_id} | "
            f"{config.checkpoint} | "
            f"{config.task} | "
            f"{config.tuning_strategy} | "
            f"{config.learning_rate:.0e} | "
            f"{config.epochs} | "
            f"{config.batch_size} | "
            f"{config.primary_metric} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(configs: list[ExperimentConfig]) -> dict[str, object]:
    """Summarize the planned experiment coverage."""

    # 汇总不同模型、任务和策略的数量，确认矩阵没有漏项。
    model_families = sorted({config.model_family for config in configs})
    tasks = sorted({config.task for config in configs})
    strategies = sorted({config.tuning_strategy for config in configs})
    return {
        "total_experiments": len(configs),
        "model_families": model_families,
        "tasks": tasks,
        "tuning_strategies": strategies,
        "remark": (
            "This is a controlled plan. Real runs should fill validation "
            "metrics and error cases after the environment is available."
        ),
    }


def main() -> None:
    """Generate and save the optional fine-tuning plan."""

    # 实验计划先于训练环境存在，能帮助后续执行时保持变量控制。
    configs = build_experiment_matrix()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 三种格式分别服务于机器读取、表格分析和人工阅读。
    write_csv(configs, OUTPUT_DIR / "week7_bert_roberta_experiments.csv")
    write_markdown(configs, OUTPUT_DIR / "week7_bert_roberta_experiments.md")
    (OUTPUT_DIR / "week7_bert_roberta_experiments.json").write_text(
        json.dumps([asdict(item) for item in configs], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 打印摘要作为命令行实验结果。
    print(json.dumps(build_summary(configs), ensure_ascii=False, indent=2))
    print(f"Artifacts written to: {OUTPUT_DIR}")


# 标准入口确保导入脚本时不会产生文件副作用。
if __name__ == "__main__":
    main()
