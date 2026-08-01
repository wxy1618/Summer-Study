"""Compare parameter costs of common fine-tuning strategies.

The goal is to translate conceptual PEFT methods into concrete numbers.  The
estimates are simplified, but they make the scale difference between full
fine-tuning, Prompt Tuning, P-Tuning, and LoRA visible.
"""

# 本脚本只做参数量估算，因此使用标准库即可完成全部实验记录。
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


# 固定输出目录，便于和本周其他实验产物集中管理。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class TransformerSpec:
    """Simplified transformer size configuration."""

    # num_layers 决定注意力和前馈模块重复次数。
    num_layers: int

    # hidden_size 对应 token 隐状态宽度，是参数量的核心因素。
    hidden_size: int

    # intermediate_size 对应前馈网络的扩展维度。
    intermediate_size: int

    # vocab_size 影响词嵌入和输出层参数量。
    vocab_size: int


@dataclass(frozen=True)
class MethodEstimate:
    """Parameter and storage estimate for one adaptation method."""

    # method 记录微调策略名称。
    method: str

    # trainable_parameters 表示训练时需要更新的参数规模。
    trainable_parameters: int

    # percent_of_full 显示相对全量微调的比例。
    percent_of_full: float

    # fp16_megabytes 估算 adapter 或可训练参数的 FP16 存储量。
    fp16_megabytes: float

    # note 用简短文字解释估算假设，避免数字脱离上下文。
    note: str


def estimate_total_parameters(spec: TransformerSpec) -> int:
    """Estimate total parameters of a decoder-like transformer."""

    # 注意力层粗略包含 q/k/v/o 四个 hidden-to-hidden 矩阵。
    attention_params = 4 * spec.hidden_size * spec.hidden_size

    # 前馈层包含 hidden->intermediate 与 intermediate->hidden 两个矩阵。
    ffn_params = 2 * spec.hidden_size * spec.intermediate_size

    # 每层参数乘以层数，得到主体 transformer block 参数。
    block_params = spec.num_layers * (attention_params + ffn_params)

    # 词嵌入是大模型的重要组成部分，这里按一份 embedding 估算。
    embedding_params = spec.vocab_size * spec.hidden_size
    return block_params + embedding_params


def estimate_lora_parameters(
    spec: TransformerSpec,
    rank: int,
    target_matrices_per_layer: int,
) -> int:
    """Estimate trainable LoRA parameters for selected projection matrices."""

    # 每个 LoRA 矩阵对的参数为 r*(d_in+d_out)，此处假设方阵投影。
    per_matrix = rank * (spec.hidden_size + spec.hidden_size)

    # LoRA 通常作用于 q/v 或 q/k/v/o 等投影，因此乘以目标矩阵个数。
    return spec.num_layers * target_matrices_per_layer * per_matrix


def estimate_prompt_tuning_parameters(
    spec: TransformerSpec,
    virtual_tokens: int,
) -> int:
    """Estimate trainable soft prompt parameters."""

    # Prompt Tuning 只在输入端学习虚拟 token embedding。
    return virtual_tokens * spec.hidden_size


def estimate_p_tuning_v2_parameters(
    spec: TransformerSpec,
    virtual_tokens: int,
) -> int:
    """Estimate trainable deep prompt parameters for P-Tuning v2."""

    # P-Tuning v2 可在每层注入 prompt，因此参数量按层数扩大。
    return spec.num_layers * virtual_tokens * spec.hidden_size * 2


def to_megabytes(parameters: int, bytes_per_parameter: int = 2) -> float:
    """Convert a parameter count to approximate storage in megabytes."""

    # FP16/BF16 每个参数通常占 2 字节，这里用于估算 adapter 体积。
    return round(parameters * bytes_per_parameter / (1024**2), 3)


def build_estimates(spec: TransformerSpec) -> list[MethodEstimate]:
    """Build comparable estimates for several adaptation methods."""

    # full_params 是参照物，其他方法都用它计算相对比例。
    full_params = estimate_total_parameters(spec)

    # 选取常见教学配置：64 个虚拟 token，LoRA rank 为 8。
    prompt_params = estimate_prompt_tuning_parameters(spec, virtual_tokens=64)
    p_tuning_params = estimate_p_tuning_v2_parameters(spec, virtual_tokens=64)
    lora_qv_params = estimate_lora_parameters(
        spec,
        rank=8,
        target_matrices_per_layer=2,
    )
    lora_qkvo_params = estimate_lora_parameters(
        spec,
        rank=8,
        target_matrices_per_layer=4,
    )

    # 每个方法保留一个解释性 note，避免参数量被误解为最终效果。
    methods = [
        ("Full fine-tuning", full_params, "Update all model weights."),
        ("Prompt Tuning", prompt_params, "Train input soft prompt only."),
        ("P-Tuning v2", p_tuning_params, "Train deep prompts across layers."),
        ("LoRA q/v", lora_qv_params, "Train rank-8 adapters on q and v."),
        ("LoRA q/k/v/o", lora_qkvo_params, "Train rank-8 adapters on q/k/v/o."),
    ]

    # 统一换算比例和存储规模，方便横向比较。
    estimates: list[MethodEstimate] = []
    for name, params, note in methods:
        estimates.append(
            MethodEstimate(
                method=name,
                trainable_parameters=params,
                percent_of_full=round(params / full_params * 100, 6),
                fp16_megabytes=to_megabytes(params),
                note=note,
            )
        )
    return estimates


def write_outputs(spec: TransformerSpec, estimates: list[MethodEstimate]) -> None:
    """Write CSV, JSON, and Markdown reports."""

    # 输出目录集中保存，便于从周报定位到参数估算表。
    OUTPUT_DIR.mkdir(exist_ok=True)

    # CSV 适合后续用 Excel 或 pandas 继续分析。
    csv_path = OUTPUT_DIR / "week7_peft_parameter_estimates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(estimates[0]).keys()))
        writer.writeheader()
        for estimate in estimates:
            writer.writerow(asdict(estimate))

    # JSON 保存模型规格和估算结果，结构化程度更高。
    json_path = OUTPUT_DIR / "week7_peft_parameter_estimates.json"
    json_path.write_text(
        json.dumps(
            {"model_spec": asdict(spec), "estimates": [asdict(e) for e in estimates]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Markdown 表格可直接复制到学习笔记或周报中。
    markdown_lines = [
        "# PEFT Parameter Estimates",
        "",
        "| Method | Trainable Parameters | % of Full | FP16 MB | Note |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for estimate in estimates:
        markdown_lines.append(
            "| "
            f"{estimate.method} | "
            f"{estimate.trainable_parameters:,} | "
            f"{estimate.percent_of_full:.6f}% | "
            f"{estimate.fp16_megabytes:.3f} | "
            f"{estimate.note} |"
        )
    (OUTPUT_DIR / "week7_peft_parameter_estimates.md").write_text(
        "\n".join(markdown_lines),
        encoding="utf-8",
    )


def main() -> None:
    """Run the parameter accounting experiment."""

    # 使用接近小型 transformer 的规格，数值可解释且运行成本为零。
    spec = TransformerSpec(
        num_layers=12,
        hidden_size=768,
        intermediate_size=3072,
        vocab_size=30522,
    )

    # 估算每种方法的可训练参数量和相对全量微调比例。
    estimates = build_estimates(spec)

    # 保存三种格式产物，分别服务于程序读取、表格分析和周报展示。
    write_outputs(spec, estimates)

    # 命令行打印核心对比，快速展示 LoRA 与全量微调的数量级差异。
    for estimate in estimates:
        print(
            f"{estimate.method:18s} "
            f"{estimate.trainable_parameters:12,d} params "
            f"({estimate.percent_of_full:.6f}% of full)"
        )
    print(f"Artifacts written to: {OUTPUT_DIR}")


# 使用标准主入口，避免导入本脚本时自动生成实验文件。
if __name__ == "__main__":
    main()
