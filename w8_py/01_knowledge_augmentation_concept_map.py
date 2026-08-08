"""Build a concept map for knowledge-enhanced language models.

The script turns week-eight lecture concepts into structured artifacts.  It is
not a neural experiment; it is a reproducible way to record methods, evaluation
criteria, and conceptual links before implementing retrieval or fine-tuning.
"""

# 本实验强调知识增强方法论，因此标准库足以生成可审计记录。
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


# 输出目录固定在第八周作业目录下，便于和后续实验结果一起归档。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class KnowledgeMethod:
    """A method for adding knowledge to a language model system."""

    # name 是知识增强方法名称，用于报告和图谱节点。
    name: str

    # stage 表示知识注入发生在预训练、微调、推理或评估哪个阶段。
    stage: str

    # knowledge_type 区分参数化知识和非参数化外部知识。
    knowledge_type: str

    # benefit 记录该方法最主要的应用收益。
    benefit: str

    # limitation 保留方法边界，避免把知识增强理解成万能方案。
    limitation: str


@dataclass(frozen=True)
class EvaluationMetric:
    """A metric used to evaluate knowledge-enhanced systems."""

    # name 是评价指标名称。
    name: str

    # target_stage 表示指标主要检查检索、生成或治理环节。
    target_stage: str

    # interpretation 说明指标高低应如何解释。
    interpretation: str


@dataclass(frozen=True)
class ConceptEdge:
    """A directed relation between two concept nodes."""

    # source 和 target 构成一条概念关系边。
    source: str
    target: str

    # relation 用自然语言解释二者之间的关系。
    relation: str


def build_methods() -> list[KnowledgeMethod]:
    """Create the core method list for knowledge enhancement."""

    # 方法覆盖课程中常见的参数化、非参数化和混合知识路径。
    return [
        KnowledgeMethod(
            name="domain_adaptive_pretraining",
            stage="pretraining",
            knowledge_type="parametric",
            benefit="Absorbs domain language patterns and frequent facts.",
            limitation="Expensive to update and difficult to cite.",
        ),
        KnowledgeMethod(
            name="supervised_instruction_tuning",
            stage="fine_tuning",
            knowledge_type="parametric_behavior",
            benefit="Teaches the model task format and response style.",
            limitation="May overfit small or noisy instruction data.",
        ),
        KnowledgeMethod(
            name="retrieval_augmented_generation",
            stage="inference",
            knowledge_type="non_parametric",
            benefit="Injects fresh and auditable external evidence.",
            limitation="Answer quality is bounded by retrieval quality.",
        ),
        KnowledgeMethod(
            name="knowledge_graph_grounding",
            stage="inference_or_training",
            knowledge_type="structured_external",
            benefit="Provides explicit entities, relations, and paths.",
            limitation="Coverage and schema construction can be costly.",
        ),
        KnowledgeMethod(
            name="tool_or_database_calling",
            stage="inference",
            knowledge_type="external_tool",
            benefit="Allows models to query authoritative structured sources.",
            limitation="Requires robust tool routing and permission control.",
        ),
    ]


def build_metrics() -> list[EvaluationMetric]:
    """Create evaluation metrics for knowledge-enhanced systems."""

    # 指标故意覆盖检索、生成和治理三层，避免只看最终文本流畅度。
    return [
        EvaluationMetric(
            name="Recall@K",
            target_stage="retrieval",
            interpretation="Measures whether relevant evidence appears in top-K.",
        ),
        EvaluationMetric(
            name="MRR",
            target_stage="retrieval",
            interpretation="Rewards relevant evidence being ranked earlier.",
        ),
        EvaluationMetric(
            name="Faithfulness",
            target_stage="generation",
            interpretation="Checks whether the answer is supported by evidence.",
        ),
        EvaluationMetric(
            name="Citation Accuracy",
            target_stage="generation",
            interpretation="Checks whether cited sources truly support claims.",
        ),
        EvaluationMetric(
            name="Risk Gate Pass Rate",
            target_stage="governance",
            interpretation="Checks whether data, safety, and rollback gates pass.",
        ),
    ]


def build_edges() -> list[ConceptEdge]:
    """Connect the methods and metrics into a concept map."""

    # 概念边帮助复习时看到方法之间的逻辑，而不是孤立记忆术语。
    return [
        ConceptEdge(
            source="retrieval_augmented_generation",
            target="Recall@K",
            relation="retrieval quality must be measured before judging answers",
        ),
        ConceptEdge(
            source="retrieval_augmented_generation",
            target="Faithfulness",
            relation="generation should be constrained by retrieved evidence",
        ),
        ConceptEdge(
            source="domain_adaptive_pretraining",
            target="supervised_instruction_tuning",
            relation="domain knowledge can precede task-specific behavior tuning",
        ),
        ConceptEdge(
            source="knowledge_graph_grounding",
            target="Citation Accuracy",
            relation="explicit sources make answer verification easier",
        ),
        ConceptEdge(
            source="tool_or_database_calling",
            target="Risk Gate Pass Rate",
            relation="external actions require permission and monitoring gates",
        ),
    ]


def build_markdown(
    methods: list[KnowledgeMethod],
    metrics: list[EvaluationMetric],
    edges: list[ConceptEdge],
) -> str:
    """Render a compact Markdown report for the concept map."""

    # Markdown 报告可以直接放入课堂笔记或周报附件。
    lines = [
        "# Knowledge Augmentation Concept Map",
        "",
        "## Methods",
        "",
        "| Method | Stage | Knowledge Type | Benefit | Limitation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for method in methods:
        lines.append(
            "| "
            f"{method.name} | {method.stage} | {method.knowledge_type} | "
            f"{method.benefit} | {method.limitation} |"
        )

    # 评价指标单独成表，强调实验报告不能只描述方法。
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Target Stage | Interpretation |",
            "| --- | --- | --- |",
        ]
    )
    for metric in metrics:
        lines.append(
            "| "
            f"{metric.name} | {metric.target_stage} | "
            f"{metric.interpretation} |"
        )

    # 概念边用列表表达，更适合阅读逻辑关系。
    lines.extend(["", "## Relations", ""])
    for edge in edges:
        lines.append(f"- `{edge.source}` -> `{edge.target}`: {edge.relation}")
    return "\n".join(lines)


def main() -> None:
    """Generate the concept-map artifacts."""

    # 构建三类结构化对象，分别代表方法、指标和关系。
    methods = build_methods()
    metrics = build_metrics()
    edges = build_edges()

    # JSON 保存结构化数据，Markdown 保存人工可读报告。
    OUTPUT_DIR.mkdir(exist_ok=True)
    payload = {
        "methods": [asdict(item) for item in methods],
        "metrics": [asdict(item) for item in metrics],
        "edges": [asdict(item) for item in edges],
    }
    json_path = OUTPUT_DIR / "week8_knowledge_augmentation_concept_map.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Markdown 产物服务于课程作业中的“视频学习笔记以及作业”要求。
    markdown_path = OUTPUT_DIR / "week8_knowledge_augmentation_concept_map.md"
    markdown_path.write_text(
        build_markdown(methods, metrics, edges),
        encoding="utf-8",
    )

    # 命令行摘要用于快速确认输出是否覆盖核心维度。
    print(
        json.dumps(
            {
                "methods": len(methods),
                "metrics": len(metrics),
                "relations": len(edges),
                "output_dir": str(OUTPUT_DIR),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


# 主入口保持单一，便于导入函数进行单元测试或 notebook 展示。
if __name__ == "__main__":
    main()
