"""Aggregate week-eight stage-four experiment outputs.

This script reads JSON artifacts produced by the other week-eight scripts and
creates a concise experiment report.  It allows the weekly report to reference
one summary instead of manually opening every result file.
"""

# 汇总脚本只读取本目录 outputs，不依赖任何深度学习环境。
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# 输入输出目录固定，确保阶段四结果可重复汇总。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class ArtifactSummary:
    """A compact summary for one experiment artifact."""

    # artifact_name 是 JSON 结果文件名称。
    artifact_name: str

    # task_name 是根据文件名推断的实验任务。
    task_name: str

    # headline_metric 保存最重要的指标或状态。
    headline_metric: str

    # note 补充解释该产物在阶段四中的作用。
    note: str


def load_json(path: Path) -> dict[str, Any] | list[Any]:
    """Load a JSON artifact with UTF-8 encoding."""

    # 统一读取函数便于后续扩展异常处理或 schema 校验。
    return json.loads(path.read_text(encoding="utf-8"))


def infer_summary(path: Path) -> ArtifactSummary:
    """Infer a concise summary from one artifact file."""

    # 每类产物的指标结构不同，因此按文件名做轻量路由。
    data = load_json(path)
    name = path.name

    # 语义检索重点看 Recall@K 和 MRR。
    if "semantic_search" in name and isinstance(data, dict):
        metrics = data.get("metrics", {})
        return ArtifactSummary(
            artifact_name=name,
            task_name="semantic_search",
            headline_metric=str(metrics),
            note="Records retrieval ranking quality and PaddleNLP availability.",
        )

    # DocVQA 重点看答案准确率和证据定位记录。
    if "doc_vqa" in name and isinstance(data, dict):
        metrics = data.get("metrics", {})
        return ArtifactSummary(
            artifact_name=name,
            task_name="document_vqa",
            headline_metric=str(metrics),
            note="Records OCR tokens, evidence boxes, answers and confidence.",
        )

    # 情感分析重点看分类指标和错误样例。
    if "sentiment" in name and isinstance(data, dict):
        metrics = data.get("metrics", {})
        return ArtifactSummary(
            artifact_name=name,
            task_name="sentiment_analysis",
            headline_metric=str(metrics),
            note="Records dry-run sentiment predictions and PaddleNLP plan.",
        )

    # 伦理审计重点看是否允许发布。
    if "ethics" in name and isinstance(data, dict):
        decision = data.get("decision", {})
        return ArtifactSummary(
            artifact_name=name,
            task_name="ethics_audit",
            headline_metric=str(decision),
            note="Records release gates for social and ethical risks.",
        )

    # 解释性分析重点看反事实是否改变预测。
    if "explainability" in name and isinstance(data, list):
        changed = sum(
            item.get("prediction") != item.get("counterfactual_prediction")
            for item in data
            if isinstance(item, dict)
        )
        return ArtifactSummary(
            artifact_name=name,
            task_name="model_explainability",
            headline_metric=f"counterfactual_changed={changed}/{len(data)}",
            note="Records token contributions, deletion tests and counterfactuals.",
        )

    # 知识增强概念图是理论作业产物，没有单一数值指标。
    if "knowledge_augmentation" in name and isinstance(data, dict):
        return ArtifactSummary(
            artifact_name=name,
            task_name="knowledge_augmentation",
            headline_metric=(
                f"methods={len(data.get('methods', []))}, "
                f"metrics={len(data.get('metrics', []))}"
            ),
            note="Records methods, metrics and conceptual relations.",
        )

    # 未识别文件仍保留，避免汇总时悄悄遗漏产物。
    return ArtifactSummary(
        artifact_name=name,
        task_name="unknown",
        headline_metric="not_inferred",
        note="Artifact detected but no specialized summary rule matched.",
    )


def collect_summaries() -> list[ArtifactSummary]:
    """Collect summaries for all week-eight JSON artifacts."""

    # 只汇总 week8 前缀文件，避免误读其他阶段产物。
    summaries: list[ArtifactSummary] = []
    for path in sorted(OUTPUT_DIR.glob("week8_*.json")):
        summaries.append(infer_summary(path))
    return summaries


def build_markdown(summaries: list[ArtifactSummary]) -> str:
    """Render a Markdown summary report."""

    # Markdown 表格可直接贴入周报或学习笔记。
    lines = [
        "# Week 8 Stage Four Experiment Summary",
        "",
        "| Task | Artifact | Headline Metric | Note |",
        "| --- | --- | --- | --- |",
    ]
    for summary in summaries:
        lines.append(
            "| "
            f"{summary.task_name} | {summary.artifact_name} | "
            f"{summary.headline_metric} | {summary.note} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Create JSON and Markdown summaries for stage-four outputs."""

    # 汇总前先确保 outputs 存在；没有产物时也生成空报告提醒用户。
    OUTPUT_DIR.mkdir(exist_ok=True)
    summaries = collect_summaries()

    # JSON 面向程序读取，Markdown 面向人类阅读。
    summary_json = OUTPUT_DIR / "week8_stage4_summary.json"
    summary_json.write_text(
        json.dumps(
            [summary.__dict__ for summary in summaries],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_md = OUTPUT_DIR / "week8_stage4_summary.md"
    summary_md.write_text(build_markdown(summaries), encoding="utf-8")

    # 命令行摘要说明当前汇总了多少个产物。
    print(
        json.dumps(
            {
                "summarized_artifacts": len(summaries),
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


# 主入口使该脚本可作为最后一步单独执行。
if __name__ == "__main__":
    main()
