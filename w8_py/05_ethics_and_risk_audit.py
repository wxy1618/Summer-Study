"""Ethics and risk audit for week-eight NLP systems.

The script converts social and ethical considerations into a reproducible audit
table.  It is designed for research reporting: every risk has a severity,
likelihood, mitigation, owner, and release decision.
"""

# 伦理审计首先是结构化记录问题，标准库即可完成透明计算。
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


# 输出目录与本周其他作业一致，便于最终汇总。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class RiskItem:
    """One ethical or social risk item."""

    # area 表示风险所属治理维度。
    area: str

    # description 描述风险的具体表现。
    description: str

    # severity 表示一旦发生造成的影响程度，范围 1-5。
    severity: int

    # likelihood 表示当前系统中发生该风险的可能性，范围 1-5。
    likelihood: int

    # mitigation 是计划采取的控制措施。
    mitigation: str

    # owner 表示建议负责处理该风险的角色。
    owner: str


@dataclass(frozen=True)
class AuditDecision:
    """A release decision derived from risk scores."""

    # release_allowed 表示是否建议进入下一阶段。
    release_allowed: bool

    # max_risk_score 是所有风险中的最大 severity*likelihood。
    max_risk_score: int

    # high_risk_count 表示超过阈值的高风险项数量。
    high_risk_count: int

    # decision_note 解释为什么允许或拒绝发布。
    decision_note: str


def build_risk_items() -> list[RiskItem]:
    """Create risk items for NLP and PaddleNLP experiments."""

    # 风险项覆盖数据、模型、部署和结果解释四个层面。
    return [
        RiskItem(
            area="data_provenance",
            description="Training or evaluation samples may lack clear source notes.",
            severity=3,
            likelihood=3,
            mitigation="Record dataset origin, license, collection date and scope.",
            owner="student",
        ),
        RiskItem(
            area="bias_and_fairness",
            description="Sentiment model may behave differently across groups.",
            severity=4,
            likelihood=3,
            mitigation="Create paired test cases and compare subgroup error rates.",
            owner="student_reviewer",
        ),
        RiskItem(
            area="privacy",
            description="Document QA inputs may contain names, IDs or invoices.",
            severity=5,
            likelihood=3,
            mitigation="Mask personal identifiers and restrict raw document access.",
            owner="data_owner",
        ),
        RiskItem(
            area="misinformation",
            description="QA system may answer without sufficient evidence.",
            severity=4,
            likelihood=2,
            mitigation="Require evidence spans and allow abstention on weak retrieval.",
            owner="model_developer",
        ),
        RiskItem(
            area="copyright",
            description="External documents may have unclear redistribution rights.",
            severity=3,
            likelihood=2,
            mitigation="Store source URLs and avoid redistributing restricted content.",
            owner="project_owner",
        ),
        RiskItem(
            area="explainability",
            description="Users may overtrust saliency or attention explanations.",
            severity=3,
            likelihood=3,
            mitigation="Label explanations as diagnostic aids, not causal proof.",
            owner="model_developer",
        ),
        RiskItem(
            area="deployment",
            description="No rollback plan may exist if online quality degrades.",
            severity=4,
            likelihood=2,
            mitigation="Record baseline version, adapter version and rollback trigger.",
            owner="maintainer",
        ),
    ]


def score_risk(item: RiskItem) -> int:
    """Compute a simple risk score."""

    # severity*likelihood 是常见风险矩阵写法，便于初步排序。
    return item.severity * item.likelihood


def build_decision(items: list[RiskItem], threshold: int) -> AuditDecision:
    """Build a conservative release decision from audit items."""

    # 高风险项不能被平均值掩盖，因此同时检查最大风险和数量。
    scores = [score_risk(item) for item in items]
    high_risk_count = sum(score >= threshold for score in scores)
    max_score = max(scores)
    release_allowed = high_risk_count == 0

    # 决策文字强调本审计是发布前门禁，而不是形式化附录。
    if release_allowed:
        note = "All risks are below the release threshold."
    else:
        note = (
            "Hold release until high-risk items have stronger mitigation "
            "evidence and reviewer approval."
        )
    return AuditDecision(
        release_allowed=release_allowed,
        max_risk_score=max_score,
        high_risk_count=high_risk_count,
        decision_note=note,
    )


def write_csv(items: list[RiskItem], path: Path) -> None:
    """Write the risk audit table as CSV."""

    # CSV 方便后续用 Excel 打开，并支持按风险分数排序。
    fieldnames = list(asdict(items[0]).keys()) + ["risk_score"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            row = asdict(item)
            row["risk_score"] = score_risk(item)
            writer.writerow(row)


def write_markdown(
    items: list[RiskItem],
    decision: AuditDecision,
    path: Path,
) -> None:
    """Write a human-readable audit report."""

    # Markdown 报告适合直接嵌入第八周笔记或周报附件。
    lines = [
        "# Ethics and Risk Audit",
        "",
        "| Area | Severity | Likelihood | Score | Mitigation | Owner |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for item in items:
        lines.append(
            "| "
            f"{item.area} | {item.severity} | {item.likelihood} | "
            f"{score_risk(item)} | {item.mitigation} | {item.owner} |"
        )

    # 决策摘要放在表后，突出审计结论。
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- release_allowed: {decision.release_allowed}",
            f"- max_risk_score: {decision.max_risk_score}",
            f"- high_risk_count: {decision.high_risk_count}",
            f"- decision_note: {decision.decision_note}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the ethics audit and save artifacts."""

    # 阈值设为 12，表示 severity*likelihood 达到 12 即需暂缓发布。
    threshold = 12
    items = build_risk_items()
    decision = build_decision(items, threshold=threshold)

    # 同时写 JSON、CSV、Markdown，满足机器读取与人工审查两种需求。
    OUTPUT_DIR.mkdir(exist_ok=True)
    payload = {
        "threshold": threshold,
        "items": [
            {**asdict(item), "risk_score": score_risk(item)} for item in items
        ],
        "decision": asdict(decision),
    }
    (OUTPUT_DIR / "week8_ethics_risk_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(items, OUTPUT_DIR / "week8_ethics_risk_audit.csv")
    write_markdown(items, decision, OUTPUT_DIR / "week8_ethics_risk_audit.md")

    # 打印最终门禁结论，方便在周报中记录是否允许继续上线。
    print(json.dumps(asdict(decision), ensure_ascii=False, indent=2))


# 主入口保持脚本可直接运行，也便于被汇总脚本复用。
if __name__ == "__main__":
    main()
