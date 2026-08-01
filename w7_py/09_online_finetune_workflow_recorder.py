"""Record an online knowledge fine-tuning workflow with safety gates.

The script does not fine-tune a model.  It formalizes the operational process
that should exist before online knowledge updates are allowed: data intake,
quality audit, adapter training, evaluation, release, monitoring, and rollback.
"""

# 在线微调强调流程治理，本脚本用标准库生成可审计记录。
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


# 固定输出目录，使流程记录可直接进入周报或项目附件。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class FeedbackExample:
    """A user-feedback item proposed for online fine-tuning."""

    # identifier 用于追踪样本从收集到训练的完整生命周期。
    identifier: str

    # instruction 是触发反馈的用户任务。
    instruction: str

    # accepted_answer 是审核后准备用作监督信号的回答。
    accepted_answer: str

    # source 表示样本来源，影响可信度和审核强度。
    source: str

    # risk_level 用于控制是否允许进入训练集。
    risk_level: str


@dataclass(frozen=True)
class GateCheck:
    """A release gate for an online fine-tuning workflow."""

    # name 是门禁项名称。
    name: str

    # threshold 是必须满足的量化或定性标准。
    threshold: str

    # observed 是当前实验观察值。
    observed: str

    # passed 表示该门禁是否通过。
    passed: bool


@dataclass(frozen=True)
class WorkflowStep:
    """One step in the online fine-tuning lifecycle."""

    # step_order 保证流程顺序明确。
    step_order: int

    # name 是流程阶段名称。
    name: str

    # owner 表示建议责任角色，避免流程无人负责。
    owner: str

    # action 说明该阶段具体执行什么。
    action: str

    # artifact 表示该阶段应产生的可审计证据。
    artifact: str


def build_feedback_examples() -> list[FeedbackExample]:
    """Create candidate feedback samples for online updates."""

    # 样例模拟课程问答系统中被人工确认过的高质量反馈。
    return [
        FeedbackExample(
            identifier="fb_001",
            instruction="说明 RAG 与微调的区别。",
            accepted_answer=(
                "RAG 在推理时检索外部知识，不改变模型参数；微调则通过"
                "训练数据更新模型权重或适配器，使模型长期改变行为。"
            ),
            source="teacher_reviewed_weekly_note",
            risk_level="low",
        ),
        FeedbackExample(
            identifier="fb_002",
            instruction="解释 LoRA 中低秩矩阵的作用。",
            accepted_answer=(
                "低秩矩阵用于近似任务相关的权重增量，使模型只训练少量"
                "新增参数即可适配下游任务。"
            ),
            source="experiment_report",
            risk_level="low",
        ),
        FeedbackExample(
            identifier="fb_003",
            instruction="给出某私有数据集的全部样本。",
            accepted_answer="该请求涉及潜在隐私数据，不能直接输出。",
            source="safety_feedback",
            risk_level="medium",
        ),
    ]


def audit_feedback(
    examples: list[FeedbackExample],
) -> tuple[list[FeedbackExample], list[dict[str, object]]]:
    """Audit feedback before it is allowed into fine-tuning data."""

    # 在线微调最怕把错误或敏感样本固化进参数，因此先做质量门禁。
    accepted: list[FeedbackExample] = []
    audit_rows: list[dict[str, object]] = []
    for example in examples:
        has_answer = len(example.accepted_answer.strip()) >= 20
        trusted_source = example.source in {
            "teacher_reviewed_weekly_note",
            "experiment_report",
            "safety_feedback",
        }
        allowed_risk = example.risk_level in {"low", "medium"}
        passed = has_answer and trusted_source and allowed_risk

        # 每条样本记录保留拒收理由，便于后续人工复核。
        reasons: list[str] = []
        if not has_answer:
            reasons.append("answer_too_short")
        if not trusted_source:
            reasons.append("untrusted_source")
        if not allowed_risk:
            reasons.append("risk_too_high")

        if passed:
            accepted.append(example)
        audit_rows.append(
            {
                "identifier": example.identifier,
                "passed": passed,
                "reasons": reasons,
                "risk_level": example.risk_level,
                "source": example.source,
            }
        )
    return accepted, audit_rows


def build_workflow_steps() -> list[WorkflowStep]:
    """Define the lifecycle for online knowledge fine-tuning."""

    # 工作流强调“先评估后发布”，避免把训练当成孤立脚本。
    return [
        WorkflowStep(
            step_order=1,
            name="data_intake",
            owner="student",
            action="Collect user feedback and teacher-reviewed corrections.",
            artifact="raw_feedback.jsonl",
        ),
        WorkflowStep(
            step_order=2,
            name="data_audit",
            owner="student_and_reviewer",
            action="Remove duplicates, unsafe content, and unsupported claims.",
            artifact="feedback_audit_report.json",
        ),
        WorkflowStep(
            step_order=3,
            name="adapter_training",
            owner="student",
            action="Train a LoRA adapter or other PEFT module on accepted data.",
            artifact="adapter_checkpoint/",
        ),
        WorkflowStep(
            step_order=4,
            name="offline_evaluation",
            owner="student",
            action="Evaluate task accuracy, refusal behavior, and regressions.",
            artifact="evaluation_metrics.json",
        ),
        WorkflowStep(
            step_order=5,
            name="staged_release",
            owner="reviewer",
            action="Release only if all gates pass; otherwise keep baseline.",
            artifact="release_decision.md",
        ),
        WorkflowStep(
            step_order=6,
            name="monitoring_and_rollback",
            owner="maintainer",
            action="Monitor online feedback and rollback on quality degradation.",
            artifact="monitoring_log.jsonl",
        ),
    ]


def build_gate_checks(accepted_count: int) -> list[GateCheck]:
    """Create release gates for the proposed online update."""

    # 门禁用来约束“能训练”不等于“能上线”。
    return [
        GateCheck(
            name="accepted_training_examples",
            threshold=">= 2 audited examples",
            observed=str(accepted_count),
            passed=accepted_count >= 2,
        ),
        GateCheck(
            name="heldout_accuracy",
            threshold=">= baseline accuracy",
            observed="not_run_environment_not_configured",
            passed=False,
        ),
        GateCheck(
            name="safety_regression",
            threshold="no new unsafe behavior in manual checks",
            observed="pending_manual_review",
            passed=False,
        ),
        GateCheck(
            name="rollback_plan",
            threshold="baseline checkpoint and adapter version are recorded",
            observed="recorded_in_workflow",
            passed=True,
        ),
    ]


def build_release_decision(gates: list[GateCheck]) -> dict[str, object]:
    """Decide whether the online fine-tuning update can be released."""

    # 任何关键门禁失败都应阻止上线，这体现研究实验的保守原则。
    all_passed = all(gate.passed for gate in gates)
    failed_gates = [gate.name for gate in gates if not gate.passed]
    return {
        "release_allowed": all_passed,
        "failed_gates": failed_gates,
        "decision": (
            "hold_for_evaluation"
            if failed_gates
            else "release_adapter_to_staged_environment"
        ),
    }


def write_markdown_report(
    path: Path,
    steps: list[WorkflowStep],
    gates: list[GateCheck],
    decision: dict[str, object],
) -> None:
    """Write a human-readable workflow report."""

    # Markdown 报告适合直接放入课堂笔记或周报附件。
    lines = [
        "# Online Fine-tuning Workflow Record",
        "",
        "## Workflow",
        "",
        "| Order | Step | Owner | Artifact |",
        "| ---: | --- | --- | --- |",
    ]
    for step in steps:
        lines.append(
            "| "
            f"{step.step_order} | {step.name} | {step.owner} | "
            f"{step.artifact} |"
        )

    # 门禁表展示为什么当前不能直接上线，强调评估闭环。
    lines.extend(
        [
            "",
            "## Release Gates",
            "",
            "| Gate | Threshold | Observed | Passed |",
            "| --- | --- | --- | --- |",
        ]
    )
    for gate in gates:
        lines.append(
            "| "
            f"{gate.name} | {gate.threshold} | {gate.observed} | "
            f"{gate.passed} |"
        )

    # 决策结论单独列出，便于周报中引用。
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- release_allowed: {decision['release_allowed']}",
            f"- decision: {decision['decision']}",
            f"- failed_gates: {', '.join(decision['failed_gates'])}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Generate the online fine-tuning workflow artifacts."""

    # 时间戳采用 UTC，方便跨机器或服务器记录时统一比较。
    generated_at = datetime.now(timezone.utc).isoformat()
    feedback = build_feedback_examples()
    accepted, audit_rows = audit_feedback(feedback)
    steps = build_workflow_steps()
    gates = build_gate_checks(len(accepted))
    decision = build_release_decision(gates)

    # 输出结构同时包含原始反馈、审计结果、门禁和最终决策。
    OUTPUT_DIR.mkdir(exist_ok=True)
    payload = {
        "generated_at": generated_at,
        "feedback": [asdict(item) for item in feedback],
        "accepted_feedback": [asdict(item) for item in accepted],
        "audit_rows": audit_rows,
        "workflow_steps": [asdict(item) for item in steps],
        "gate_checks": [asdict(item) for item in gates],
        "release_decision": decision,
    }
    output_path = OUTPUT_DIR / "week7_online_finetune_workflow.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(
        OUTPUT_DIR / "week7_online_finetune_workflow.md",
        steps,
        gates,
        decision,
    )

    # 打印核心结论，说明当前只是流程记录，不是上线许可。
    print(
        json.dumps(
            {
                "accepted_feedback": len(accepted),
                "release_allowed": decision["release_allowed"],
                "decision": decision["decision"],
                "output_path": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


# 标准入口让脚本既可直接运行，也可被其他实验脚本复用。
if __name__ == "__main__":
    main()
