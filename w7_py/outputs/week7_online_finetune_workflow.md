# Online Fine-tuning Workflow Record

## Workflow

| Order | Step | Owner | Artifact |
| ---: | --- | --- | --- |
| 1 | data_intake | student | raw_feedback.jsonl |
| 2 | data_audit | student_and_reviewer | feedback_audit_report.json |
| 3 | adapter_training | student | adapter_checkpoint/ |
| 4 | offline_evaluation | student | evaluation_metrics.json |
| 5 | staged_release | reviewer | release_decision.md |
| 6 | monitoring_and_rollback | maintainer | monitoring_log.jsonl |

## Release Gates

| Gate | Threshold | Observed | Passed |
| --- | --- | --- | --- |
| accepted_training_examples | >= 2 audited examples | 2 | True |
| heldout_accuracy | >= baseline accuracy | not_run_environment_not_configured | False |
| safety_regression | no new unsafe behavior in manual checks | pending_manual_review | False |
| rollback_plan | baseline checkpoint and adapter version are recorded | recorded_in_workflow | True |

## Decision

- release_allowed: False
- decision: hold_for_evaluation
- failed_gates: heldout_accuracy, safety_regression