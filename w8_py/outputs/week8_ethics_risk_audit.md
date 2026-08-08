# Ethics and Risk Audit

| Area | Severity | Likelihood | Score | Mitigation | Owner |
| --- | ---: | ---: | ---: | --- | --- |
| data_provenance | 3 | 3 | 9 | Record dataset origin, license, collection date and scope. | student |
| bias_and_fairness | 4 | 3 | 12 | Create paired test cases and compare subgroup error rates. | student_reviewer |
| privacy | 5 | 3 | 15 | Mask personal identifiers and restrict raw document access. | data_owner |
| misinformation | 4 | 2 | 8 | Require evidence spans and allow abstention on weak retrieval. | model_developer |
| copyright | 3 | 2 | 6 | Store source URLs and avoid redistributing restricted content. | project_owner |
| explainability | 3 | 3 | 9 | Label explanations as diagnostic aids, not causal proof. | model_developer |
| deployment | 4 | 2 | 8 | Record baseline version, adapter version and rollback trigger. | maintainer |

## Decision

- release_allowed: False
- max_risk_score: 15
- high_risk_count: 2
- decision_note: Hold release until high-risk items have stronger mitigation evidence and reviewer approval.