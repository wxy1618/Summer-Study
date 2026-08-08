# Knowledge Augmentation Concept Map

## Methods

| Method | Stage | Knowledge Type | Benefit | Limitation |
| --- | --- | --- | --- | --- |
| domain_adaptive_pretraining | pretraining | parametric | Absorbs domain language patterns and frequent facts. | Expensive to update and difficult to cite. |
| supervised_instruction_tuning | fine_tuning | parametric_behavior | Teaches the model task format and response style. | May overfit small or noisy instruction data. |
| retrieval_augmented_generation | inference | non_parametric | Injects fresh and auditable external evidence. | Answer quality is bounded by retrieval quality. |
| knowledge_graph_grounding | inference_or_training | structured_external | Provides explicit entities, relations, and paths. | Coverage and schema construction can be costly. |
| tool_or_database_calling | inference | external_tool | Allows models to query authoritative structured sources. | Requires robust tool routing and permission control. |

## Metrics

| Metric | Target Stage | Interpretation |
| --- | --- | --- |
| Recall@K | retrieval | Measures whether relevant evidence appears in top-K. |
| MRR | retrieval | Rewards relevant evidence being ranked earlier. |
| Faithfulness | generation | Checks whether the answer is supported by evidence. |
| Citation Accuracy | generation | Checks whether cited sources truly support claims. |
| Risk Gate Pass Rate | governance | Checks whether data, safety, and rollback gates pass. |

## Relations

- `retrieval_augmented_generation` -> `Recall@K`: retrieval quality must be measured before judging answers
- `retrieval_augmented_generation` -> `Faithfulness`: generation should be constrained by retrieved evidence
- `domain_adaptive_pretraining` -> `supervised_instruction_tuning`: domain knowledge can precede task-specific behavior tuning
- `knowledge_graph_grounding` -> `Citation Accuracy`: explicit sources make answer verification easier
- `tool_or_database_calling` -> `Risk Gate Pass Rate`: external actions require permission and monitoring gates