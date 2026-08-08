# Week 8 Stage Four Experiment Summary

| Task | Artifact | Headline Metric | Note |
| --- | --- | --- | --- |
| document_vqa | week8_doc_vqa_workflow.json | {'accuracy': 1.0} | Records OCR tokens, evidence boxes, answers and confidence. |
| ethics_audit | week8_ethics_risk_audit.json | {'release_allowed': False, 'max_risk_score': 15, 'high_risk_count': 2, 'decision_note': 'Hold release until high-risk items have stronger mitigation evidence and reviewer approval.'} | Records release gates for social and ethical risks. |
| knowledge_augmentation | week8_knowledge_augmentation_concept_map.json | methods=5, metrics=5 | Records methods, metrics and conceptual relations. |
| model_explainability | week8_model_explainability_analysis.json | counterfactual_changed=2/3 | Records token contributions, deletion tests and counterfactuals. |
| semantic_search | week8_semantic_search_dryrun.json | {'recall_at_3': 1.0, 'mrr': 1.0} | Records retrieval ranking quality and PaddleNLP availability. |
| sentiment_analysis | week8_sentiment_analysis_dryrun.json | {'accuracy': 1.0, 'precision_positive': 1.0, 'recall_positive': 1.0, 'f1_positive': 1.0} | Records dry-run sentiment predictions and PaddleNLP plan. |