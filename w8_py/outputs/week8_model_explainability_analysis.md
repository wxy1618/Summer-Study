# Model Explanation Analysis

说明：本报告使用透明词典基线展示解释流程。对于深度模型，attention 或 saliency 只能作为诊断线索，仍需 deletion、counterfactual 等忠实性检查。

| Text | Prediction | Score | Contributions | Counterfactual |
| --- | --- | ---: | --- | --- |
| The semantic search demo is clear and useful. | positive | 2 | {'clear': 1, 'useful': 1} | negative: the semantic search demo is confusing and poorly |
| The retrieval result is confusing and poorly ranked. | negative | -2 | {'confusing': -1, 'poorly': -1} | positive: the retrieval result is clear and useful ranked |
| The ethical audit is helpful but the model ignores privacy. | negative | -1 | {'helpful': 1, 'ignores': -1, 'privacy': -1} | negative: the ethical audit is ignores but the model helpful privacy |