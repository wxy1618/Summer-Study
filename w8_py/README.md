# 第八周作业代码说明

## 文件列表

```text
第八周作业代码
├── README.md
├── 01_knowledge_augmentation_concept_map.py
├── 02_paddlenlp_semantic_search_dryrun.py
├── 03_paddlenlp_doc_vqa_workflow.py
├── 04_paddlenlp_sentiment_analysis_dryrun.py
├── 05_ethics_and_risk_audit.py
├── 06_model_explainability_analysis.py
└── 07_stage4_experiment_report_builder.py
```

## 运行建议

1. `01_knowledge_augmentation_concept_map.py`

   整理知识增强方法、适用场景、评价指标和方法关系，只使用 Python 标准库。

2. `02_paddlenlp_semantic_search_dryrun.py`

   语义检索系统 dry-run。默认使用 TF-IDF 基线模拟召回与 MRR 评价，不配置 PaddleNLP 环境。

3. `03_paddlenlp_doc_vqa_workflow.py`

   智能问答/DocVQA 工作流。默认使用结构化 OCR token 和规则证据定位模拟文档问答。

4. `04_paddlenlp_sentiment_analysis_dryrun.py`

   情感分析 dry-run。默认使用标准库词典基线记录 accuracy、precision、recall、F1 和错误样例。

5. `05_ethics_and_risk_audit.py`

   NLP/大模型伦理风险审计，覆盖数据来源、偏见、隐私、安全、版权和上线门禁。

6. `06_model_explainability_analysis.py`

   模型分析与解释性实验，包含词级贡献、删除实验、反事实替换和解释可靠性说明。

7. `07_stage4_experiment_report_builder.py`

   汇总本目录 `outputs/` 中的阶段四实验记录，生成 JSON 和 Markdown 总结报告。

## 示例命令

```powershell
python .\01_knowledge_augmentation_concept_map.py
python .\02_paddlenlp_semantic_search_dryrun.py --top-k 3
python .\03_paddlenlp_doc_vqa_workflow.py
python .\04_paddlenlp_sentiment_analysis_dryrun.py
python .\05_ethics_and_risk_audit.py
python .\06_model_explainability_analysis.py
python .\07_stage4_experiment_report_builder.py
```

所有脚本默认把结果写入本目录下的 `outputs/`。
