# 第七周作业代码说明

## 文件列表

```text
第七周作业代码
├── README.md
├── 01_t5_text_to_text_tasks.py
├── 02_instruction_tuning_dataset_builder.py
├── 03_peft_method_parameter_accounting.py
├── 04_offline_rag_pipeline.py
├── 05_hf_text_classification_finetune.py
├── 06_hf_question_answering_feature_builder.py
├── 07_hf_question_answering_finetune.py
├── 08_bert_roberta_experiment_planner.py
└── 09_online_finetune_workflow_recorder.py
```

## 运行建议

1. `01_t5_text_to_text_tasks.py`

   T5 风格的 text-to-text 数据组织与任务评估示例，只使用 Python 标准库。

2. `02_instruction_tuning_dataset_builder.py`

   指令微调数据集构造、质量校验、训练/验证划分与 JSONL 输出，只使用标准库。

3. `03_peft_method_parameter_accounting.py`

   对比全量微调、Prompt Tuning、P-Tuning 和 LoRA 的参数量与存储成本，只使用标准库。

4. `04_offline_rag_pipeline.py`

   离线 RAG 原型：文档切分、TF-IDF 检索、证据 prompt 与抽取式回答，只使用标准库。

5. `05_hf_text_classification_finetune.py`

   HuggingFace 文本分类微调流程。未安装依赖时会给出 dry-run 计划，不会配置环境。

6. `06_hf_question_answering_feature_builder.py`

   抽取式问答的字符 span、token offset 和滑动窗口特征构造，只使用标准库。

7. `07_hf_question_answering_finetune.py`

   HuggingFace 问答微调流程。未安装依赖时会输出可复现实验说明，不会配置环境。

8. `08_bert_roberta_experiment_planner.py`

   BERT 与 RoBERTa 微调实验矩阵生成，用于记录选做实验计划，只使用标准库。

9. `09_online_finetune_workflow_recorder.py`

   在线知识微调闭环记录，包括数据审核、训练评估、发布门禁与回滚条件，只使用标准库。

## 示例命令

```powershell
python .\01_t5_text_to_text_tasks.py
python .\02_instruction_tuning_dataset_builder.py --seed 7
python .\03_peft_method_parameter_accounting.py
python .\04_offline_rag_pipeline.py --top-k 3
python .\05_hf_text_classification_finetune.py --dry-run
python .\06_hf_question_answering_feature_builder.py
python .\07_hf_question_answering_finetune.py --dry-run
python .\08_bert_roberta_experiment_planner.py
python .\09_online_finetune_workflow_recorder.py
```

所有标准库脚本会把结果写入本目录下的 `outputs/`。代码仅生成作业文件和实验记录，不安装、不修改、不配置任何运行环境。
