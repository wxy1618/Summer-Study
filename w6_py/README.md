# 第六周作业代码说明

## 文件列表

```text
第六周作业代码/
├── README.md
├── 01_translation_seq2seq_attention.py
├── 02_question_answering_span_baseline.py
├── 03_nlg_decoding_strategies.py
├── 04_coreference_resolution_baseline.py
├── 05_transe_embedding_tuning.py
├── 06_glove_embedding_tuning.py
├── 07_sentiment_attention_classifier.py
├── 08_relation_extraction_toy_model.py
└── 09_llm_prompt_evaluation_offline.py
```

## 运行建议

1. `01_translation_seq2seq_attention.py`

   小型 Seq2Seq + Attention 翻译实验，需要 PyTorch。

2. `02_question_answering_span_baseline.py`

   抽取式问答 span 选择基线，只使用 Python 标准库。

3. `03_nlg_decoding_strategies.py`

   Trigram 语言模型与多种解码策略，只使用 Python 标准库。

4. `04_coreference_resolution_baseline.py`

   共指消解规则/打分基线与调参，只使用 Python 标准库。

5. `05_transe_embedding_tuning.py`

   TransE 知识图谱嵌入与超参数调优，需要 PyTorch。

6. `06_glove_embedding_tuning.py`

   GloVe 共现词嵌入训练与窗口大小调优，需要 PyTorch。

7. `07_sentiment_attention_classifier.py`

   注意力池化情感分析模型，需要 PyTorch。

8. `08_relation_extraction_toy_model.py`

   实体标记关系抽取分类模型，需要 PyTorch。

9. `09_llm_prompt_evaluation_offline.py`

   大模型 prompt 离线评估框架，只使用 Python 标准库，不调用外部 API。

## 示例命令

```powershell
python .\01_translation_seq2seq_attention.py --epochs 30
python .\02_question_answering_span_baseline.py
python .\03_nlg_decoding_strategies.py
python .\04_coreference_resolution_baseline.py
python .\05_transe_embedding_tuning.py --epochs 80
python .\06_glove_embedding_tuning.py --epochs 80
python .\07_sentiment_attention_classifier.py --epochs 40
python .\08_relation_extraction_toy_model.py --epochs 45
python .\09_llm_prompt_evaluation_offline.py
```

所有脚本运行后会把结果写入当前目录下的 `outputs/`。脚本不会安装、修改或配置任何环境。
