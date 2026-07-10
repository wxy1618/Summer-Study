# 第五周作业代码说明

## 文件列表

```text
第五周作业代码/
├── README.md
├── 01_word2vec_skipgram_negative_sampling.py
├── 02_dependency_transition_parser.py
├── 03_rnn_language_model.py
├── 04_transformer_masked_lm.py
├── 05_stage1_nlp_text_classifier.py
├── 06_midlevel_data_api_demo.py
└── 07_toy_multimodal_retrieval.py
```

## 运行建议

1. `01_word2vec_skipgram_negative_sampling.py`

   Skip-gram + Negative Sampling 词向量实验，需要 PyTorch。

2. `02_dependency_transition_parser.py`

   Arc-standard 依存句法分析演示，只使用 Python 标准库即可。

3. `03_rnn_language_model.py`

   GRU 词级语言模型，记录 loss、perplexity 和采样文本，需要 PyTorch。

4. `04_transformer_masked_lm.py`

   小型 Transformer Encoder 的 Masked Language Modeling 预训练实验，需要 PyTorch。

5. `05_stage1_nlp_text_classifier.py`

   阶段一 NLP 文本分类流程：分词、词表、padding、模型训练、评估和结果记录，需要 PyTorch。

6. `06_midlevel_data_api_demo.py`

   fastbook mid-level data API 思想复现：Transform、Pipeline、setup、decode，只使用 Python 标准库。

7. `07_toy_multimodal_retrieval.py`

   图文对比学习和检索小实验，用于理解多模态表示对齐，需要 PyTorch。

## 示例命令

```powershell
python .\01_word2vec_skipgram_negative_sampling.py --epochs 40
python .\02_dependency_transition_parser.py
python .\03_rnn_language_model.py --epochs 20
python .\04_transformer_masked_lm.py --epochs 30
python .\05_stage1_nlp_text_classifier.py --epochs 35
python .\06_midlevel_data_api_demo.py
python .\07_toy_multimodal_retrieval.py --epochs 80
```

