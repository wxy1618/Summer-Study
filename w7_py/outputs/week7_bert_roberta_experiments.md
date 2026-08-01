# BERT/RoBERTa Fine-tuning Experiment Matrix

| ID | Model | Task | Strategy | LR | Epochs | Batch | Metric |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| w7_exp_01 | bert-base-uncased | text_classification | full_finetune | 2e-05 | 2 | 8 | macro_f1 |
| w7_exp_02 | bert-base-uncased | text_classification | full_finetune | 5e-05 | 2 | 8 | macro_f1 |
| w7_exp_03 | bert-base-uncased | text_classification | freeze_encoder_head_only | 2e-05 | 2 | 8 | macro_f1 |
| w7_exp_04 | bert-base-uncased | text_classification | freeze_encoder_head_only | 5e-05 | 2 | 8 | macro_f1 |
| w7_exp_05 | bert-base-uncased | text_classification | lora_adapter_optional | 2e-05 | 2 | 8 | macro_f1 |
| w7_exp_06 | bert-base-uncased | text_classification | lora_adapter_optional | 5e-05 | 2 | 8 | macro_f1 |
| w7_exp_07 | bert-base-uncased | question_answering | full_finetune | 2e-05 | 2 | 8 | exact_match_f1 |
| w7_exp_08 | bert-base-uncased | question_answering | full_finetune | 5e-05 | 2 | 8 | exact_match_f1 |
| w7_exp_09 | bert-base-uncased | question_answering | freeze_encoder_head_only | 2e-05 | 2 | 8 | exact_match_f1 |
| w7_exp_10 | bert-base-uncased | question_answering | freeze_encoder_head_only | 5e-05 | 2 | 8 | exact_match_f1 |
| w7_exp_11 | bert-base-uncased | question_answering | lora_adapter_optional | 2e-05 | 2 | 8 | exact_match_f1 |
| w7_exp_12 | bert-base-uncased | question_answering | lora_adapter_optional | 5e-05 | 2 | 8 | exact_match_f1 |
| w7_exp_13 | roberta-base | text_classification | full_finetune | 2e-05 | 2 | 8 | macro_f1 |
| w7_exp_14 | roberta-base | text_classification | full_finetune | 5e-05 | 2 | 8 | macro_f1 |
| w7_exp_15 | roberta-base | text_classification | freeze_encoder_head_only | 2e-05 | 2 | 8 | macro_f1 |
| w7_exp_16 | roberta-base | text_classification | freeze_encoder_head_only | 5e-05 | 2 | 8 | macro_f1 |
| w7_exp_17 | roberta-base | text_classification | lora_adapter_optional | 2e-05 | 2 | 8 | macro_f1 |
| w7_exp_18 | roberta-base | text_classification | lora_adapter_optional | 5e-05 | 2 | 8 | macro_f1 |
| w7_exp_19 | roberta-base | question_answering | full_finetune | 2e-05 | 2 | 8 | exact_match_f1 |
| w7_exp_20 | roberta-base | question_answering | full_finetune | 5e-05 | 2 | 8 | exact_match_f1 |
| w7_exp_21 | roberta-base | question_answering | freeze_encoder_head_only | 2e-05 | 2 | 8 | exact_match_f1 |
| w7_exp_22 | roberta-base | question_answering | freeze_encoder_head_only | 5e-05 | 2 | 8 | exact_match_f1 |
| w7_exp_23 | roberta-base | question_answering | lora_adapter_optional | 2e-05 | 2 | 8 | exact_match_f1 |
| w7_exp_24 | roberta-base | question_answering | lora_adapter_optional | 5e-05 | 2 | 8 | exact_match_f1 |