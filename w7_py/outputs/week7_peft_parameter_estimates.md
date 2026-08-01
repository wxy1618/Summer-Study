# PEFT Parameter Estimates

| Method | Trainable Parameters | % of Full | FP16 MB | Note |
| --- | ---: | ---: | ---: | --- |
| Full fine-tuning | 108,375,552 | 100.000000% | 206.710 | Update all model weights. |
| Prompt Tuning | 49,152 | 0.045353% | 0.094 | Train input soft prompt only. |
| P-Tuning v2 | 1,179,648 | 1.088482% | 2.250 | Train deep prompts across layers. |
| LoRA q/v | 294,912 | 0.272120% | 0.562 | Train rank-8 adapters on q and v. |
| LoRA q/k/v/o | 589,824 | 0.544241% | 1.125 | Train rank-8 adapters on q/k/v/o. |