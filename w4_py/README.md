# 第四周作业代码说明

## 文件列表

```text
第四周作业代码/
├── README.md
├── 01_knn_language_model.py
├── 02_self_attention_homework.py
├── 03_pytorch_complete_tutorial.py
├── 04_transformer_homework.py
├── 05_diffusion_toy_demo.py
├── 06_gan_image_homework.py
└── 07_paddle_initial_practice.py
```

## 运行建议

1. `01_knn_language_model.py`

   纯 Python 标准库实现的小型 KNN-LM 演示，不需要第三方依赖。

2. `02_self_attention_homework.py`

   Homework 4 的 Self-attention 序列分类练习，需要 PyTorch。

3. `03_pytorch_complete_tutorial.py`

   串联 Dataset、DataLoader、模型、训练、验证、保存、加载与推理，需要 PyTorch。

4. `04_transformer_homework.py`

   Homework 5 的 Transformer 序列到序列练习，需要 PyTorch。

5. `05_diffusion_toy_demo.py`

   二维 Diffusion 原理演示，需要 PyTorch；若已安装 Matplotlib，会额外保存散点图。

6. `06_gan_image_homework.py`

   Homework 6 的轻量 GAN 图像生成练习，需要 PyTorch。训练数据由程序生成，不需要联网下载。

7. `07_paddle_initial_practice.py`

   PaddlePaddle 的 Tensor、自动求导、训练、验证与模型保存练习。未安装 PaddlePaddle 时只会给出提示，不会尝试安装。

## 示例命令

```powershell
python .\01_knn_language_model.py
python .\02_self_attention_homework.py --epochs 8
python .\03_pytorch_complete_tutorial.py --epochs 12
python .\04_transformer_homework.py --epochs 12
python .\05_diffusion_toy_demo.py --train-steps 400
python .\06_gan_image_homework.py --epochs 15
python .\07_paddle_initial_practice.py --epochs 12
```

运行需要训练的脚本后，模型参数和生成结果会写入当前目录下的 `outputs/`。该目录由脚本运行时创建。
