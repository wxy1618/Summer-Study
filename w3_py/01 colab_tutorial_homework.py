# # Colab Tutorial 作业练习
#
# 第三周课程作业要求：P10-P12、P22、P28。
#
# 内容包括：
# - 运行环境检查
# - Colab 环境检测
# - 文件读写
# - NumPy / Pandas 基础
# - Matplotlib 可视化
# - PyTorch 可用性检查与简单训练流程

# %%
import os
import platform
import sys
from pathlib import Path

print("Python:", sys.version)
print("Platform:", platform.platform())
print("Current working directory:", Path.cwd())

# %% [markdown]
# ## 1. 检测是否在 Google Colab 中运行

# %%
try:
    import google.colab  # type: ignore

    IN_COLAB = True
except Exception:
    IN_COLAB = False

print("In Google Colab:", IN_COLAB)

if IN_COLAB:
    print("提示：如果课程要求保存到 Google Drive，可以手动运行：")
    print("from google.colab import drive")
    print("drive.mount('/content/drive')")

# %% [markdown]
# ## 2. 文件读写练习

# %%
workspace = Path("week3_colab_outputs")
workspace.mkdir(exist_ok=True)

notes_path = workspace / "colab_homework_notes.txt"
notes_path.write_text(
    "第三周 Colab Tutorial 作业\n"
    "1. 熟悉代码单元格运行\n"
    "2. 熟悉文件读写\n"
    "3. 熟悉数据处理和可视化\n",
    encoding="utf-8",
)

print(notes_path.read_text(encoding="utf-8"))

# %% [markdown]
# ## 3. NumPy 基础操作

# %%
import numpy as np

rng = np.random.default_rng(seed=42)
x = rng.normal(loc=0.0, scale=1.0, size=(8, 3))

print("x shape:", x.shape)
print("mean by column:", x.mean(axis=0))
print("std by column:", x.std(axis=0))

# %% [markdown]
# ## 4. Pandas 表格处理

# %%
import pandas as pd

df = pd.DataFrame(x, columns=["feature_1", "feature_2", "feature_3"])
df["label"] = (df["feature_1"] + df["feature_2"] > 0).astype(int)

csv_path = workspace / "toy_dataset.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")

loaded_df = pd.read_csv(csv_path)
print(loaded_df.head())
print(loaded_df.describe())

# %% [markdown]
# ## 5. Matplotlib 可视化

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
for label_value, group in loaded_df.groupby("label"):
    plt.scatter(group["feature_1"], group["feature_2"], label=f"label={label_value}")
plt.xlabel("feature_1")
plt.ylabel("feature_2")
plt.title("Toy Dataset Scatter Plot")
plt.legend()
plt.tight_layout()

plot_path = workspace / "toy_dataset_scatter.png"
plt.savefig(plot_path, dpi=150)
print("Saved plot:", plot_path)

# %% [markdown]
# ## 6. PyTorch 可用性检查

# %%
try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except Exception as exc:
    TORCH_AVAILABLE = False
    print("PyTorch is not available:", repr(exc))

print("PyTorch available:", TORCH_AVAILABLE)

if TORCH_AVAILABLE:
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

# %% [markdown]
# ## 7. PyTorch 简单训练演示
#
# 说明：本段只依赖 CPU，用于理解 Colab 中运行深度学习代码的基本流程。

# %%
if TORCH_AVAILABLE:
    torch.manual_seed(42)

    features = torch.tensor(loaded_df[["feature_1", "feature_2", "feature_3"]].values, dtype=torch.float32)
    labels = torch.tensor(loaded_df["label"].values.reshape(-1, 1), dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    for epoch in range(80):
        logits = model(features)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"epoch={epoch + 1:03d}, loss={loss.item():.4f}")

    with torch.no_grad():
        probs = torch.sigmoid(model(features))
        preds = (probs >= 0.5).float()
        acc = (preds == labels).float().mean().item()

    print("training accuracy:", round(acc, 4))
else:
    print("跳过 PyTorch 训练演示：当前环境未安装 PyTorch。")