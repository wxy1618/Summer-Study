# %% [markdown]
# # PaddlePaddle + Jupyter Notebook 作业教程
#
# 本文件参考 `01 colab_tutorial_homework.py` 的作业结构，把原来的 Colab/PyTorch
# 练习改成适合当前环境的版本：
#
# - Windows 主机 + VMware Ubuntu 虚拟机
# - Ubuntu 中使用 Miniconda 管理 Python 环境
# - 在 `paddle-cpu` 环境中安装 PaddlePaddle CPU 版
# - 在 Windows 浏览器中打开 Ubuntu 里启动的 Jupyter Notebook
#
# ## 一、环境搭建命令
#
# 在 Ubuntu 终端执行：
#
# ```bash
# conda create -n paddle-cpu --override-channels -c conda-forge python=3.12 pip -y
# conda activate paddle-cpu
# python -m pip install --upgrade pip setuptools wheel
# python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
#   "numpy==1.26.4" pandas matplotlib notebook ipykernel
# python -m pip install paddlepaddle==3.3.0 \
#   -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
#   --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
# python -m ipykernel install --user --name paddle-cpu --display-name "Python (paddle-cpu)"
# jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
# ```
#
# 在 Windows 浏览器中打开 Jupyter 输出的链接时，把 `0.0.0.0` 或 `localhost`
# 替换成 Ubuntu 虚拟机 IP，例如：
#
# ```text
# http://192.168.34.130:8888/?token=...
# ```
#
# ## 二、运行目标
#
# 从上到下运行本 notebook，完成环境检查、文件读写、NumPy/Pandas、Matplotlib、
# PaddlePaddle 张量操作、自动求导和一个简单线性回归训练实验。

# %%
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


OUTPUT_DIR = Path("week3_paddle_jupyter_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

section("1. Python 与运行环境检查")
print("Python:", sys.version)
print("Platform:", platform.platform())
print("Current working directory:", Path.cwd())
print("Output directory:", OUTPUT_DIR.resolve())
print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV", "unknown"))
print("Conda prefix:", os.environ.get("CONDA_PREFIX", "unknown"))

# %% [markdown]
# ## 2. 检查是否在 Google Colab 中运行
#
# 原作业是 Colab Tutorial，这里保留这个检查。当前我们是在 Ubuntu 虚拟机中运行，
# 所以 `In Google Colab` 应该是 `False`。

# %%
try:
    import google.colab  # type: ignore  # noqa: F401

    IN_COLAB = True
except Exception:
    IN_COLAB = False

print("In Google Colab:", IN_COLAB)
print("Current task environment: VMware Ubuntu + Jupyter Notebook")

# %% [markdown]
# ## 3. 文件读写练习
#
# 创建一个输出目录，写入文本文件，再读取回来。这一步用于熟悉 notebook 中的文件路径。

# %%
section("3. 文件读写练习")

notes_path = OUTPUT_DIR / "paddle_homework_notes.txt"
notes_path.write_text(
    "第三周 Jupyter/PaddlePaddle 作业\n"
    "1. 使用 Miniconda 创建 paddle-cpu 环境\n"
    "2. 在 Ubuntu 虚拟机中启动 Jupyter Notebook\n"
    "3. 在 Windows 浏览器中访问 Notebook 页面\n"
    "4. 使用 PaddlePaddle 完成张量、自动求导和简单训练实验\n",
    encoding="utf-8",
)

print(notes_path.read_text(encoding="utf-8"))

# %% [markdown]
# ## 4. NumPy 基础操作
#
# NumPy 常用于数据构造、预处理和结果检查。深度学习框架中的 tensor 经常可以和
# NumPy array 相互转换。

# %%
section("4. NumPy 基础操作")

import numpy as np

rng = np.random.default_rng(seed=42)
x_np = rng.normal(loc=0.0, scale=1.0, size=(12, 3)).astype("float32")

print("x_np shape:", x_np.shape)
print("mean by column:", x_np.mean(axis=0))
print("std by column:", x_np.std(axis=0))
print("first 3 rows:\n", x_np[:3])

# %% [markdown]
# ## 5. Pandas 表格处理
#
# 把 NumPy 数据转换成表格，保存为 CSV，再读取回来。这部分对应参考作业中的
# Pandas 基础练习。

# %%
section("5. Pandas 表格处理")

import pandas as pd

df = pd.DataFrame(x_np, columns=["feature_1", "feature_2", "feature_3"])
df["label"] = (df["feature_1"] + df["feature_2"] > 0).astype("int64")

csv_path = OUTPUT_DIR / "toy_dataset.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")

loaded_df = pd.read_csv(csv_path)
print(loaded_df.head())
print(loaded_df.describe())
print("Saved CSV:", csv_path)

# %% [markdown]
# ## 6. Matplotlib 可视化
#
# 绘制简单散点图，并保存到输出目录。Jupyter 中可以直接显示图像；作为脚本运行时，
# 也可以通过保存的 PNG 文件查看结果。

# %%
section("6. Matplotlib 可视化")

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
for label_value, group in loaded_df.groupby("label"):
    plt.scatter(group["feature_1"], group["feature_2"], label=f"label={label_value}")
plt.xlabel("feature_1")
plt.ylabel("feature_2")
plt.title("Toy Dataset Scatter Plot")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

scatter_path = OUTPUT_DIR / "toy_dataset_scatter.png"
plt.savefig(scatter_path, dpi=150)
plt.show()
print("Saved plot:", scatter_path)

# %% [markdown]
# ## 7. PaddlePaddle 环境验证
#
# 这一节检查 PaddlePaddle 是否安装成功，并确认当前使用的是 CPU。

# %%
section("7. PaddlePaddle 环境验证")

try:
    import paddle
    import paddle.nn as nn

    PADDLE_AVAILABLE = True
except ModuleNotFoundError as exc:
    PADDLE_AVAILABLE = False
    raise ModuleNotFoundError(
        "当前环境没有安装 PaddlePaddle。请先激活 paddle-cpu 环境，"
        "或运行 install_paddle_env.sh 完成安装。"
    ) from exc

paddle.seed(42)
paddle.set_device("cpu")

print("PaddlePaddle version:", paddle.__version__)
print("Paddle device:", paddle.get_device())
paddle.utils.run_check()

# %% [markdown]
# ## 8. PaddlePaddle 张量操作
#
# PaddlePaddle 中的 `Tensor` 类似于 NumPy array，也类似于 PyTorch Tensor。

# %%
section("8. PaddlePaddle 张量操作")

a = paddle.to_tensor([[1, 2, 3], [4, 5, 6]], dtype="float32")
b = paddle.ones([2, 3], dtype="float32")
c = paddle.arange(1, 7, dtype="float32").reshape([3, 2])

print("a:\n", a)
print("b:\n", b)
print("a + b:\n", a + b)
print("a mean:", float(paddle.mean(a).numpy()))
print("a @ c:\n", paddle.matmul(a, c))
print("a numpy:\n", a.numpy())

# %% [markdown]
# ## 9. 自动求导
#
# 自动求导是深度学习训练的基础。下面计算：
#
# \[
# y = x^2 + 3x + 1
# \]
#
# 当 \(x=2\) 时，理论导数是 \(2x+3=7\)。

# %%
section("9. PaddlePaddle 自动求导")

x = paddle.to_tensor([2.0], dtype="float32")
x.stop_gradient = False

y = x**2 + 3 * x + 1
y.backward()

print("x:", x.numpy())
print("y:", y.numpy())
print("dy/dx:", x.grad.numpy())

# %% [markdown]
# ## 10. 简单线性回归实验
#
# 使用 PaddlePaddle 训练一个一层线性模型，拟合：
#
# \[
# y = 3x + 2 + noise
# \]
#
# 训练完成后，模型学到的权重应接近 3，偏置应接近 2。

# %%
section("10. PaddlePaddle 简单线性回归实验")

rng = np.random.default_rng(seed=2026)
train_x_np = np.linspace(-3, 3, 120, dtype="float32").reshape(-1, 1)
noise_np = rng.normal(0, 0.25, size=train_x_np.shape).astype("float32")
train_y_np = 3.0 * train_x_np + 2.0 + noise_np

train_x = paddle.to_tensor(train_x_np, dtype="float32")
train_y = paddle.to_tensor(train_y_np, dtype="float32")

model = nn.Linear(in_features=1, out_features=1)
loss_fn = nn.MSELoss()
optimizer = paddle.optimizer.SGD(learning_rate=0.05, parameters=model.parameters())

loss_history: list[float] = []

for epoch in range(200):
    pred_y = model(train_x)
    loss = loss_fn(pred_y, train_y)

    loss.backward()
    optimizer.step()
    optimizer.clear_grad()

    loss_value = float(loss.numpy())
    loss_history.append(loss_value)

    if (epoch + 1) % 40 == 0:
        print(f"epoch={epoch + 1:03d}, loss={loss_value:.6f}")

with paddle.no_grad():
    pred_y = model(train_x)
    final_loss = float(loss_fn(pred_y, train_y).numpy())

learned_weight = float(model.weight.numpy().reshape(-1)[0])
learned_bias = float(model.bias.numpy().reshape(-1)[0])

print("final loss:", round(final_loss, 6))
print("learned weight:", round(learned_weight, 4))
print("learned bias:", round(learned_bias, 4))

# %% [markdown]
# ## 11. 保存训练结果图像
#
# 保存损失曲线和拟合效果图，作为实验结果材料。

# %%
section("11. 保存训练结果图像")

plt.figure(figsize=(6, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("PaddlePaddle Linear Regression Loss")
plt.grid(alpha=0.25)
plt.tight_layout()

loss_plot_path = OUTPUT_DIR / "paddle_linear_regression_loss.png"
plt.savefig(loss_plot_path, dpi=150)
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(train_x_np, train_y_np, s=18, alpha=0.75, label="training data")
plt.plot(train_x_np, pred_y.numpy(), color="red", linewidth=2, label="model prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.title("PaddlePaddle Linear Regression Fit")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

fit_plot_path = OUTPUT_DIR / "paddle_linear_regression_fit.png"
plt.savefig(fit_plot_path, dpi=150)
plt.show()

print("Saved loss plot:", loss_plot_path)
print("Saved fit plot:", fit_plot_path)

# %% [markdown]
# ## 12. 作业小结
#
# 本教程完成了：
#
# - Ubuntu 虚拟机中的 PaddlePaddle/Jupyter 环境搭建说明
# - Python、Conda、PaddlePaddle 环境验证
# - 文件读写、NumPy、Pandas、Matplotlib 基础练习
# - PaddlePaddle 张量操作
# - PaddlePaddle 自动求导
# - PaddlePaddle 简单线性回归训练与可视化

# %%
section("12. 作业小结")

summary_path = OUTPUT_DIR / "paddle_homework_summary.txt"
summary_path.write_text(
    "\n".join(
        [
            "PaddlePaddle + Jupyter Notebook 作业运行完成",
            f"Python: {sys.version.split()[0]}",
            f"PaddlePaddle: {paddle.__version__}",
            f"Device: {paddle.get_device()}",
            f"Final loss: {final_loss:.6f}",
            f"Learned weight: {learned_weight:.4f}",
            f"Learned bias: {learned_bias:.4f}",
            f"Output directory: {OUTPUT_DIR.resolve()}",
        ]
    )
    + "\n",
    encoding="utf-8",
)

print(summary_path.read_text(encoding="utf-8"))
print("All tasks finished successfully.")
