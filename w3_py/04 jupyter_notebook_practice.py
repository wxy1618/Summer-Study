# %% [markdown]
# # Jupyter Notebook 使用练习
#
# 这个文件采用 `# %%` 单元格格式，可在 VS Code、Jupyter、部分 IDE 中按 Notebook 单元格运行。
# 目标是熟悉 Notebook 中“代码 + 文字 + 输出”的学习方式。

# %%
from pathlib import Path

print("Hello, Jupyter-style Python script!")
print("Current file workspace:", Path.cwd())

# %% [markdown]
# ## 1. 创建一组实验数据

# %%
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
hours = np.arange(1, 11)
scores = 50 + hours * 4 + rng.normal(0, 3, size=len(hours))

df = pd.DataFrame({"study_hours": hours, "score": scores.round(2)})
df

# %% [markdown]
# ## 2. 保存和读取数据

# %%
output_dir = Path("week3_jupyter_outputs")
output_dir.mkdir(exist_ok=True)

csv_path = output_dir / "study_score.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")

loaded = pd.read_csv(csv_path)
print(loaded)

# %% [markdown]
# ## 3. 简单可视化

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(loaded["study_hours"], loaded["score"], marker="o")
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Study Hours vs Score")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = output_dir / "study_score_plot.png"
plt.savefig(plot_path, dpi=150)
print("Saved:", plot_path)

# %% [markdown]
# ## 4. Notebook 使用小结
#
# - Markdown 单元格适合写解释。
# - 代码单元格适合运行实验。
# - 输出结果可以直接保留在 Notebook 中。
# - 正式提交前建议重启 kernel 后从上到下重新运行一次。
