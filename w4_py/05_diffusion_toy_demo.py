"""
在二维点集上演示 Diffusion Model 的训练与反向采样。

该示例保留 DDPM 的关键步骤，但把图像替换为二维高斯混合数据，使 CPU
也能在较短时间内观察前向加噪和逐步去噪。

本模块包含：
- 二维高斯混合数据生成（八个簇）
- 正弦时间步嵌入
- 噪声预测网络（MLP）
- 前向扩散加噪过程（闭式采样）
- 训练循环
- 反向采样（DDPM 去噪）
- 结果保存（CSV 和可选 PNG）

"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

# 尝试导入 PyTorch，若未安装则给出友好提示
try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:
    raise SystemExit(
        "当前环境未安装 PyTorch；本脚本不会自动安装依赖。"
    ) from exc


# ---------- 全局常量 ----------
DIFFUSION_STEPS = 80          # 扩散步数（前向/反向总步数）
TIME_EMBEDDING_SIZE = 32      # 时间步嵌入的维度


# ---------- 辅助函数 ----------
def parse_args() -> argparse.Namespace:
    """
    读取训练步数和输出目录。

    返回：
        argparse.Namespace: 包含 train_steps, batch_size, output_dir
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=400,
                        help="训练迭代步数（默认：400）")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="批次大小（默认：256）")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="保存结果的目录（默认：outputs）")
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """
    固定随机数生成器，确保实验可复现。

    参数：
        seed (int): 随机种子
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- 数据生成 ----------
def make_mixture_points(
    sample_count: int,
    device: torch.device,
) -> torch.Tensor:
    """
    从环形排列的八个高斯分量中采样二维点。

    八个簇均匀分布在半径为 3 的圆上，每个簇内标准差为 0.18，
    形成清晰可辨的环状分布，便于观察生成质量。

    参数：
        sample_count (int): 采样点数
        device (torch.device): 数据所在设备

    返回：
        torch.Tensor: 形状 (sample_count, 2) 的二维点集
    """
    # 生成 8 个均匀分布在 360° 上的中心点（角度从 0 到 2π，间隔 2π/8）
    angles = torch.linspace(0.0, 2.0 * math.pi, 9, device=device)[:-1]  # 去掉最后一个重复的 2π
    centers = torch.stack([angles.cos(), angles.sin()], dim=1) * 3.0    # 半径为 3

    # 为每个样本随机选择一个簇
    component_ids = torch.randint(0, centers.size(0), (sample_count,), device=device)

    # 生成高斯噪声，标准差 0.18，使簇内紧凑但不重叠
    noise = torch.randn(sample_count, 2, device=device) * 0.18
    return centers[component_ids] + noise


# ---------- 时间步嵌入 ----------
def sinusoidal_time_embedding(time_steps: torch.Tensor) -> torch.Tensor:
    """
    把离散时间步转换为正弦和余弦特征。

    采用 Transformer 中的位置编码方式，将标量时间步映射为高维向量，
    使得模型能够感知当前处于哪个扩散阶段。

    参数：
        time_steps (torch.Tensor): 形状 (batch_size,) 的时间步整数（0 ~ DIFFUSION_STEPS-1）

    返回：
        torch.Tensor: 形状 (batch_size, TIME_EMBEDDING_SIZE) 的嵌入向量
    """
    half_size = TIME_EMBEDDING_SIZE // 2
    # 计算频率衰减因子：从 1 到 1/10000 的对数分布
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half_size, device=time_steps.device)
        / (half_size - 1)
    )
    # 计算角度：时间步 * 频率
    angles = time_steps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    # 拼接正弦和余弦，形成位置编码
    return torch.cat([angles.sin(), angles.cos()], dim=1)


# ---------- 噪声预测模型 ----------
class NoisePredictor(nn.Module):
    """
    根据带噪二维点和时间步预测高斯噪声。

    网络结构：MLP，输入为二维坐标 + 时间嵌入，输出预测的噪声（二维）。
    激活函数使用 SiLU（Swish），适合扩散模型。
    """

    def __init__(self) -> None:
        super().__init__()
        input_size = 2 + TIME_EMBEDDING_SIZE   # 坐标 (2) + 时间嵌入 (TIME_EMBEDDING_SIZE)
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 2),                 # 输出噪声，与输入点形状相同
        )

    def forward(
        self,
        noisy_points: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> torch.Tensor:
        """
        预测给定带噪点对应的噪声。

        参数：
            noisy_points (torch.Tensor): 形状 (batch_size, 2) 的带噪二维点
            time_steps (torch.Tensor): 形状 (batch_size,) 的时间步

        返回：
            torch.Tensor: 形状 (batch_size, 2) 的预测噪声
        """
        # 生成时间特征并拼接到输入
        time_features = sinusoidal_time_embedding(time_steps)
        model_input = torch.cat([noisy_points, time_features], dim=1)
        return self.network(model_input)


# ---------- 扩散过程参数 ----------
def make_noise_schedule(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    创建线性 beta、alpha 与累计 alpha 序列。

    Beta 从 1e-4 线性增加到 0.02，对应 DDPM 论文中的线性调度。

    参数：
        device (torch.device): 张量所在设备

    返回：
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - betas: 形状 (DIFFUSION_STEPS,) 的 beta 序列
            - alphas: 形状 (DIFFUSION_STEPS,) 的 alpha = 1 - beta
            - cumulative_alphas: 形状 (DIFFUSION_STEPS,) 的 alpha 累乘 (alpha_bar)
    """
    betas = torch.linspace(1e-4, 0.02, DIFFUSION_STEPS, device=device)
    alphas = 1.0 - betas
    cumulative_alphas = torch.cumprod(alphas, dim=0)   # 累乘得到 alpha_bar_t
    return betas, alphas, cumulative_alphas


# ---------- 前向加噪 ----------
def add_noise(
    clean_points: torch.Tensor,
    time_steps: torch.Tensor,
    cumulative_alphas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    直接采样任意时间步的 x_t。

    利用重参数化技巧：x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    避免逐步执行完整前向扩散，提高训练效率。

    参数：
        clean_points (torch.Tensor): 形状 (batch_size, 2) 的干净点
        time_steps (torch.Tensor): 形状 (batch_size,) 的时间步
        cumulative_alphas (torch.Tensor): 预计算的 alpha_bar 序列

    返回：
        tuple[torch.Tensor, torch.Tensor]:
            - noisy_points: 带噪点，形状 (batch_size, 2)
            - target_noise: 实际加入的噪声，形状 (batch_size, 2)
    """
    # 生成与 clean_points 同形状的随机噪声
    random_noise = torch.randn_like(clean_points)
    # 根据时间步索引取出对应的 alpha_bar
    alpha_bar = cumulative_alphas[time_steps].unsqueeze(1)   # (batch_size, 1)

    # 计算带噪点
    noisy_points = alpha_bar.sqrt() * clean_points
    noisy_points += (1.0 - alpha_bar).sqrt() * random_noise
    return noisy_points, random_noise


# ---------- 训练函数 ----------
def train_model(
    model: NoisePredictor,
    train_steps: int,
    batch_size: int,
    cumulative_alphas: torch.Tensor,
    device: torch.device,
) -> None:
    """
    训练模型预测每个时间步加入的噪声。

    每步随机采样一批干净点和时间步，加噪后让模型预测噪声，
    优化 MSE 损失。

    参数：
        model (NoisePredictor): 噪声预测模型
        train_steps (int): 总训练迭代步数
        batch_size (int): 批次大小
        cumulative_alphas (torch.Tensor): alpha_bar 序列
        device (torch.device): 设备
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()
    model.train()

    for step in range(1, train_steps + 1):
        # 从数据分布中采样干净点
        clean_points = make_mixture_points(batch_size, device)

        # 随机采样时间步（0 ~ DIFFUSION_STEPS-1）
        time_steps = torch.randint(
            0,
            DIFFUSION_STEPS,
            (batch_size,),
            device=device,
        )

        # 前向加噪，得到带噪点和目标噪声
        noisy_points, target_noise = add_noise(
            clean_points,
            time_steps,
            cumulative_alphas,
        )

        # 模型预测噪声
        optimizer.zero_grad()
        predicted_noise = model(noisy_points, time_steps)
        loss = loss_function(predicted_noise, target_noise)
        loss.backward()
        optimizer.step()

        # 定期打印损失，监控训练进度
        report_interval = max(1, train_steps // 10)
        if step % report_interval == 0 or step == 1:
            print(f"step={step:04d} noise_mse={loss.item():.5f}")


# ---------- 反向采样（去噪） ----------
@torch.no_grad()
def sample_points(
    model: NoisePredictor,
    sample_count: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    cumulative_alphas: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    从高斯噪声开始逐步采样二维点（DDPM 反向过程）。

    参数：
        model (NoisePredictor): 训练好的噪声预测模型
        sample_count (int): 要生成的样本数
        betas (torch.Tensor): beta 序列
        alphas (torch.Tensor): alpha 序列
        cumulative_alphas (torch.Tensor): alpha_bar 序列
        device (torch.device): 设备

    返回：
        torch.Tensor: 形状 (sample_count, 2) 的生成点，位于 CPU
    """
    model.eval()
    # 从标准正态分布初始化
    points = torch.randn(sample_count, 2, device=device)

    # 从 T-1 倒推到 0
    for step in reversed(range(DIFFUSION_STEPS)):
        # 当前时间步（所有样本使用相同步数）
        time_steps = torch.full(
            (sample_count,),
            step,
            dtype=torch.long,
            device=device,
        )

        # 预测当前步的噪声
        predicted_noise = model(points, time_steps)

        # 获取当前步的参数
        beta = betas[step]
        alpha = alphas[step]
        alpha_bar = cumulative_alphas[step]

        # DDPM 反向更新公式：
        # x_{t-1} = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon_theta)
        # 然后加上噪声（除最后一步）
        points = (
            points - beta * predicted_noise / (1.0 - alpha_bar).sqrt()
        ) / alpha.sqrt()

        # 除最后一步（t=0）外，添加高斯噪声
        if step > 0:
            points += beta.sqrt() * torch.randn_like(points)

    # 返回 CPU 上的张量
    return points.cpu()


# ---------- 保存结果 ----------
def save_points_csv(points: torch.Tensor, path: Path) -> None:
    """
    将生成点保存为便于检查的 CSV。

    参数：
        points (torch.Tensor): 形状 (N, 2) 的点集
        path (Path): 输出文件路径
    """
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y"])           # 写入表头
        writer.writerows(points.tolist())     # 写入数据行


def save_optional_plot(points: torch.Tensor, path: Path) -> None:
    """
    若 Matplotlib 可用，则保存生成点散点图。

    此函数不强制依赖 Matplotlib，以免增加安装负担。

    参数：
        points (torch.Tensor): 形状 (N, 2) 的点集
        path (Path): 输出图片路径
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        # 若未安装，则静默跳过，仅输出提示
        print("Matplotlib 未安装，已跳过 PNG 绘图。")
        return

    # 创建散点图
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(points[:, 0], points[:, 1], s=8, alpha=0.6)
    axis.set_title("Toy Diffusion Samples")
    axis.set_aspect("equal")                 # 保持坐标轴比例一致
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)                        # 释放内存


# ---------- 主程序 ----------
def main() -> None:
    """
    训练二维 Diffusion 模型并保存采样结果。

    流程：
        1. 解析参数、固定随机种子、设置设备
        2. 创建噪声调度表
        3. 初始化模型
        4. 训练模型
        5. 从噪声生成样本
        6. 保存 checkpoint、CSV 和可选图片
    """
    args = parse_args()
    set_random_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 准备扩散调度参数
    betas, alphas, cumulative_alphas = make_noise_schedule(device)

    # 初始化模型并移至设备
    model = NoisePredictor().to(device)
    print(f"device: {device}")

    # 训练模型
    train_model(
        model,
        args.train_steps,
        args.batch_size,
        cumulative_alphas,
        device,
    )

    # 生成样本
    generated_points = sample_points(
        model,
        sample_count=1600,
        betas=betas,
        alphas=alphas,
        cumulative_alphas=cumulative_alphas,
        device=device,
    )

    # 定义输出文件路径
    checkpoint_path = args.output_dir / "toy_diffusion.pt"
    csv_path = args.output_dir / "toy_diffusion_samples.csv"
    plot_path = args.output_dir / "toy_diffusion_samples.png"

    # 保存模型权重、CSV 数据和图片（可选）
    torch.save(model.state_dict(), checkpoint_path)
    save_points_csv(generated_points, csv_path)
    save_optional_plot(generated_points, plot_path)

    print(f"checkpoint: {checkpoint_path.resolve()}")
    print(f"samples: {csv_path.resolve()}")


if __name__ == "__main__":
    main()