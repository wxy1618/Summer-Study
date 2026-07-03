"""
Homework 6：在程序生成的几何图形上训练轻量 GAN。

真实数据由 16 x 16 的方块和十字图案组成，不需要下载外部数据。生成结果
保存为通用 PGM 灰度图片，可用多数图片查看器打开。

本模块包含：
- 几何图形数据集生成（方块和十字）
- 生成器网络（MLP 将潜向量映射为图像）
- 判别器网络（MLP 二分类真假）
- 标准 GAN 训练循环（交替优化）
- 保存结果为 PGM 网格图像

"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

# 尝试导入 PyTorch，若未安装则给出友好提示
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "当前环境未安装 PyTorch；本脚本不会自动安装依赖。"
    ) from exc


# ---------- 全局常量 ----------
IMAGE_SIZE = 16          # 图像尺寸（16x16 像素）
LATENT_SIZE = 32         # 生成器输入潜向量维度


# ---------- 辅助函数 ----------
def parse_args() -> argparse.Namespace:
    """
    读取 GAN 训练参数。

    返回：
        argparse.Namespace: 包含 epochs, batch_size, output_dir
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15,
                        help="训练轮数（默认：15）")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批大小（默认：64）")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="输出目录（默认：outputs）")
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """
    固定随机状态，确保实验可复现。

    参数：
        seed (int): 随机种子
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- 合成数据生成 ----------
def make_shape_image(generator: torch.Generator) -> torch.Tensor:
    """
    随机创建方块或十字图案的单通道图像。

    生成过程：
        1. 初始化全零 16x16 图像
        2. 随机选择中心坐标（范围 4~11）
        3. 随机选择半尺寸（2 或 3）
        4. 随机选择形状类型（0=方块，1=十字）
        5. 绘制相应的形状（像素值设为 1.0）
        6. 添加少量高斯噪声（标准差 0.03）并截断到 [0,1]

    参数：
        generator (torch.Generator): 随机数生成器

    返回：
        torch.Tensor: 形状 (1, IMAGE_SIZE, IMAGE_SIZE) 的单通道图像
    """
    # 初始化全零图像
    image = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
    # 随机中心（避免贴边，保证形状完整）
    center_x = int(torch.randint(4, 12, (1,), generator=generator))
    center_y = int(torch.randint(4, 12, (1,), generator=generator))
    # 随机半大小（2 或 3，控制形状大小）
    half_size = int(torch.randint(2, 4, (1,), generator=generator))
    # 随机形状类型
    shape_type = int(torch.randint(0, 2, (1,), generator=generator))

    if shape_type == 0:
        # 方块：填充矩形区域
        image[
            center_y - half_size:center_y + half_size + 1,
            center_x - half_size:center_x + half_size + 1,
        ] = 1.0
    else:
        # 十字：水平线和垂直线
        image[center_y, center_x - half_size:center_x + half_size + 1] = 1.0
        image[center_y - half_size:center_y + half_size + 1, center_x] = 1.0

    # 添加小噪声使数据略显自然，防止判别器过拟合到严格二值模式
    noise = torch.randn(IMAGE_SIZE, IMAGE_SIZE, generator=generator) * 0.03
    image = (image + noise).clamp(0.0, 1.0)   # 保持像素值范围
    return image.unsqueeze(0)                # 添加通道维度


class ShapeDataset(Dataset):
    """
    预先生成的几何图形数据集（用于训练 GAN）。

    在初始化时生成固定数量的图像，并映射到 [-1, 1] 范围（匹配生成器的 Tanh 输出）。
    """

    def __init__(self, sample_count: int, seed: int) -> None:
        """
        生成指定数量的图像。

        参数：
            sample_count (int): 样本数量
            seed (int): 随机种子，保证可复现
        """
        generator = torch.Generator().manual_seed(seed)
        # 生成所有图像并堆叠成张量
        self.images = torch.stack(
            [make_shape_image(generator) for _ in range(sample_count)]
        )
        # 将 [0,1] 缩放到 [-1,1]，与生成器 Tanh 输出一致
        self.images = self.images * 2.0 - 1.0

    def __len__(self) -> int:
        """返回图像数量。"""
        return self.images.size(0)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        读取一张单通道图像。

        参数：
            index (int): 索引

        返回：
            torch.Tensor: 形状 (1, IMAGE_SIZE, IMAGE_SIZE)
        """
        return self.images[index]


# ---------- 生成器 ----------
class Generator(nn.Module):
    """
    把潜向量映射为 16 x 16 灰度图。

    结构：三层全连接，使用 BatchNorm 和 LeakyReLU，输出 Tanh。
    输入：形状 (batch, LATENT_SIZE) 的随机噪声
    输出：形状 (batch, 1, IMAGE_SIZE, IMAGE_SIZE) 的图像
    """

    def __init__(self) -> None:
        super().__init__()
        pixel_count = IMAGE_SIZE * IMAGE_SIZE
        self.network = nn.Sequential(
            nn.Linear(LATENT_SIZE, 128),
            nn.BatchNorm1d(128),        # 稳定训练
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, pixel_count),
            nn.Tanh(),                  # 输出范围 [-1, 1]
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        前向传播，生成图像。

        参数：
            noise (torch.Tensor): 形状 (batch_size, LATENT_SIZE)

        返回：
            torch.Tensor: 形状 (batch_size, 1, IMAGE_SIZE, IMAGE_SIZE)
        """
        images = self.network(noise)
        # 重塑为图像格式
        return images.view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)


# ---------- 判别器 ----------
class Discriminator(nn.Module):
    """
    判断输入图像更接近真实数据还是生成数据。

    结构：三层全连接，使用 LeakyReLU 和 Dropout，输出单个 logit。
    输入：形状 (batch, 1, IMAGE_SIZE, IMAGE_SIZE)
    输出：形状 (batch,) 的未经过 Sigmoid 的 logits
    """

    def __init__(self) -> None:
        super().__init__()
        pixel_count = IMAGE_SIZE * IMAGE_SIZE
        self.network = nn.Sequential(
            nn.Flatten(),                # 展平图像为向量
            nn.Linear(pixel_count, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),             # 防止判别器过拟合
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 1),           # 输出 logit
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播，输出真假 logits。

        参数：
            images (torch.Tensor): 形状 (batch_size, 1, IMAGE_SIZE, IMAGE_SIZE)

        返回：
            torch.Tensor: 形状 (batch_size,) 的 logits
        """
        return self.network(images).squeeze(1)


# ---------- GAN 训练函数 ----------
def train_gan(
    generator: Generator,
    discriminator: Discriminator,
    loader: DataLoader,
    epochs: int,
    device: torch.device,
) -> None:
    """
    交替训练判别器和生成器。

    每个 epoch 遍历所有 batch：
        1. 更新判别器：区分真实图像和生成图像（生成图像停止梯度）
        2. 更新生成器：使判别器将其生成的图像判断为真

    参数：
        generator (Generator): 生成器
        discriminator (Discriminator): 判别器
        loader (DataLoader): 训练数据加载器
        epochs (int): 训练轮数
        device (torch.device): 设备
    """
    # 损失函数：二分类交叉熵（带 logits）
    loss_function = nn.BCEWithLogitsLoss()

    # 优化器：Adam，使用标准 GAN 的 beta 参数
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=2e-4,
        betas=(0.5, 0.999),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=2e-4,
        betas=(0.5, 0.999),
    )

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        generator_loss_sum = 0.0
        discriminator_loss_sum = 0.0

        for real_images in loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # 标签：使用平滑标签（0.9 代替 1.0）有助于稳定训练
            real_targets = torch.full((batch_size,), 0.9, device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            # ---------- 训练判别器 ----------
            discriminator_optimizer.zero_grad()
            # 生成假图像，detach 避免梯度传到生成器
            noise = torch.randn(batch_size, LATENT_SIZE, device=device)
            fake_images = generator(noise).detach()

            # 真实图像损失
            real_loss = loss_function(discriminator(real_images), real_targets)
            # 假图像损失
            fake_loss = loss_function(discriminator(fake_images), fake_targets)
            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # ---------- 训练生成器 ----------
            generator_optimizer.zero_grad()
            # 重新生成假图像（不 detach，让梯度传递）
            noise = torch.randn(batch_size, LATENT_SIZE, device=device)
            generated_images = generator(noise)
            # 生成器的目标是让判别器认为生成的图像为真（标签为 1）
            generator_loss = loss_function(
                discriminator(generated_images),
                torch.ones(batch_size, device=device),   # 目标为真
            )
            generator_loss.backward()
            generator_optimizer.step()

            # 累加损失用于统计
            generator_loss_sum += generator_loss.item()
            discriminator_loss_sum += discriminator_loss.item()

        # 打印当前 epoch 的平均损失
        batch_count = len(loader)
        print(
            f"epoch={epoch:02d} "
            f"generator_loss={generator_loss_sum / batch_count:.4f} "
            f"discriminator_loss={discriminator_loss_sum / batch_count:.4f}"
        )


# ---------- 保存 PGM 网格 ----------
def save_pgm_grid(images: torch.Tensor, path: Path, columns: int = 8) -> None:
    """
    把一组单通道图像写成二进制 PGM 网格。

    PGM 格式（P5）支持灰度图，无需额外库。
    图像值从 [-1,1] 映射到 [0,255]，并排成网格以便查看所有生成样本。

    参数：
        images (torch.Tensor): 形状 (N, 1, IMAGE_SIZE, IMAGE_SIZE) 的图像
        path (Path): 输出文件路径
        columns (int): 网格列数（默认 8）
    """
    # 将图像从 [-1,1] 映射到 [0,255] 并转为 uint8
    images = images.detach().cpu().clamp(-1.0, 1.0)
    images = ((images + 1.0) * 127.5).to(torch.uint8)
    image_count = images.size(0)
    rows = (image_count + columns - 1) // columns   # 计算行数

    # 创建网格张量
    grid = torch.zeros(rows * IMAGE_SIZE, columns * IMAGE_SIZE, dtype=torch.uint8)

    # 将每个图像放入网格对应位置
    for index, image in enumerate(images):
        row = index // columns
        col = index % columns
        row_start = row * IMAGE_SIZE
        col_start = col * IMAGE_SIZE
        grid[
            row_start:row_start + IMAGE_SIZE,
            col_start:col_start + IMAGE_SIZE,
        ] = image[0]   # image 形状为 (1, H, W)，取第 0 通道

    # 写入 PGM 头：格式 P5，宽度 高度，最大灰度值 255
    header = f"P5\n{grid.size(1)} {grid.size(0)}\n255\n".encode("ascii")
    # 将网格数据序列化为字节并保存
    path.write_bytes(header + bytes(grid.contiguous().view(-1).tolist()))


# ---------- 主程序 ----------
def main() -> None:
    """
    训练 GAN，并保存模型参数和固定噪声生成结果。

    流程：
        1. 解析参数、固定随机种子、设置设备
        2. 创建数据集和数据加载器
        3. 初始化生成器和判别器
        4. 训练 GAN
        5. 使用固定种子生成 64 张图像用于可视化比较
        6. 保存生成器权重和 PGM 网格图像
    """
    args = parse_args()
    set_random_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 创建数据集（包含 2400 张真实图像）
    dataset = ShapeDataset(sample_count=2400, seed=42)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,      # 丢弃最后一个不完整的 batch，保持训练稳定
    )

    # 初始化模型
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    print(f"device: {device}")
    # 训练
    train_gan(generator, discriminator, loader, args.epochs, device)

    # ---------- 推理：生成固定样本 ----------
    # 使用固定种子生成 64 张图，便于不同实验之间进行视觉比较
    sample_generator = torch.Generator(device=device).manual_seed(2026)
    fixed_noise = torch.randn(
        64,
        LATENT_SIZE,
        generator=sample_generator,
        device=device,
    )
    generator.eval()
    with torch.no_grad():
        sample_images = generator(fixed_noise)

    # 保存生成器权重
    model_path = args.output_dir / "shape_gan_generator.pt"
    image_path = args.output_dir / "shape_gan_samples.pgm"
    torch.save(generator.state_dict(), model_path)
    save_pgm_grid(sample_images, image_path)

    print(f"generator checkpoint: {model_path.resolve()}")
    print(f"generated image grid: {image_path.resolve()}")


if __name__ == "__main__":
    main()