"""
PyTorch 入门教程综合练习。

内容覆盖自定义 Dataset、DataLoader、模型定义、训练、验证、checkpoint
保存与加载，以及单样本推理。数据由程序生成，不需要联网下载。

本模块包含：
- 合成三分类数据集生成
- 标准化转换类
- 自定义 Dataset 实现
- 带 BatchNorm 和 Dropout 的 MLP 分类器
- 训练与验证循环
- Checkpoint 保存与加载
- 单样本推理演示

"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

# 尝试导入 PyTorch，若未安装则给出友好提示
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, random_split
except ModuleNotFoundError as exc:
    raise SystemExit(
        "当前环境未安装 PyTorch；本脚本不会自动安装依赖。"
    ) from exc


# ---------- 全局常量 ----------
FEATURE_COUNT = 6      # 每个样本的特征维度
CLASS_COUNT = 3        # 分类类别数（三分类）


# ---------- 辅助函数 ----------
def parse_args() -> argparse.Namespace:
    """
    读取命令行训练参数。

    返回：
        argparse.Namespace: 包含 epochs, batch_size, learning_rate, output_dir
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12,
                        help="训练轮数（默认：12）")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批大小（默认：64）")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="学习率（默认：0.001）")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="保存检查点的目录（默认：outputs）")
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    """
    固定常用随机数生成器，确保实验可复现。

    参数：
        seed (int): 随机种子
    """
    random.seed(seed)
    torch.manual_seed(seed)
    # 若使用 CUDA，也固定 GPU 随机状态
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- 数据预处理 ----------
class StandardizeTransform:
    """
    使用给定均值和标准差标准化特征。

    公式： (x - mean) / std
    std 会被钳制到至少 1e-6 避免除零。
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """
        初始化标准化变换。

        参数：
            mean (torch.Tensor): 各特征的均值，形状 (FEATURE_COUNT,)
            std (torch.Tensor): 各特征的标准差，形状 (FEATURE_COUNT,)
        """
        self.mean = mean
        # 防止除零，将标准差中过小的值设为 1e-6
        self.std = std.clamp_min(1e-6)

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """
        对输入特征进行标准化。

        参数：
            features (torch.Tensor): 原始特征，形状 (..., FEATURE_COUNT)

        返回：
            torch.Tensor: 标准化后的特征，形状不变
        """
        return (features - self.mean) / self.std


# ---------- 自定义 Dataset ----------
class SyntheticClassificationDataset(Dataset):
    """
    保存三分类合成数据，并支持可选特征变换。

    数据集包含特征张量和标签张量，可应用 StandardizeTransform。
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        transform: StandardizeTransform | None = None,
    ) -> None:
        """
        初始化数据集。

        参数：
            features (torch.Tensor): 特征张量，形状 (样本数, FEATURE_COUNT)
            labels (torch.Tensor): 标签张量，形状 (样本数,)，值为 0, 1, 2
            transform (StandardizeTransform | None): 可选的标准化变换
        """
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """返回样本总数。"""
        return self.labels.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据索引读取一条特征和标签。

        参数：
            index (int): 样本索引

        返回：
            tuple[torch.Tensor, torch.Tensor]: (特征, 标签)
        """
        features = self.features[index]
        # 若有变换则应用（如标准化）
        if self.transform is not None:
            features = self.transform(features)
        return features, self.labels[index]


def make_raw_data(
    samples_per_class: int = 500,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    创建具有不同中心的三分类数据。

    每个类别从以不同中心点、相同方差的高斯分布采样。
    最终数据会被打乱。

    参数：
        samples_per_class (int): 每个类别的样本数（默认 500）
        seed (int): 随机种子

    返回：
        tuple[torch.Tensor, torch.Tensor]:
            - 特征张量，形状 (总样本数, FEATURE_COUNT)
            - 标签张量，形状 (总样本数,)
    """
    generator = torch.Generator().manual_seed(seed)

    # 三个类别的中心点（6 维空间中的不同位置）
    centers = torch.tensor(
        [
            [-2.0, -1.0, 0.0, 1.0, 0.5, -0.5],   # 类别 0 中心
            [1.5, -1.0, 2.0, -0.5, -1.0, 1.0],   # 类别 1 中心
            [0.0, 2.0, -1.5, 1.5, 1.0, 0.5],     # 类别 2 中心
        ]
    )

    feature_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []

    for class_index, center in enumerate(centers):
        # 生成围绕中心的高斯噪声，标准差为 0.9
        noise = torch.randn(
            samples_per_class,
            FEATURE_COUNT,
            generator=generator,
        )
        feature_parts.append(center + 0.9 * noise)
        # 创建对应的标签
        label_parts.append(
            torch.full((samples_per_class,), class_index, dtype=torch.long)
        )

    # 拼接所有类别数据
    features = torch.cat(feature_parts)
    labels = torch.cat(label_parts)

    # 打乱数据，使类别混合
    permutation = torch.randperm(labels.size(0), generator=generator)
    return features[permutation], labels[permutation]


# ---------- 模型定义 ----------
class MultilayerClassifier(nn.Module):
    """
    带 BatchNorm 和 Dropout 的小型分类网络。

    结构：
        - 线性层 (6 -> 32)
        - BatchNorm1d(32)
        - ReLU
        - Dropout(0.15)
        - 线性层 (32 -> 16)
        - ReLU
        - 线性层 (16 -> 3)  输出 logits
    """

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_COUNT, 32),
            nn.BatchNorm1d(32),          # 对 batch 维度归一化，加速训练
            nn.ReLU(),
            nn.Dropout(0.15),            # 防止过拟合
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, CLASS_COUNT),  # 输出三分类 logits
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        将输入特征映射为类别 logits。

        参数：
            features (torch.Tensor): 形状 (batch_size, FEATURE_COUNT)

        返回：
            torch.Tensor: 形状 (batch_size, CLASS_COUNT) 的 logits
        """
        return self.network(features)


# ---------- 训练/验证指标 ----------
@dataclass(frozen=True)
class EpochMetrics:
    """
    记录一轮的平均损失和准确率。

    属性：
        loss (float): 平均损失
        accuracy (float): 准确率（0~1）
    """
    loss: float
    accuracy: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    """
    执行训练或验证循环。

    若 optimizer 不为 None 则进入训练模式并更新参数；
    否则为验证模式，仅计算指标。

    参数：
        model (nn.Module): 模型
        loader (DataLoader): 数据加载器
        loss_function (nn.Module): 损失函数
        device (torch.device): 设备
        optimizer (torch.optim.Optimizer | None): 若提供则训练

    返回：
        EpochMetrics: 包含平均损失和准确率
    """
    is_training = optimizer is not None
    model.train(is_training)   # 根据模式切换 dropout 和 batchnorm 行为

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    # 验证时禁用梯度计算，节省资源
    with torch.set_grad_enabled(is_training):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            if optimizer is not None:
                optimizer.zero_grad()

            logits = model(features)
            loss = loss_function(logits, labels)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            # 统计
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct_count += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    return EpochMetrics(
        loss=total_loss / sample_count,
        accuracy=correct_count / sample_count,
    )


# ---------- Checkpoint 操作 ----------
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_accuracy: float,
) -> None:
    """
    保存恢复训练所需的状态。

    保存内容包括：
        - 当前轮次
        - 模型状态字典
        - 优化器状态字典
        - 验证准确率

    参数：
        path (Path): 保存路径
        model (nn.Module): 模型
        optimizer (torch.optim.Optimizer): 优化器
        epoch (int): 当前轮次
        validation_accuracy (float): 当前验证准确率
    """
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "validation_accuracy": validation_accuracy,
    }
    torch.save(checkpoint, path)


def load_model_for_inference(
    path: Path,
    device: torch.device,
) -> MultilayerClassifier:
    """
    从 checkpoint 恢复一个推理模型。

    仅加载模型权重，不加载优化器状态，并设置为评估模式。

    参数：
        path (Path): checkpoint 文件路径
        device (torch.device): 目标设备

    返回：
        MultilayerClassifier: 加载好权重的模型（eval 模式）
    """
    model = MultilayerClassifier().to(device)

    # 使用 weights_only=True 提高安全性（若 PyTorch 版本支持）
    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # 旧版本 PyTorch 不支持 weights_only，回退到普通加载
        checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()   # 切换到推理模式（禁用 dropout 等）
    return model


# ---------- 主程序 ----------
def main() -> None:
    """
    运行完整 PyTorch 机器学习工作流。

    步骤：
        1. 解析参数、固定随机种子、设置设备
        2. 生成原始数据，计算标准化参数，创建 Dataset
        3. 划分训练集和验证集 (80% / 20%)
        4. 创建 DataLoader
        5. 初始化模型、损失函数、优化器
        6. 循环训练，每轮验证并保存最佳模型
        7. 加载最佳模型，进行单样本推理演示
        8. 输出最终结果
    """
    args = parse_args()
    set_random_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 数据准备 ----------
    # 生成原始特征和标签
    raw_features, labels = make_raw_data()

    # 基于整个数据集计算标准化参数（均值、标准差）
    transform = StandardizeTransform(
        mean=raw_features.mean(dim=0),
        std=raw_features.std(dim=0),
    )

    # 创建完整数据集（应用标准化）
    full_dataset = SyntheticClassificationDataset(
        raw_features,
        labels,
        transform,
    )

    # 按 80%/20% 划分训练集和验证集（固定种子保证可复现）
    train_size = int(0.8 * len(full_dataset))
    validation_size = len(full_dataset) - train_size
    train_dataset, validation_dataset = random_split(
        full_dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,           # 训练集打乱
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,          # 验证集不打乱
    )

    # ---------- 模型、损失、优化器 ----------
    model = MultilayerClassifier().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,      # L2 正则化
    )

    checkpoint_path = args.output_dir / "pytorch_tutorial_checkpoint.pt"
    best_accuracy = 0.0

    print(f"device: {device}")
    print(f"train samples: {len(train_dataset)}")
    print(f"validation samples: {len(validation_dataset)}")

    # ---------- 训练循环 ----------
    for epoch in range(1, args.epochs + 1):
        # 训练一轮
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
        )
        # 验证一轮
        validation_metrics = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            optimizer=None,    # 不更新参数
        )

        # 若验证准确率提升，保存模型
        if validation_metrics.accuracy > best_accuracy:
            best_accuracy = validation_metrics.accuracy
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                validation_metrics.accuracy,
            )

        # 打印本轮指标
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.accuracy:.3f} "
            f"val_loss={validation_metrics.loss:.4f} "
            f"val_acc={validation_metrics.accuracy:.3f}"
        )

    # ---------- 推理演示 ----------
    # 重新加载保存的最佳模型
    inference_model = load_model_for_inference(checkpoint_path, device)

    # 取数据集中第一个样本进行预测
    sample_features, sample_label = full_dataset[0]
    with torch.no_grad():
        logits = inference_model(sample_features.unsqueeze(0).to(device))
        predicted_label = int(logits.argmax(dim=1).item())

    # 输出最终结果
    print(f"best validation accuracy: {best_accuracy:.3f}")
    print(f"sample prediction: {predicted_label}")
    print(f"sample label: {int(sample_label)}")
    print(f"checkpoint: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()