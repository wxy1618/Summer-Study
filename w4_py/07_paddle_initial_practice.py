"""
PaddlePaddle 初步实践：Tensor、自动求导、训练和模型保存。

本脚本只运行已有环境中的 PaddlePaddle，不包含 pip、conda 或其他安装命令。
若依赖缺失，程序会明确提示并退出。

内容涵盖：
- Tensor 创建与自动求导演示
- 自定义 Dataset
- DataLoader 使用
- 多层感知机分类模型
- 训练与验证循环
- 模型参数保存与加载
- 单样本推理

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# 尝试导入 PaddlePaddle 相关模块，若未安装则给出友好提示
try:
    import numpy as np
    import paddle
    from paddle import nn
    from paddle.io import DataLoader, Dataset
except ModuleNotFoundError as exc:
    raise SystemExit(
        "当前环境未安装 PaddlePaddle；本脚本不会自动安装依赖。"
    ) from exc


# ---------- 全局常量 ----------
FEATURE_COUNT = 5      # 每个样本的特征维度
CLASS_COUNT = 3        # 分类类别数（三分类）


# ---------- 辅助函数 ----------
def parse_args() -> argparse.Namespace:
    """
    读取训练轮数和输出目录。

    返回：
        argparse.Namespace: 包含 epochs, batch_size, output_dir
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12,
                        help="训练轮数（默认：12）")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批大小（默认：64）")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="保存模型参数的目录（默认：outputs）")
    return parser.parse_args()


def demonstrate_autograd() -> None:
    """
    展示 Paddle Tensor 与自动求导。

    创建一个标量张量 x=2.0，计算 y = x^2 + 3x，然后反向传播计算梯度。
    理论梯度为 dy/dx = 2x + 3，在 x=2 时应为 7。
    """
    # 创建张量，stop_gradient=False 表示需要计算梯度
    value = paddle.to_tensor([2.0], stop_gradient=False)
    # 计算表达式
    result = value.square() + 3.0 * value
    # 反向传播自动求导
    result.backward()

    # 输出值和梯度
    print("autograd value:", float(result.numpy().item()))
    print("autograd gradient:", float(value.grad.numpy().item()))


def make_data(
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    创建三分类 NumPy 数据。

    数据分布：三个类别围绕不同中心点，共享方差，适合分类练习。
    生成方式：先随机分配类别标签，再根据类别中心添加高斯噪声。

    参数：
        sample_count (int): 样本总数
        seed (int): 随机种子

    返回：
        tuple[np.ndarray, np.ndarray]:
            - 特征数组，形状 (sample_count, FEATURE_COUNT)，dtype=float32
            - 标签数组，形状 (sample_count,)，dtype=int64
    """
    # 使用 NumPy 的随机数生成器（可复现）
    generator = np.random.default_rng(seed)

    # 三个类别的中心点（5 维空间）
    centers = np.array(
        [
            [-2.0, -1.0, 0.0, 1.0, 0.5],       # 类别 0 中心
            [1.5, -1.0, 2.0, -0.5, -1.0],     # 类别 1 中心
            [0.0, 2.0, -1.5, 1.5, 1.0],       # 类别 2 中心
        ],
        dtype="float32",
    )

    # 随机分配类别标签（均匀分布）
    labels = generator.integers(0, CLASS_COUNT, size=sample_count)

    # 生成高斯噪声，标准差为 0.9
    noise = generator.normal(0.0, 0.9, size=(sample_count, FEATURE_COUNT))
    # 根据标签选择对应的中心，并加上噪声
    features = centers[labels] + noise.astype("float32")
    return features.astype("float32"), labels.astype("int64")


# ---------- 自定义 Dataset ----------
class ClassificationDataset(Dataset):
    """
    PaddlePaddle 自定义三分类数据集。

    封装特征和标签的 NumPy 数组，提供 __len__ 和 __getitem__ 方法。
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        初始化数据集。

        参数：
            features (np.ndarray): 特征数组
            labels (np.ndarray): 标签数组
        """
        super().__init__()
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        """返回样本数。"""
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.int64]:
        """
        根据索引读取一条样本。

        参数：
            index (int): 索引

        返回：
            tuple[np.ndarray, np.int64]: (特征向量, 标签)
        """
        return self.features[index], self.labels[index]


# ---------- 模型定义 ----------
class PaddleClassifier(nn.Layer):
    """
    用于三分类的 PaddlePaddle 多层感知机。

    结构：
        - 线性层 (5 -> 32)
        - ReLU
        - Dropout(0.1)
        - 线性层 (32 -> 16)
        - ReLU
        - 线性层 (16 -> 3) 输出 logits
    """

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_COUNT, 32),
            nn.ReLU(),
            nn.Dropout(0.1),            # 防止过拟合
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, CLASS_COUNT), # 输出三分类 logits
        )

    def forward(self, features: paddle.Tensor) -> paddle.Tensor:
        """
        前向传播，返回类别 logits。

        参数：
            features (paddle.Tensor): 形状 (batch_size, FEATURE_COUNT)

        返回：
            paddle.Tensor: 形状 (batch_size, CLASS_COUNT) 的 logits
        """
        return self.network(features)


# ---------- 训练与验证指标 ----------
@dataclass(frozen=True)
class PaddleMetrics:
    """
    记录平均损失和准确率。

    属性：
        loss (float): 平均损失
        accuracy (float): 准确率（0~1）
    """
    loss: float
    accuracy: float


def train_one_epoch(
    model: PaddleClassifier,
    loader: DataLoader,
    loss_function: nn.Layer,
    optimizer: paddle.optimizer.Optimizer,
) -> PaddleMetrics:
    """
    执行一轮 PaddlePaddle 训练。

    遍历 DataLoader，计算损失、反向传播、更新参数，并统计平均损失和准确率。

    参数：
        model (PaddleClassifier): 模型
        loader (DataLoader): 训练数据加载器
        loss_function (nn.Layer): 损失函数（如 CrossEntropyLoss）
        optimizer (paddle.optimizer.Optimizer): 优化器

    返回：
        PaddleMetrics: 包含平均损失和准确率
    """
    model.train()          # 切换到训练模式（启用 Dropout 等）
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for features, labels in loader:
        # 前向传播
        logits = model(features)
        loss = loss_function(logits, labels)

        # 反向传播与参数更新
        optimizer.clear_grad()     # Paddle 需要显式清空梯度（不同于 PyTorch 的 zero_grad）
        loss.backward()
        optimizer.step()

        # 统计准确率
        predictions = logits.argmax(axis=1)          # 预测类别
        batch_size = int(labels.shape[0])
        total_loss += float(loss.numpy().item()) * batch_size
        # 比较预测和标签，求和得到正确数量
        correct_count += int(
            (predictions == labels).astype("int64").sum().item()
        )
        sample_count += batch_size

    return PaddleMetrics(
        loss=total_loss / sample_count,
        accuracy=correct_count / sample_count,
    )


@paddle.no_grad()
def evaluate(
    model: PaddleClassifier,
    loader: DataLoader,
    loss_function: nn.Layer,
) -> PaddleMetrics:
    """
    在关闭梯度的情况下评估模型。

    参数：
        model (PaddleClassifier): 模型
        loader (DataLoader): 验证数据加载器
        loss_function (nn.Layer): 损失函数

    返回：
        PaddleMetrics: 包含平均损失和准确率
    """
    model.eval()            # 切换到评估模式（禁用 Dropout）
    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for features, labels in loader:
        logits = model(features)
        loss = loss_function(logits, labels)
        predictions = logits.argmax(axis=1)
        batch_size = int(labels.shape[0])

        total_loss += float(loss.numpy().item()) * batch_size
        correct_count += int(
            (predictions == labels).astype("int64").sum().item()
        )
        sample_count += batch_size

    return PaddleMetrics(
        loss=total_loss / sample_count,
        accuracy=correct_count / sample_count,
    )


# ---------- 主程序 ----------
def main() -> None:
    """
    完成 PaddlePaddle 数据、训练、保存、加载和推理流程。

    步骤：
        1. 解析参数、固定随机种子、创建输出目录
        2. 演示自动求导
        3. 生成合成数据并划分为训练集和验证集
        4. 创建 DataLoader
        5. 初始化模型、损失函数、优化器
        6. 循环训练和验证，保存最佳模型
        7. 加载最佳模型，进行单样本推理演示
        8. 输出最终结果
    """
    args = parse_args()
    # 固定 Paddle 和 NumPy 随机种子
    paddle.seed(42)
    np.random.seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 展示自动求导功能
    demonstrate_autograd()

    # ---------- 数据准备 ----------
    features, labels = make_data(sample_count=1800, seed=42)
    # 按 1400/400 划分训练集和验证集
    split_index = 1400
    train_dataset = ClassificationDataset(
        features[:split_index],
        labels[:split_index],
    )
    validation_dataset = ClassificationDataset(
        features[split_index:],
        labels[split_index:],
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,           # 训练集打乱顺序
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,          # 验证集保持固定顺序
    )

    # ---------- 模型、损失、优化器 ----------
    model = PaddleClassifier()
    loss_function = nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        parameters=model.parameters(),
    )

    # 模型参数保存路径（Paddle 推荐使用 .pdparams 后缀）
    parameter_path = args.output_dir / "paddle_classifier.pdparams"
    best_accuracy = 0.0

    print("device:", paddle.device.get_device())

    # ---------- 训练循环 ----------
    for epoch in range(1, args.epochs + 1):
        # 训练一轮
        train_metrics = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
        )
        # 验证一轮
        validation_metrics = evaluate(
            model,
            validation_loader,
            loss_function,
        )

        # 如果验证准确率提高，保存模型参数
        if validation_metrics.accuracy > best_accuracy:
            best_accuracy = validation_metrics.accuracy
            # paddle.save 保存 state_dict，推荐使用字符串路径
            paddle.save(model.state_dict(), str(parameter_path))

        # 打印本轮指标
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_acc={train_metrics.accuracy:.3f} "
            f"val_loss={validation_metrics.loss:.4f} "
            f"val_acc={validation_metrics.accuracy:.3f}"
        )

    # ---------- 推理演示 ----------
    # 创建新模型并加载保存的参数（验证持久化流程）
    restored_model = PaddleClassifier()
    restored_state = paddle.load(str(parameter_path))   # 加载 state_dict
    restored_model.set_state_dict(restored_state)       # 设置参数
    restored_model.eval()                               # 切换到推理模式

    # 取第一个样本进行预测
    sample = paddle.to_tensor(features[:1])
    with paddle.no_grad():
        prediction = int(restored_model(sample).argmax(axis=1).item())

    # 输出最终结果
    print(f"best validation accuracy: {best_accuracy:.3f}")
    print(f"sample prediction: {prediction}")
    print(f"sample label: {int(labels[0])}")
    print(f"parameters: {parameter_path.resolve()}")


# 当脚本作为主程序运行时执行
if __name__ == "__main__":
    main()