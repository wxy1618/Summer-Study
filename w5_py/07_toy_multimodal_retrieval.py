# -*- coding: utf-8 -*-
"""Toy multimodal contrastive retrieval.

本脚本对应第五周“多模态学习”阅读任务。它用合成图像特征和短文本描述
模拟 CLIP 式对比学习，观察匹配图文是否能在共享空间中相互检索。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

try:
    # 多模态检索实验需要 PyTorch 完成双编码器训练；缺失时只提示不安装。
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    torch = None

    class _MissingNN:
        Module = object

    nn = _MissingNN()
    F = None
    Dataset = object
    DataLoader = None


TOKEN_RE = re.compile(r"[a-zA-Z]+|[0-9]+")

# 这里把“图像”压缩成人工特征向量，是为了隔离多模态对齐目标本身。
# 真实系统会用 CNN/ViT 得到视觉表示，但对比学习损失的形式是一致的。

# image_features 是人为构造的低维“视觉属性”：颜色、形状、纹理等。
# 真实多模态模型会用 CNN/ViT 编码图像，这里保留接口但去掉沉重计算。
PAIRS = [
    ("red square with sharp edges", [1.0, 0.0, 0.0, 1.0, 0.0, 0.2]),
    ("blue circle with smooth boundary", [0.0, 1.0, 0.0, 0.0, 1.0, 0.1]),
    ("green triangle with three corners", [0.0, 0.0, 1.0, 0.7, 0.0, 0.8]),
    ("red circle on a plain background", [1.0, 0.0, 0.0, 0.0, 1.0, 0.1]),
    ("blue square with regular structure", [0.0, 1.0, 0.0, 1.0, 0.0, 0.2]),
    ("green circle with soft outline", [0.0, 0.0, 1.0, 0.0, 1.0, 0.1]),
]


def require_torch() -> None:
    # 不在脚本中配置环境，保证它可以安全放入已有课程作业目录。
    if torch is None:
        print("当前环境未安装 PyTorch；本脚本不会自动安装依赖。")
        raise SystemExit(0)


def tokenize(text: str) -> list[str]:
    # 简单英文分词足够表达颜色、形状、纹理等可控语义属性。
    return TOKEN_RE.findall(text.lower())


def build_vocab() -> dict[str, int]:
    # 词表从所有文本描述中构建；本实验关注检索对齐，不额外设置验证集词表。
    words = sorted({token for text, _ in PAIRS for token in tokenize(text)})
    # <pad> 负责批处理补齐，<unk> 保留给潜在的新文本查询。
    return {"<pad>": 0, "<unk>": 1, **{word: idx + 2 for idx, word in enumerate(words)}}


class MultimodalDataset(Dataset):
    """Return tokenized text and its paired synthetic image vector."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        # Dataset 保存词表引用，使每次索引访问能把自然语言转成 id 序列。
        self.token_to_id = token_to_id

    def __len__(self) -> int:
        # 样本数等于图文配对数，也是对比学习中 logits 矩阵的边长。
        return len(PAIRS)

    def __getitem__(self, index: int) -> tuple[list[int], "torch.Tensor", str]:
        # 返回原文 text 是为了评估时输出可解释的检索结果。
        text, image_features = PAIRS[index]
        unk_id = self.token_to_id["<unk>"]
        # 文本走离散 token 通道，图像走连续属性通道，模拟两种模态的差异。
        token_ids = [self.token_to_id.get(token, unk_id) for token in tokenize(text)]
        return token_ids, torch.tensor(image_features, dtype=torch.float32), text


def collate_batch(
    batch: list[tuple[list[int], "torch.Tensor", str]],
) -> tuple["torch.Tensor", "torch.Tensor", list[str]]:
    # 文本长度不同，需要 padding；图像特征已经是定长向量，可以直接 stack。
    max_length = max(len(token_ids) for token_ids, _, _ in batch)
    padded = []
    images = []
    texts = []
    for token_ids, image, text in batch:
        # padding id 为 0，后续文本编码器会用 mask 排除这些位置。
        padded.append(token_ids + [0] * (max_length - len(token_ids)))
        images.append(image)
        texts.append(text)
    return torch.tensor(padded, dtype=torch.long), torch.stack(images), texts


class ToyCLIP(nn.Module):
    """A small dual encoder trained with symmetric contrastive loss."""

    def __init__(self, vocab_size: int, image_dim: int, projection_dim: int) -> None:
        super().__init__()
        # 文本编码器先做词向量平均，再投影到共享语义空间。
        self.text_embedding = nn.Embedding(vocab_size, projection_dim, padding_idx=0)
        self.text_projection = nn.Linear(projection_dim, projection_dim)
        # 图像编码器把人工视觉属性映射到同一个 projection_dim。
        self.image_projection = nn.Linear(image_dim, projection_dim)
        # 可学习温度控制相似度分布的尖锐程度，是 CLIP 类模型的关键标量。
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def encode_text(self, text_ids: "torch.Tensor") -> "torch.Tensor":
        # mask-aware mean pooling 避免 PAD token 改变文本向量均值。
        mask = text_ids.ne(0).float()
        embedded = self.text_embedding(text_ids)
        token_sum = (embedded * mask.unsqueeze(-1)).sum(dim=1)
        token_count = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        pooled = token_sum / token_count
        # L2 归一化后，点积等价于余弦相似度，更适合跨模态检索。
        return F.normalize(self.text_projection(pooled), dim=1)

    def encode_image(self, image_features: "torch.Tensor") -> "torch.Tensor":
        # 图像侧同样归一化，保证相似度主要由方向决定而非向量范数。
        return F.normalize(self.image_projection(image_features), dim=1)

    def forward(
        self,
        text_ids: "torch.Tensor",
        image_features: "torch.Tensor",
    ) -> "torch.Tensor":
        # 双编码器分别得到文本和图像向量，再构造所有文本-图像两两相似度。
        text_vectors = self.encode_text(text_ids)
        image_vectors = self.encode_image(image_features)
        # clamp 限制温度范围，避免极小温度导致 logits 过大、训练不稳定。
        temperature = self.log_temperature.exp().clamp(min=0.05, max=10.0)
        return text_vectors @ image_vectors.t() / temperature


def contrastive_loss(logits: "torch.Tensor") -> "torch.Tensor":
    """Symmetric InfoNCE loss for text-to-image and image-to-text retrieval."""

    # 一个 batch 中第 i 条文本和第 i 个图像是正样本，其他配对都是负样本。
    labels = torch.arange(logits.size(0))
    # 对称损失同时优化 text->image 和 image->text 两个检索方向。
    text_to_image = F.cross_entropy(logits, labels)
    image_to_text = F.cross_entropy(logits.t(), labels)
    return 0.5 * (text_to_image + image_to_text)


def evaluate_retrieval(
    model: ToyCLIP,
    loader: "DataLoader",
) -> tuple[float, list[dict[str, object]]]:
    """Compute Recall@1 and save the nearest image text for each query."""

    # eval 模式用于关闭潜在随机层；当前模型无 dropout，但保留规范写法。
    model.eval()
    with torch.no_grad():
        # 教学数据很小，评估时一次性取完整集合构造全量相似度矩阵。
        text_ids, image_features, texts = next(iter(loader))
        logits = model(text_ids, image_features)
        # 每行按相似度降序排序，第一名即文本查询检索到的图像。
        rankings = logits.argsort(dim=1, descending=True)
        correct = 0
        rows: list[dict[str, object]] = []
        for query_index, ranking in enumerate(rankings.tolist()):
            predicted_index = ranking[0]
            correct += int(predicted_index == query_index)
            # 把 top1 映射回文本描述，便于报告中直观看到检索是否匹配。
            rows.append(
                {
                    "query_text": texts[query_index],
                    "top1_matched_text": texts[predicted_index],
                    "is_correct": predicted_index == query_index,
                }
            )
    return correct / len(texts), rows


def train(args: argparse.Namespace) -> dict[str, object]:
    # 固定随机种子控制参数初始化和 DataLoader shuffle，保证结果可复现。
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 词表、Dataset、DataLoader 构成多模态实验的数据入口。
    token_to_id = build_vocab()
    dataset = MultimodalDataset(token_to_id)
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=True,
        collate_fn=collate_batch,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        collate_fn=collate_batch,
    )

    # image_dim 由人工图像属性长度决定，projection_dim 是共享空间维度。
    image_dim = len(PAIRS[0][1])
    model = ToyCLIP(
        vocab_size=len(token_to_id),
        image_dim=image_dim,
        projection_dim=args.projection_dim,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for text_ids, image_features, _ in loader:
            # logits 是 batch 内所有图文组合的相似度矩阵。
            logits = model(text_ids, image_features)
            loss = contrastive_loss(logits)

            # 对比学习的梯度会同时拉近正配对、推远负配对。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每个 epoch 记录 Recall@1，观察共享空间对齐是否逐步改善。
        recall, _ = evaluate_retrieval(model, eval_loader)
        history.append(
            {
                "epoch": epoch,
                "loss": round(float(loss.item()), 4),
                "recall_at_1": round(recall, 4),
            }
        )

    # 最终评估同时保存数值指标和可读检索样例。
    recall, retrieval_rows = evaluate_retrieval(model, eval_loader)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "toy_multimodal_retrieval.pt")
    result = {
        # 记录 num_pairs 和 vocab_size，帮助解释该 toy 实验的规模限制。
        "task": "toy_multimodal_contrastive_retrieval",
        "num_pairs": len(PAIRS),
        "vocab_size": len(token_to_id),
        "final_recall_at_1": round(recall, 4),
        "history": history,
        "retrieval_examples": retrieval_rows,
    }
    (output_dir / "toy_multimodal_retrieval_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    # 暴露投影维度和学习率，便于后续观察对齐空间容量与收敛速度。
    parser = argparse.ArgumentParser(description="Toy multimodal contrastive retrieval")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--projection-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    # main 入口保持简洁，便于后续把 train 嵌入 notebook 或周报实验流程。
    require_torch()
    result = train(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"图文检索 Recall@1：{result['final_recall_at_1']}")


if __name__ == "__main__":
    main()
