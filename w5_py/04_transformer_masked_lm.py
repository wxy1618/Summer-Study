# -*- coding: utf-8 -*-
"""A miniature Transformer masked language model.

本脚本对应第五周“Transformer 模型、预训练模型”部分。实验使用小语料
训练 Masked Language Modeling，让模型根据双向上下文恢复被遮蔽 token。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

try:
    # 本脚本只在训练小型 MLM 时依赖 PyTorch；缺失依赖时按课程要求温和退出。
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None

    class _MissingNN:
        Module = object

    nn = _MissingNN()
    F = None
    DataLoader = None
    TensorDataset = None


TOKEN_RE = re.compile(r"[a-zA-Z]+|[0-9]+")
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[MASK]"]

# MLM 任务必须显式区分 [MASK] 和普通未知词 [UNK]：
# 前者是训练信号，后者是词表覆盖不足时的退化表示。

SENTENCES = [
    "transformers learn contextual representations with self attention",
    "masked language modeling predicts hidden tokens from both sides",
    "pretrained encoders can transfer knowledge to classification tasks",
    "queries keys and values are projected from token embeddings",
    "positional encodings inject word order into attention models",
    "large language models combine scale data and optimization",
    "dependency structure can help explain relations between words",
    "word vectors provide a dense interface for discrete language",
    "graduate research reports should preserve reproducible settings",
    "evaluation should include metrics examples and error analysis",
]


def require_torch() -> None:
    # 这里不做环境安装，保证脚本行为和“只生成作业代码”的要求一致。
    if torch is None:
        print("当前环境未安装 PyTorch；本脚本不会自动安装依赖。")
        raise SystemExit(0)


def tokenize(text: str) -> list[str]:
    # 简化分词能让预训练目标更透明；真实 BERT 类模型通常使用 WordPiece/BPE。
    return TOKEN_RE.findall(text.lower())


def build_vocab(sentences: Iterable[str]) -> tuple[dict[str, int], list[str]]:
    # 词表由小语料直接构建，SPECIAL_TOKENS 放在前面以便固定语义 id。
    counter: Counter[str] = Counter()
    for sentence in sentences:
        counter.update(tokenize(sentence))

    # 按频次和字母序排序可以减少运行间差异，利于复现实验结果。
    words = sorted(counter, key=lambda word: (-counter[word], word))
    id_to_token = SPECIAL_TOKENS + words
    token_to_id = {token: idx for idx, token in enumerate(id_to_token)}
    return token_to_id, id_to_token


def encode_sentences(
    sentences: Iterable[str],
    token_to_id: dict[str, int],
    max_length: int,
) -> "torch.Tensor":
    """Encode and pad sentences to a fixed length."""

    # Transformer 以 batch 矩阵输入，因此变长句子需要截断和 padding。
    # 这里不使用 segment embedding，因为小实验只有单句输入。
    encoded_rows: list[list[int]] = []
    pad_id = token_to_id["[PAD]"]
    unk_id = token_to_id["[UNK]"]
    cls_id = token_to_id["[CLS]"]

    for sentence in sentences:
        # [CLS] 放在句首，模拟预训练模型常用的全局聚合位置。
        ids = [cls_id]
        ids.extend(token_to_id.get(token, unk_id) for token in tokenize(sentence))
        # 截断保证所有样本长度不超过 max_length，padding 保证 batch 维度一致。
        ids = ids[:max_length]
        ids.extend([pad_id] * (max_length - len(ids)))
        encoded_rows.append(ids)
    return torch.tensor(encoded_rows, dtype=torch.long)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used by the original Transformer."""

    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()
        # 自注意力本身不包含顺序偏置，因此必须额外注入位置信息。
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        div_terms = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        # 正弦和余弦交替编码不同频率的位置模式，允许模型外推相对位置关系。
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * div_terms)
        encoding[:, 1::2] = torch.cos(positions * div_terms)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, embeddings: "torch.Tensor") -> "torch.Tensor":
        # positional encoding 与 token embedding 相加，保持表示维度不变。
        return embeddings + self.encoding[:, :embeddings.size(1), :]


class MiniTransformerMLM(nn.Module):
    """Transformer encoder with a token-level prediction head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        max_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # 输入 embedding 是 token id 到连续向量的映射，也是 MLM 的可训练表示表。
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model, max_length)
        # EncoderLayer 封装多头注意力、前馈网络、残差连接和归一化。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        try:
            # 新版 PyTorch 默认启用 nested tensor 优化，会在部分环境打印实验性提示。
            # 这里关闭它只影响内部加速路径，不改变 Transformer 的教学逻辑。
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            # 兼容旧版 PyTorch：旧 API 没有 enable_nested_tensor 参数。
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出层在每个位置上预测词表分布，而不是只预测句子级标签。
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: "torch.Tensor", pad_id: int) -> "torch.Tensor":
        # padding_mask 告诉注意力层哪些位置只是补齐，不应作为上下文参与计算。
        padding_mask = input_ids.eq(pad_id)
        embeddings = self.position(self.embedding(input_ids))
        # Transformer Encoder 使用双向上下文，符合 BERT 风格 MLM 的建模假设。
        hidden = self.encoder(embeddings, src_key_padding_mask=padding_mask)
        return self.output(hidden)


def mask_inputs(
    batch_ids: "torch.Tensor",
    token_to_id: dict[str, int],
    mask_probability: float,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Create MLM inputs and labels from a clean token-id batch."""

    # labels 中的 -100 是 PyTorch cross_entropy 的 ignore_index 默认值，
    # 这样 loss 只会在被 mask 的位置计算。
    pad_id = token_to_id["[PAD]"]
    cls_id = token_to_id["[CLS]"]
    mask_id = token_to_id["[MASK]"]

    # inputs 是模型看到的句子，labels 保留原始 token 作为监督信号。
    inputs = batch_ids.clone()
    labels = torch.full_like(batch_ids, fill_value=-100)
    # PAD 和 CLS 不参与遮蔽，否则模型会学习无意义的特殊符号恢复任务。
    can_mask = ~batch_ids.eq(pad_id) & ~batch_ids.eq(cls_id)
    random_values = torch.rand(batch_ids.shape)
    selected = (random_values < mask_probability) & can_mask

    # 小 batch 中有时一个 token 都没选中，强制每行至少 mask 一个非特殊 token。
    # 这保证每个 batch 都能产生有效 loss，避免教学小数据中出现空监督。
    for row_index in range(batch_ids.size(0)):
        if not selected[row_index].any():
            candidates = torch.nonzero(can_mask[row_index], as_tuple=False).flatten()
            if len(candidates) > 0:
                selected[row_index, candidates[0]] = True

    # 被选中的位置替换为 [MASK]，原始 token 写入 labels。
    labels[selected] = batch_ids[selected]
    inputs[selected] = mask_id
    return inputs, labels


def evaluate_mask_accuracy(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
) -> tuple[int, int]:
    """Count correct predictions only on masked positions."""

    # MLM 的准确率只在被遮蔽 token 上有意义；未遮蔽位置没有监督目标。
    predictions = logits.argmax(dim=-1)
    active = labels.ne(-100)
    correct = int(predictions[active].eq(labels[active]).sum().item())
    total = int(active.sum().item())
    return correct, total


def decode_mask_predictions(
    model: MiniTransformerMLM,
    token_to_id: dict[str, int],
    id_to_token: list[str],
    max_length: int,
) -> list[dict[str, object]]:
    """Inspect several hand-written masked examples."""

    # 手写样例用于定性检查模型是否能利用左右上下文恢复词义。
    # 这类定性观察不能替代指标，但能帮助发现 token 处理错误。
    examples = [
        "transformers learn [MASK] representations with self attention",
        "masked language modeling predicts hidden [MASK] from both sides",
        "positional encodings inject word [MASK] into attention models",
    ]
    pad_id = token_to_id["[PAD]"]
    mask_id = token_to_id["[MASK]"]
    rows: list[dict[str, object]] = []

    model.eval()
    with torch.no_grad():
        for text in examples:
            # 特殊 token 需要保留原样；普通词再统一小写。
            # 若把 [MASK] 小写成 [mask]，会被当作未知词，预测检查就失效。
            tokens = ["[CLS]"] + [
                token if token in token_to_id else token.lower()
                for token in text.split()
            ]
            ids = [token_to_id.get(token, token_to_id["[UNK]"]) for token in tokens]
            # 推理样例也必须遵循训练时的截断和 padding 规则。
            ids = ids[:max_length] + [pad_id] * max(0, max_length - len(ids))
            input_ids = torch.tensor([ids], dtype=torch.long)
            logits = model(input_ids, pad_id=pad_id)
            mask_positions = torch.nonzero(
                input_ids[0].eq(mask_id),
                as_tuple=False,
            ).flatten()
            predictions = []
            for position in mask_positions.tolist():
                # 输出 top-3 而不是单个预测，有助于观察模型的不确定性结构。
                top_ids = torch.topk(logits[0, position], k=3).indices.tolist()
                predictions.append([id_to_token[token_id] for token_id in top_ids])
            rows.append({"input": text, "top_predictions": predictions})
    return rows


def train(args: argparse.Namespace) -> dict[str, object]:
    # 固定随机种子控制初始化、shuffle 和随机 mask，保证指标曲线可复现。
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 预训练任务的数据准备包括词表、定长编码和 TensorDataset 包装。
    token_to_id, id_to_token = build_vocab(SENTENCES)
    encoded = encode_sentences(SENTENCES, token_to_id, max_length=args.max_length)
    dataset = TensorDataset(encoded)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 小模型使用 AdamW，符合 Transformer 训练中常用的解耦权重衰减优化器。
    model = MiniTransformerMLM(
        vocab_size=len(id_to_token),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_length=args.max_length,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    pad_id = token_to_id["[PAD]"]

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        # 只累计 mask 位置的 loss 和准确率，避免 padding/非监督位置稀释指标。
        total_loss = 0.0
        total_correct = 0
        total_masked = 0

        for (batch_ids,) in loader:
            # 每个 batch 动态 mask，相当于对同一句子生成不同预训练视角。
            inputs, labels = mask_inputs(batch_ids, token_to_id, args.mask_probability)
            logits = model(inputs, pad_id=pad_id)
            loss = F.cross_entropy(logits.view(-1, len(id_to_token)), labels.view(-1))

            # 梯度裁剪防止小数据高学习率下注意力层产生过大更新。
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # detach 后只做指标统计，避免把评估计算接入反向图。
            correct, masked = evaluate_mask_accuracy(logits.detach(), labels)
            total_loss += float(loss.item()) * masked
            total_correct += correct
            total_masked += masked

        history.append(
            {
                "epoch": epoch,
                "loss": round(total_loss / max(total_masked, 1), 4),
                "mask_accuracy": round(total_correct / max(total_masked, 1), 4),
            }
        )

    # 保存模型权重和 JSON 结果，分别服务于复现实验和周报撰写。
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "transformer_masked_lm.pt")
    result = {
        "task": "mini_transformer_masked_language_modeling",
        "vocab_size": len(id_to_token),
        "max_length": args.max_length,
        "final_mask_accuracy": history[-1]["mask_accuracy"],
        "history": history,
        "masked_examples": decode_mask_predictions(
            model,
            token_to_id,
            id_to_token,
            args.max_length,
        ),
    }
    (output_dir / "transformer_masked_lm_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    # 超参数暴露给命令行，便于比较层数、头数、mask 比例对小实验的影响。
    parser = argparse.ArgumentParser(description="Mini Transformer MLM experiment")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-probability", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    # main 入口只做依赖检查和结果打印，训练细节保留在 train 中。
    require_torch()
    result = train(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"最终 mask accuracy = {result['final_mask_accuracy']}")


if __name__ == "__main__":
    main()
