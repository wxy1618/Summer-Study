# -*- coding: utf-8 -*-
"""Toy Seq2Seq translation with additive attention and tuning records.

本脚本对应第六周翻译任务。注意力机制第四周已学习过，因此这里重点展示
翻译任务中的源端编码、目标端解码、teacher forcing、验证和调参记录。
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path

try:
    # PyTorch 用于训练 Seq2Seq 模型；脚本不会安装或修改环境。
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None

    class _MissingNN:
        Module = object

    nn = _MissingNN()
    F = None


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"


@dataclass(frozen=True)
class Pair:
    """A tiny parallel sentence pair."""

    # src 是英文源句，tgt 是用空格分开的中文目标 token，便于不依赖中文分词器。
    src: str
    tgt: str


# 平行语料很小，只用于验证 Seq2Seq 训练流程和注意力对齐接口。
PAIRS = [
    Pair("i like machine learning", "我 喜欢 机器 学习"),
    Pair("attention improves translation", "注意力 改进 翻译"),
    Pair("the model answers questions", "模型 回答 问题"),
    Pair("language generation needs evaluation", "语言 生成 需要 评价"),
    Pair("coreference links mentions", "共指 连接 指称"),
    Pair("relation extraction finds facts", "关系 抽取 发现 事实"),
]


def require_torch() -> None:
    """Exit gracefully when PyTorch is unavailable."""

    # 课程要求不配置环境，因此依赖缺失时只提示。
    if torch is None:
        print("当前环境未安装 PyTorch；本脚本不会自动安装依赖。")
        raise SystemExit(0)


def tokenize_english(text: str) -> list[str]:
    """Tokenize the English source sentence."""

    # 源端使用正则分词，目标端已经人工空格切分，避免引入额外工具。
    return TOKEN_RE.findall(text.lower())


def tokenize_target(text: str) -> list[str]:
    """Tokenize the target sentence."""

    # 中文目标句用空格预切分，保证实验可复现且不依赖 jieba 等外部库。
    return text.split()


def build_vocab(sequences: list[list[str]]) -> tuple[dict[str, int], list[str]]:
    """Build a vocabulary with special tokens."""

    # 特殊 token 固定在词表前部，便于 padding、解码起止和未知词处理。
    words = sorted({token for sequence in sequences for token in sequence})
    id_to_token = [PAD, BOS, EOS, UNK] + words
    token_to_id = {token: index for index, token in enumerate(id_to_token)}
    return token_to_id, id_to_token


def encode(
    sequence: list[str],
    vocab: dict[str, int],
    add_boundaries: bool,
) -> list[int]:
    """Encode tokens as ids, optionally with BOS/EOS."""

    # 目标端需要 BOS/EOS 训练自回归解码，源端只需要 EOS 表示输入结束。
    ids = [vocab.get(token, vocab[UNK]) for token in sequence]
    if add_boundaries:
        return [vocab[BOS]] + ids + [vocab[EOS]]
    return ids + [vocab[EOS]]


def pad_batch(rows: list[list[int]], pad_id: int) -> "torch.Tensor":
    """Pad a list of id sequences to a rectangular tensor."""

    # RNN batch 需要矩阵输入；padding 后再用 mask 排除无效位置。
    max_len = max(len(row) for row in rows)
    padded = [row + [pad_id] * (max_len - len(row)) for row in rows]
    return torch.tensor(padded, dtype=torch.long)


class Seq2SeqAttention(nn.Module):
    """GRU encoder-decoder with additive attention."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        # 源端和目标端语言不同，因此各自使用独立 embedding 表。
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim, padding_idx=0)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_dim, padding_idx=0)

        # 编码器读取完整源句，解码器逐步生成目标句。
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(embedding_dim + hidden_dim, hidden_dim, batch_first=True)

        # additive attention 将 decoder state 与每个 encoder state 做匹配。
        self.attn_query = nn.Linear(hidden_dim, hidden_dim)
        self.attn_key = nn.Linear(hidden_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1)
        self.output = nn.Linear(hidden_dim * 2, tgt_vocab_size)

    def attend(
        self,
        decoder_state: "torch.Tensor",
        encoder_states: "torch.Tensor",
        src_mask: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Compute attention context for one decoder step."""

        # decoder_state 形状为 [batch, hidden]，需要扩展到源端长度维度。
        query = self.attn_query(decoder_state).unsqueeze(1)
        keys = self.attn_key(encoder_states)
        scores = self.attn_score(torch.tanh(query + keys)).squeeze(-1)

        # mask 掉源端 PAD，防止注意力把概率分给无意义位置。
        scores = scores.masked_fill(~src_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_states).squeeze(1)
        return context, weights

    def forward(
        self,
        src_ids: "torch.Tensor",
        tgt_ids: "torch.Tensor",
        teacher_forcing_ratio: float,
    ) -> "torch.Tensor":
        """Train with teacher forcing."""

        # encoder_states 保留每个源端位置的隐藏状态，供 attention 使用。
        src_mask = src_ids.ne(0)
        encoder_states, hidden = self.encoder(self.src_embedding(src_ids))

        # 解码从 BOS 开始，逐步预测目标序列的下一个 token。
        batch_size, tgt_len = tgt_ids.shape
        input_token = tgt_ids[:, 0]
        logits_steps = []

        # 训练时第 t 步预测 tgt_ids[:, t]，因此循环从 1 开始。
        for step in range(1, tgt_len):
            embedded = self.tgt_embedding(input_token).unsqueeze(1)
            decoder_state = hidden[-1]
            context, _ = self.attend(decoder_state, encoder_states, src_mask)
            decoder_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
            output_state, hidden = self.decoder(decoder_input, hidden)
            step_logits = self.output(
                torch.cat([output_state.squeeze(1), context], dim=-1),
            )
            logits_steps.append(step_logits)

            # teacher forcing 在训练早期稳定解码器，比例越高越接近监督条件生成。
            use_teacher = random.random() < teacher_forcing_ratio
            input_token = tgt_ids[:, step] if use_teacher else step_logits.argmax(dim=1)
        return torch.stack(logits_steps, dim=1)

    def translate(
        self,
        src_ids: "torch.Tensor",
        bos_id: int,
        eos_id: int,
        max_len: int,
    ) -> list[int]:
        """Greedy translation for one source sentence."""

        # 推理阶段没有真实目标词，只能使用上一步预测作为下一步输入。
        self.eval()
        src_mask = src_ids.ne(0)
        encoder_states, hidden = self.encoder(self.src_embedding(src_ids))
        input_token = torch.tensor([bos_id], dtype=torch.long)
        outputs = []

        # greedy 解码简单可复现，适合作为小型翻译实验的基线。
        with torch.no_grad():
            for _ in range(max_len):
                embedded = self.tgt_embedding(input_token).unsqueeze(1)
                context, _ = self.attend(hidden[-1], encoder_states, src_mask)
                decoder_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
                output_state, hidden = self.decoder(decoder_input, hidden)
                logits = self.output(
                    torch.cat([output_state.squeeze(1), context], dim=-1),
                )
                next_id = int(logits.argmax(dim=1).item())
                if next_id == eos_id:
                    break
                outputs.append(next_id)
                input_token = torch.tensor([next_id], dtype=torch.long)
        return outputs


def prepare_data() -> tuple["torch.Tensor", "torch.Tensor", dict[str, object]]:
    """Build vocabularies and padded tensors."""

    # 源端和目标端分别分词，并分别构建词表。
    src_tokens = [tokenize_english(pair.src) for pair in PAIRS]
    tgt_tokens = [tokenize_target(pair.tgt) for pair in PAIRS]
    src_vocab, src_id_to_token = build_vocab(src_tokens)
    tgt_vocab, tgt_id_to_token = build_vocab(tgt_tokens)

    # 目标端加 BOS/EOS，源端加 EOS；padding id 均为 0。
    src_rows = [
        encode(tokens, src_vocab, add_boundaries=False)
        for tokens in src_tokens
    ]
    tgt_rows = [encode(tokens, tgt_vocab, add_boundaries=True) for tokens in tgt_tokens]
    metadata = {
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "src_id_to_token": src_id_to_token,
        "tgt_id_to_token": tgt_id_to_token,
    }
    return (
        pad_batch(src_rows, src_vocab[PAD]),
        pad_batch(tgt_rows, tgt_vocab[PAD]),
        metadata,
    )


def token_accuracy(logits: "torch.Tensor", targets: "torch.Tensor") -> float:
    """Compute non-padding target token accuracy."""

    # logits 对应目标第 1..T 位，因此 targets 也右移到同一口径。
    gold = targets[:, 1:]
    predictions = logits.argmax(dim=-1)
    mask = gold.ne(0)
    correct = predictions.eq(gold) & mask
    return float(correct.sum().item() / max(mask.sum().item(), 1))


def train_config(
    src_ids: "torch.Tensor",
    tgt_ids: "torch.Tensor",
    metadata: dict[str, object],
    args: argparse.Namespace,
    config: dict[str, float | int],
) -> dict[str, object]:
    """Train one translation configuration."""

    # 每组配置固定随机性，确保调参比较主要来自超参数而非初始化波动。
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = Seq2SeqAttention(
        src_vocab_size=len(metadata["src_id_to_token"]),
        tgt_vocab_size=len(metadata["tgt_id_to_token"]),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    history = []

    # 小平行语料直接全量 batch 训练，重点观察流程而非泛化。
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(src_ids, tgt_ids, args.teacher_forcing)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_ids[:, 1:].reshape(-1),
            ignore_index=0,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 记录若干 epoch 的 loss 和 token accuracy，体现调优过程。
        if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 3, 1) == 0:
            history.append(
                {
                    "epoch": epoch,
                    "loss": round(float(loss.item()), 4),
                    "token_accuracy": round(
                        token_accuracy(logits.detach(), tgt_ids),
                        4,
                    ),
                }
            )

    # 用第一条样例做定性翻译检查，便于发现解码是否完全失效。
    tgt_id_to_token = metadata["tgt_id_to_token"]
    translated_ids = model.translate(
        src_ids[:1],
        bos_id=metadata["tgt_vocab"][BOS],
        eos_id=metadata["tgt_vocab"][EOS],
        max_len=8,
    )
    translation = " ".join(tgt_id_to_token[index] for index in translated_ids)
    return {
        "config": config,
        "final_loss": history[-1]["loss"],
        "final_token_accuracy": history[-1]["token_accuracy"],
        "history": history,
        "sample_source": PAIRS[0].src,
        "sample_prediction": translation,
        "sample_gold": PAIRS[0].tgt,
    }


def run_tuning(args: argparse.Namespace) -> dict[str, object]:
    """Run compact hyperparameter tuning for the translation model."""

    # 数据准备与调参分离，便于确认所有配置使用同一批语料。
    src_ids, tgt_ids, metadata = prepare_data()
    search_space = {
        "embedding_dim": [24],
        "hidden_dim": [32, 48],
        "learning_rate": [0.01, 0.02],
    }
    trials = []

    # 搜索隐藏维度和学习率，模拟阶段二“完成模型调优”的要求。
    for values in product(*search_space.values()):
        config = dict(zip(search_space.keys(), values))
        trials.append(train_config(src_ids, tgt_ids, metadata, args, config))

    best = max(
        trials,
        key=lambda row: (row["final_token_accuracy"], -row["final_loss"]),
    )
    return {
        "task": "translation_seq2seq_attention",
        "num_pairs": len(PAIRS),
        "best_result": best,
        "trials": trials,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    # teacher_forcing 暴露出来，便于观察监督输入比例对收敛的影响。
    parser = argparse.ArgumentParser(description="Toy Seq2Seq attention translator")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--teacher-forcing", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    """Run translation tuning and save the result."""

    # 脚本只生成和运行作业代码，不做环境配置。
    require_torch()
    result = run_tuning(parse_args())
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "translation_seq2seq_attention_results.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
