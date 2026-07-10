# -*- coding: utf-8 -*-
"""A small word-level GRU language model.

本脚本对应第五周“语言模型与 RNN”部分。脚本使用内置小语料训练词级
GRU 语言模型，并记录 cross entropy、perplexity 和采样生成文本。
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
    # 训练 GRU 语言模型依赖 PyTorch；导入失败时仍允许脚本给出清晰提示。
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
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

# 特殊 token 显式进入词表，能把序列边界、未知词和 padding 分开建模。
# 对语言模型而言，<bos>/<eos> 是估计句子概率时非常重要的边界信号。

CORPUS = [
    "language models estimate the probability of token sequences",
    "a recurrent network updates hidden states step by step",
    "gated recurrent units reduce the difficulty of long dependencies",
    "perplexity measures how uncertain a language model remains",
    "word embeddings transform sparse ids into dense vectors",
    "attention models compare tokens without strict recurrent order",
    "pretraining uses raw text as a self supervised learning signal",
    "fine tuning adapts a general representation to a target task",
    "dependency parsing describes syntactic relations between tokens",
    "careful experiments record data assumptions metrics and errors",
]


def require_torch() -> None:
    # 依赖检查放在 main 入口而不是模块顶层，便于静态阅读和代码复用。
    if torch is None:
        print("当前环境未安装 PyTorch；本脚本不会自动安装依赖。")
        raise SystemExit(0)


def tokenize(text: str) -> list[str]:
    # 小实验使用英文正则分词，避免引入外部分词器造成环境依赖。
    # 真实 NLP 研究中需要针对语种和任务选择词级、子词级或字符级切分。
    return TOKEN_RE.findall(text.lower())


def build_vocab(sentences: Iterable[str]) -> tuple[dict[str, int], list[str]]:
    # 词表按照频次稳定排序，保证同一随机种子下 token id 可复现。
    counter: Counter[str] = Counter()
    for sentence in sentences:
        counter.update(tokenize(sentence))

    # SPECIAL_TOKENS 放在词表前部，便于在训练和解码阶段用固定 id 引用。
    words = sorted(counter, key=lambda word: (-counter[word], word))
    id_to_word = SPECIAL_TOKENS + words
    word_to_id = {word: idx for idx, word in enumerate(id_to_word)}
    return word_to_id, id_to_word


def encode_sentences(sentences: Iterable[str], word_to_id: dict[str, int]) -> list[int]:
    """Flatten sentences into one token stream with BOS/EOS markers."""

    # 把句子串成一个长 token 流，是词级语言模型常见的训练数据组织方式。
    # BOS/EOS 防止模型把前一句末尾和后一句开头误认为自然连续。
    token_ids: list[int] = []
    for sentence in sentences:
        token_ids.append(word_to_id["<bos>"])
        token_ids.extend(
            word_to_id.get(token, word_to_id["<unk>"])
            for token in tokenize(sentence)
        )
        token_ids.append(word_to_id["<eos>"])
    return token_ids


def make_windows(
    token_ids: list[int],
    context_size: int,
) -> list[tuple[list[int], int]]:
    """Build fixed-length contexts for next-token prediction."""

    # 语言模型的自监督信号来自“右移一位”的目标；
    # 这里用固定窗口近似完整历史，降低教学实验的计算复杂度。
    windows: list[tuple[list[int], int]] = []
    for index in range(context_size, len(token_ids)):
        context = token_ids[index - context_size:index]
        target = token_ids[index]
        windows.append((context, target))
    return windows


class LanguageModelDataset(Dataset):
    """Dataset of context windows and next-token labels."""

    def __init__(self, samples: list[tuple[list[int], int]]) -> None:
        # samples 已经是数值化窗口，Dataset 只承担索引访问职责。
        self.samples = samples

    def __len__(self) -> int:
        # 样本数同时决定一个 epoch 中参数更新的总规模。
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        # 训练时每个样本由 context 序列和单个 next-token 标签组成。
        context, target = self.samples[index]
        return (
            torch.tensor(context, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


class GRULanguageModel(nn.Module):
    """Embedding + GRU + linear projection language model."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        # Embedding 层把离散 token id 映射到连续空间，是序列模型的输入接口。
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # GRU 用门控状态压缩历史上下文，比普通 RNN 更适合较长依赖。
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        # 输出层把最后一个隐藏状态投影回词表空间，得到 next-token logits。
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        # 输入形状为 [batch, context_size]，输出是每个样本对整个词表的打分。
        embedded = self.embedding(input_ids)
        hidden_states, _ = self.gru(embedded)
        # 只取最后时刻隐藏状态，因为它已经聚合了固定窗口内的上下文。
        last_state = hidden_states[:, -1, :]
        return self.output(last_state)


def split_samples(
    samples: list[tuple[list[int], int]],
    valid_ratio: float,
) -> tuple[list[tuple[list[int], int]], list[tuple[list[int], int]]]:
    """Deterministically split windows into train and validation sets."""

    # 这里采用顺序切分，保留文本流的时间结构；大规模语料可用文档级随机切分。
    split_index = max(1, int(len(samples) * (1 - valid_ratio)))
    return samples[:split_index], samples[split_index:]


def run_epoch(
    model: GRULanguageModel,
    loader: "DataLoader",
    optimizer: "torch.optim.Optimizer | None",
) -> float:
    """Train or evaluate one epoch depending on whether optimizer is given."""

    # optimizer 为 None 时进入验证模式；同一函数复用训练和评估统计逻辑。
    total_loss = 0.0
    total_items = 0
    is_training = optimizer is not None
    model.train(is_training)

    for input_ids, targets in loader:
        # cross entropy 等价于最大化真实下一个 token 的条件对数似然。
        logits = model(input_ids)
        loss = F.cross_entropy(logits, targets)

        if is_training:
            # 语言模型训练中梯度可能随时间展开变大，裁剪能提升小模型稳定性。
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 按样本数累计 loss，避免最后一个小 batch 对平均值产生偏差。
        batch_size = input_ids.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / total_items


def sample_text(
    model: GRULanguageModel,
    word_to_id: dict[str, int],
    id_to_word: list[str],
    seed_words: list[str],
    context_size: int,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate a short sequence by repeatedly sampling the next token."""

    # 采样阶段不更新参数，但保留随机采样以观察模型学到的分布特征。
    model.eval()
    # seed_words 不足窗口长度时用 BOS 左填充，保持输入维度固定。
    ids = [word_to_id.get(word, word_to_id["<unk>"]) for word in seed_words]
    ids = (
        [word_to_id["<bos>"]] * max(0, context_size - len(ids))
        + ids[-context_size:]
    )

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 每次只使用最近 context_size 个 token，和训练窗口保持一致。
            context = torch.tensor([ids[-context_size:]], dtype=torch.long)
            # temperature 控制分布尖锐程度；温度越低，输出越接近贪心解码。
            logits = model(context).squeeze(0) / temperature
            probabilities = torch.softmax(logits, dim=0)
            next_id = int(torch.multinomial(probabilities, num_samples=1).item())
            if id_to_word[next_id] == "<eos>":
                break
            ids.append(next_id)

    # 解码时去掉特殊 token，使生成文本更接近自然语言观察结果。
    generated = [
        id_to_word[token_id]
        for token_id in ids
        if id_to_word[token_id] not in SPECIAL_TOKENS
    ]
    return " ".join(generated)


def train(args: argparse.Namespace) -> dict[str, object]:
    # 固定 Python 和 PyTorch 随机种子，使初始化、shuffle 和采样生成可复现。
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 预处理链路严格对应 NLP 基本流程：词表 -> token 流 -> 训练窗口。
    word_to_id, id_to_word = build_vocab(CORPUS)
    token_stream = encode_sentences(CORPUS, word_to_id)
    samples = make_windows(token_stream, context_size=args.context_size)
    train_samples, valid_samples = split_samples(samples, args.valid_ratio)

    # train_loader 打乱窗口顺序，valid_loader 保持固定顺序，便于稳定评估。
    train_loader = DataLoader(
        LanguageModelDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        LanguageModelDataset(valid_samples),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 模型规模故意较小，适合在 CPU 环境中快速验证语言模型训练流程。
    model = GRULanguageModel(
        vocab_size=len(id_to_word),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # best_valid_loss 用于保存最优权重，避免最后一个 epoch 恰好过拟合。
    history: list[dict[str, float | int]] = []
    best_valid_loss = math.inf
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        valid_loss = run_epoch(model, valid_loader, optimizer=None)
        # perplexity 是 cross entropy 的指数形式，便于解释模型平均不确定性。
        perplexity = math.exp(min(valid_loss, 20.0))
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "valid_loss": round(valid_loss, 4),
                "valid_perplexity": round(perplexity, 4),
            }
        )

        if valid_loss < best_valid_loss:
            # 保存验证集最优模型，体现“以泛化表现选择模型”的实验规范。
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), output_dir / "rnn_language_model.pt")

    # 生成样例不是严格指标，但能帮助观察模型是否学到语料主题和词序偏好。
    samples_text = [
        sample_text(
            model,
            word_to_id,
            id_to_word,
            seed_words=["language", "models"],
            context_size=args.context_size,
            max_new_tokens=10,
            temperature=args.temperature,
        ),
        sample_text(
            model,
            word_to_id,
            id_to_word,
            seed_words=["attention", "models"],
            context_size=args.context_size,
            max_new_tokens=10,
            temperature=args.temperature,
        ),
    ]

    # 结果 JSON 记录数据规模、指标曲线和生成样例，便于直接写入实验报告。
    result = {
        "task": "gru_word_language_model",
        "vocab_size": len(id_to_word),
        "num_train_windows": len(train_samples),
        "num_valid_windows": len(valid_samples),
        "best_valid_loss": round(best_valid_loss, 4),
        "best_valid_perplexity": round(math.exp(min(best_valid_loss, 20.0)), 4),
        "history": history,
        "generated_samples": samples_text,
    }
    (output_dir / "rnn_language_model_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    # 命令行参数让训练轮数、模型维度和采样温度都可以独立调节。
    parser = argparse.ArgumentParser(description="Toy GRU language model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    # 入口函数保持简单，便于把 train 函数作为 notebook 实验的一部分复用。
    require_torch()
    result = train(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"最佳验证困惑度：{result['best_valid_perplexity']}")


if __name__ == "__main__":
    main()
