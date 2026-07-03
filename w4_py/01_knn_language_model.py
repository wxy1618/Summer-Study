"""用纯 Python 演示 KNN-LM 的检索与概率插值。

脚本构造一个小型语料库，同时训练平滑 Bigram 模型并建立上下文记忆库。
它不是工业级语言模型，而是用于理解课程 P31 中 KNN-LM 的核心流程。

本模块包含：
- 分词工具
- 稀疏向量余弦相似度计算
- Bigram 语言模型（带加法平滑）
- KNN 记忆库（存储上下文‑下一个 token 对）
- 基础模型与记忆库的线性插值
- 演示主流程
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

# ---------- 常量定义 ----------
# 用于匹配英文单词（包括缩写形式）
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
START_TOKEN = "<start>"   # 句子起始标记
END_TOKEN = "<end>"       # 句子结束标记



def tokenize(text: str) -> list[str]:
    """将英文文本转换为小写 token 列表。

    使用正则表达式提取所有匹配的单词，并统一转为小写。
    忽略数字、标点符号和其他非字母字符。

    参数：
        text (str): 原始英文文本

    返回：
        list[str]: 小写 token 序列
    
    """

    return TOKEN_PATTERN.findall(text.lower())


def context_vector(tokens: Iterable[str]) -> Counter[str]:
    """使用词频向量表示一个短上下文。
    
    将 token 序列转换为 Counter 对象，每个 token 的出现次数作为特征值。
    这是一种简单的词袋表示，忽略顺序。

    参数：
        tokens (Iterable[str]): token 序列（如上下文窗口）

    返回：
        Counter[str]: 词频计数器
    """

    return Counter(tokens)


def cosine_similarity(
    first: Counter[str],
    second: Counter[str],
) -> float:
    """计算两个稀疏词频向量的余弦相似度。

    余弦相似度 = (A · B) / (||A|| * ||B||)。
    为提升性能，遍历较短的字典，减少乘法运算。
    若任一向量为零向量，则相似度定义为 0.0（无方向）。

    参数：
        first (Counter[str]): 第一个向量
        second (Counter[str]): 第二个向量

    返回：
        float: 余弦相似度，范围 [0.0, 1.0]"""

    # 只遍历较短的词典，可以减少无意义的乘法操作。
    if len(first) > len(second):
        first, second = second, first

    # 计算点积（仅遍历第一个向量的键，检查第二个向量中是否存在）
    dot_product = sum(value * second[token] for token, value in first.items())
    # 计算 L2 范数
    first_norm = math.sqrt(sum(value**2 for value in first.values()))
    second_norm = math.sqrt(sum(value**2 for value in second.values()))

    # 空上下文没有方向，因此将相似度定义为零。
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return dot_product / (first_norm * second_norm)


@dataclass(frozen=True)
class MemoryEntry:
    """保存一个上下文向量及其后续真实 token。
    
    保存一个上下文向量（词频 Counter）、对应的下一个 token 以及原始的上下文 token 元组。
    使用 frozen=True 保证不可变性，便于哈希和作为字典键（虽然本实现未用作键）。
    
    """

    key: Counter[str]
    value: str
    source_context: tuple[str, ...]


class BigramLanguageModel:
    """带加法平滑的 Bigram 语言模型。
    
    基于统计的 n-gram 模型，仅考虑前一个 token 来预测下一个 token。
    使用加法（拉普拉斯）平滑避免零概率问题。
    
    """

    #初始化 Bigram 模型。
    def __init__(self, smoothing: float = 0.2) -> None:
        self.smoothing = smoothing
        # 使用 defaultdict 存储前词到后词计数器的映射
        self.counts: dict[str, Counter[str]] = defaultdict(Counter)
        # 词汇表，用于平滑时计算分母中的 |V|
        self.vocabulary: set[str] = set()

    def fit(self, sentences: Iterable[str]) -> None:
        """统计语料中的相邻 token。"""

        for sentence in sentences:
            # 每个句子单独增加起止标记，避免跨句建立错误关系。
            tokens = [START_TOKEN, *tokenize(sentence), END_TOKEN]
            # 更新词汇表
            self.vocabulary.update(tokens[1:])

            # 统计相邻对：当前 token 和下一个 token
            for current_token, next_token in zip(tokens, tokens[1:]):
                self.counts[current_token][next_token] += 1

    def distribution(self, context: list[str]) -> dict[str, float]:
        """返回给定上下文后的 Bigram 概率分布。"""

        # 取最后一个 token，若无则使用起始标记
        previous_token = context[-1] if context else START_TOKEN
        next_counts = self.counts[previous_token]
        vocabulary_size = len(self.vocabulary)
        
        # 计算分母：总频次 + 平滑参数 * 词汇表大小
        denominator = sum(next_counts.values())
        denominator += self.smoothing * vocabulary_size

        # 平滑保证语料中未出现的转移仍有一个很小的概率。
        return {
            token: (next_counts[token] + self.smoothing) / denominator
            for token in self.vocabulary
        }


class KnnMemory:
    """保存训练上下文，并在预测时执行最近邻检索。
    
    在构建阶段，将所有上下文（固定窗口大小）及其后续 token 存储为 MemoryEntry。
    预测时，根据查询上下文与所有记忆项的余弦相似度，选出 k 个最近邻，
    并基于相似度加权（温度缩放）得到 token 分布。
    """

    def __init__(self, context_size: int = 4) -> None:
        
        #初始化 KNN 记忆库。
        self.context_size = context_size
        self.entries: list[MemoryEntry] = []

    def build(self, sentences: Iterable[str]) -> None:
        """从语料构建上下文到下一个 token 的记忆库。"""

        for sentence in sentences:
            tokens = [START_TOKEN, *tokenize(sentence), END_TOKEN]

            for index in range(1, len(tokens)):
                # 只保留当前位置之前的固定窗口，模拟隐藏状态的局部信息。
                start = max(0, index - self.context_size)
                context = tuple(tokens[start:index])
                self.entries.append(
                    MemoryEntry(
                        key=context_vector(context),
                        value=tokens[index],
                        source_context=context,
                    )
                )

    def distribution(
        self,
        context: list[str],
        k_neighbors: int = 5,
        temperature: float = 0.2,
    ) -> tuple[dict[str, float], list[tuple[float, MemoryEntry]]]:
        """返回 KNN token 分布以及选中的最近邻。"""

        # 查询窗口必须与构建记忆库时的窗口定义保持一致。
        query_tokens = context[-self.context_size:]
        query_vector = context_vector(query_tokens)
        
        # 计算与所有记忆项的余弦相似度
        scored_entries = [
            (cosine_similarity(query_vector, entry.key), entry)
            for entry in self.entries
        ]
        # 按相似度降序排序
        scored_entries.sort(key=lambda item: item[0], reverse=True)
        neighbors = scored_entries[:k_neighbors]

        # 根据相似度加权计数（temperature 缩放）
        weighted_counts: Counter[str] = Counter()
        for similarity, entry in neighbors:
            # 温度越低，最相似邻居在概率中占据的权重越高。
            weight = math.exp(similarity / temperature)
            weighted_counts[entry.value] += weight

        # 归一化为概率
        total_weight = sum(weighted_counts.values())
        probabilities = {
            token: weight / total_weight
            for token, weight in weighted_counts.items()
        }
        return probabilities, neighbors


def interpolate_distributions(
    base_distribution: dict[str, float],
    knn_distribution: dict[str, float],
    memory_weight: float,
) -> dict[str, float]:
    """线性插值基础模型概率与 KNN 概率。"""

    if not 0.0 <= memory_weight <= 1.0:
        raise ValueError("memory_weight 必须位于 [0, 1]。")

    # 使用并集，避免丢失只出现在某一个分布中的 token。
    tokens = set(base_distribution) | set(knn_distribution)
    return {
        token: (1.0 - memory_weight) * base_distribution.get(token, 0.0)
        + memory_weight * knn_distribution.get(token, 0.0)
        for token in tokens
    }


def print_top_predictions(
    title: str,
    distribution: dict[str, float],
    top_k: int = 5,
) -> None:
    """打印概率最高的若干 token。"""

    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    print(f"\n{title}")
    for token, probability in ranked[:top_k]:
        print(f"  {token:<12} {probability:.4f}")


def build_demo_corpus() -> list[str]:
    """返回用于演示领域检索的小型语料。"""

    return [
        "machine learning models learn patterns from data",
        "machine learning systems need clean training data",
        "deep learning models use neural networks",
        "large language models predict the next token",
        "language models learn statistical patterns from text",
        "pytorch helps students build neural networks",
        "pytorch models use tensors and automatic gradients",
        "clean data helps models generalize to new examples",
        "nearest neighbor search retrieves similar contexts",
        "a memory datastore can provide rare factual tokens",
    ]


def main() -> None:
    """构建模型，并比较插值前后的预测结果。"""

    corpus = build_demo_corpus()
    base_model = BigramLanguageModel()
    memory = KnnMemory(context_size=4)

    # 主语言模型和外部记忆使用同一语料，但承担不同职责。
    base_model.fit(corpus)
    memory.build(corpus)

    # 定义查询上下文（模拟正在生成的序列）
    query = tokenize("machine learning systems need")
    # 获取基础模型预测
    base_probabilities = base_model.distribution(query)
    # 获取 KNN 预测和最近邻信息
    knn_probabilities, neighbors = memory.distribution(query)
    # 插值得到最终概率
    final_probabilities = interpolate_distributions(
        base_probabilities,
        knn_probabilities,
        memory_weight=0.65,  # 记忆库权重 65%
    )

    # 打印查询和检索到的最近邻上下文
    print("query:", " ".join(query))
    print("\nnearest contexts:")
    for similarity, entry in neighbors:
        context = " ".join(entry.source_context)
        print(f"  similarity={similarity:.3f} | {context} -> {entry.value}")

    # 对比三组结果，可以直观看到外部记忆如何修正预测。
    print_top_predictions("base language model", base_probabilities)
    print_top_predictions("KNN memory", knn_probabilities)
    print_top_predictions("interpolated result", final_probabilities)


if __name__ == "__main__":
    main()
