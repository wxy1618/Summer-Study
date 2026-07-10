# -*- coding: utf-8 -*-
"""A small reproduction of fastbook's mid-level data idea.

本脚本对应 fastbook Chapter 11。它不调用 fastai，而是用标准库复现
Transform、Pipeline、setup、decode 这些中层数据 API 的核心思想。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol


TOKEN_RE = re.compile(r"[a-zA-Z]+|[0-9]+")

# 本脚本不引入 fastai，而是用最小实现复现中层数据 API 的思想。
# 这样可以把注意力放在“数据变换对象如何组合”这个抽象本身。


class TransformProtocol(Protocol):
    """Minimal protocol expected by the pipeline below."""

    # Protocol 用于描述结构化接口：只要对象实现这些方法，
    # 就能被 Pipeline 使用，不强制继承某个基类。
    def setup(self, items: Iterable[Any]) -> None:
        ...

    def __call__(self, item: Any) -> Any:
        # __call__ 表示正向变换，例如文本 -> token 或 token -> id。
        ...

    def decode(self, item: Any) -> Any:
        # decode 是中层 API 的关键：它让张量化后的结果能回到可读形式。
        ...


class Tokenize:
    """Split raw text into lowercase tokens."""

    def setup(self, items: Iterable[str]) -> None:
        # Tokenization has no trainable state; setup keeps the API uniform.
        # 统一保留 setup 接口，可以让无状态和有状态 transform 被同样调度。
        _ = list(items)

    def __call__(self, item: str) -> list[str]:
        # 这里的 tokenization 是确定性函数，便于和 decode 结果逐项核对。
        return TOKEN_RE.findall(item.lower())

    def decode(self, item: list[str]) -> str:
        # 简单 join 不是完全可逆分词，但足够展示“人类可读化”的调试价值。
        return " ".join(item)


class Numericalize:
    """Map tokens to ids and provide a reversible decode method."""

    def __init__(self, min_freq: int = 1) -> None:
        # min_freq 是词表裁剪阈值，真实项目中可用于抑制低频噪声。
        self.min_freq = min_freq
        self.id_to_token = ["<pad>", "<unk>"]
        self.token_to_id = {"<pad>": 0, "<unk>": 1}

    def setup(self, items: Iterable[list[str]]) -> None:
        # Numericalize 是有状态变换：它必须先从训练数据中学习词表。
        counter: Counter[str] = Counter()
        for tokens in items:
            counter.update(tokens)

        # 词表排序固定后，数值化结果才具有可复现性。
        words = [
            word
            for word, count in sorted(
                counter.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if count >= self.min_freq
        ]
        self.id_to_token = ["<pad>", "<unk>"] + words
        # token_to_id 和 id_to_token 成对保存，分别服务编码和解码。
        self.token_to_id = {
            token: idx
            for idx, token in enumerate(self.id_to_token)
        }

    def __call__(self, item: list[str]) -> list[int]:
        # 未登录词统一映射为 <unk>，避免推理阶段因新词直接失败。
        unk_id = self.token_to_id["<unk>"]
        return [self.token_to_id.get(token, unk_id) for token in item]

    def decode(self, item: list[int]) -> list[str]:
        # decode 可用于检查 padding、未知词和词表索引是否符合预期。
        return [self.id_to_token[token_id] for token_id in item]


class LabelEncode:
    """Encode string labels into integers."""

    def __init__(self) -> None:
        # 标签编码同样需要保存可逆映射，便于报告中还原类别名称。
        self.classes: list[str] = []
        self.class_to_id: dict[str, int] = {}

    def setup(self, items: Iterable[str]) -> None:
        # 类别排序固定，避免不同运行中 label id 改变导致结果解释错位。
        self.classes = sorted(set(items))
        self.class_to_id = {label: idx for idx, label in enumerate(self.classes)}

    def __call__(self, item: str) -> int:
        # 正向变换把监督标签从字符串变成损失函数可接受的整数类别。
        return self.class_to_id[item]

    def decode(self, item: int) -> str:
        # 预测结果解码回字符串后，才方便和原始标注做人工对比。
        return self.classes[item]


class Pipeline:
    """Apply a list of transforms sequentially."""

    def __init__(self, transforms: list[TransformProtocol]) -> None:
        # Pipeline 明确保存变换顺序，因为文本处理通常不可交换。
        self.transforms = transforms

    def setup(self, items: Iterable[Any]) -> None:
        # setup 必须按流水线顺序执行：后一个 transform 看到的是前一个的输出。
        current_items = list(items)
        for transform in self.transforms:
            transform.setup(current_items)
            # 这里 materialize 中间结果，是为了让后续有状态变换能统计全量数据。
            current_items = [transform(item) for item in current_items]

    def __call__(self, item: Any) -> Any:
        # 推理时复用 setup 阶段学到的状态，例如词表和标签表。
        value = item
        for transform in self.transforms:
            value = transform(value)
        return value

    def decode(self, item: Any) -> Any:
        # 解码必须反向穿过 pipeline，这和神经网络中的反向映射思路相似。
        value = item
        for transform in reversed(self.transforms):
            value = transform.decode(value)
        return value


@dataclass
class TextRow:
    """A tiny row object close to what a real project dataset may contain."""

    # dataclass 让样本字段显式命名，比裸 tuple 更接近真实数据工程实践。
    text: str
    label: str


class MiniDatasets:
    """Store raw items and independent x/y pipelines."""

    def __init__(
        self,
        rows: list[TextRow],
        x_pipeline: Pipeline,
        y_transform: LabelEncode,
    ) -> None:
        # x_pipeline 处理输入文本，y_transform 处理监督标签；
        # 拆开两条路径能表达输入和目标的不同变换规则。
        self.rows = rows
        self.x_pipeline = x_pipeline
        self.y_transform = y_transform

    def setup(self) -> None:
        # setup 只运行一次，模拟 fastai 在训练前拟合所有必要预处理状态。
        self.x_pipeline.setup(row.text for row in self.rows)
        self.y_transform.setup(row.label for row in self.rows)

    def __len__(self) -> int:
        # 数据集长度用于迭代和后续构造 DataLoader。
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        # 每次索引访问时才执行 transform，保留了惰性处理的思想。
        row = self.rows[index]
        return self.x_pipeline(row.text), self.y_transform(row.label)

    def decode(self, item: tuple[list[int], int]) -> tuple[str, str]:
        # 同时解码输入和标签，方便 show_batch / show_results 类调试输出。
        x_ids, y_id = item
        return self.x_pipeline.decode(x_ids), self.y_transform.decode(y_id)


def demo_rows() -> list[TextRow]:
    # 样例覆盖 preprocess、model、evaluation 三类，展示标签编码的完整流程。
    return [
        TextRow("Tokenization prepares text for numerical models", "preprocess"),
        TextRow("Numericalization converts tokens into integer ids", "preprocess"),
        TextRow("A recurrent model predicts the next token", "model"),
        TextRow("A transformer model uses attention over tokens", "model"),
        TextRow("Validation metrics record whether training generalizes", "evaluation"),
        TextRow("Error analysis explains where the model fails", "evaluation"),
    ]


def run_demo() -> dict[str, Any]:
    # 构造原始行后，分别定义输入流水线和标签编码器。
    rows = demo_rows()
    x_pipeline = Pipeline([Tokenize(), Numericalize(min_freq=1)])
    y_transform = LabelEncode()
    datasets = MiniDatasets(rows, x_pipeline, y_transform)
    # setup 是中层 API 和普通函数式预处理的关键差异：它学习并固定状态。
    datasets.setup()

    # 编码和解码样例共同输出，可以验证 transform 是否按预期工作。
    encoded_items = [datasets[index] for index in range(len(datasets))]
    decoded_items = [datasets.decode(item) for item in encoded_items]
    # 这里显式取出 Numericalize，是为了把词表写入实验记录。
    numericalize = x_pipeline.transforms[1]

    # 结果 JSON 不只保存最终输出，也保存词表和类别表，方便复现实验。
    result = {
        "task": "midlevel_data_api_demo",
        "num_rows": len(datasets),
        "vocab": numericalize.id_to_token,
        "classes": y_transform.classes,
        "encoded_samples": encoded_items[:3],
        "decoded_samples": decoded_items[:3],
        "observation": (
            "setup 负责从训练数据学习词表和标签表；decode 让中间张量重新变回"
            "可检查的人类可读形式。"
        ),
    }
    return result


def main() -> None:
    # 标准库脚本无需依赖检查，直接运行演示并输出 JSON 记录。
    result = run_demo()
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    # 保存文件使 notebook 外的命令行实验也能留下可追溯结果。
    (output_dir / "midlevel_data_api_demo.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("中层数据 API 演示完成，结果已写入 outputs/midlevel_data_api_demo.json")


if __name__ == "__main__":
    main()
