# -*- coding: utf-8 -*-
"""Toy TransE knowledge graph embedding with hyperparameter tuning.

本脚本对应第六周 TransE 词嵌入实战。它用小型知识图谱展示
``head + relation ≈ tail`` 的平移思想，并记录 Hits@1、MRR 和调参结果。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path

try:
    # PyTorch 只用于嵌入训练；缺失时脚本按课程要求温和退出。
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover
    torch = None

    class _MissingNN:
        Module = object

    nn = _MissingNN()
    F = None


@dataclass(frozen=True)
class Triple:
    """A knowledge graph triple."""

    # head、relation、tail 都使用字符串保存，便于输出时直接阅读。
    head: str
    relation: str
    tail: str


# 小型知识图谱覆盖 person、field、place、institution 等关系模式。
TRIPLES = [
    Triple("Marie_Curie", "discovered", "radium"),
    Triple("Marie_Curie", "worked_in", "Paris"),
    Triple("Alan_Turing", "designed", "Turing_machine"),
    Triple("Alan_Turing", "worked_in", "Cambridge"),
    Triple("Ada_Lovelace", "wrote_about", "analytical_engine"),
    Triple("Transformer", "uses", "attention"),
    Triple("GloVe", "uses", "cooccurrence"),
    Triple("TransE", "models", "knowledge_graph"),
]


def require_torch() -> None:
    """Stop without modifying the environment when PyTorch is unavailable."""

    # 不执行安装命令，确保脚本不会改变用户已有课程环境。
    if torch is None:
        print("当前环境未安装 PyTorch；本脚本不会自动安装依赖。")
        raise SystemExit(0)


def build_mappings(triples: list[Triple]) -> tuple[dict[str, int], dict[str, int]]:
    """Build entity and relation id mappings."""

    # TransE 需要分别为实体和关系维护嵌入表，因此二者词表必须分开。
    entities = sorted(
        {triple.head for triple in triples}
        | {triple.tail for triple in triples}
    )
    relations = sorted({triple.relation for triple in triples})
    entity_to_id = {entity: index for index, entity in enumerate(entities)}
    relation_to_id = {relation: index for index, relation in enumerate(relations)}
    return entity_to_id, relation_to_id


def encode_triples(
    triples: list[Triple],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> list[tuple[int, int, int]]:
    """Encode string triples as integer triples."""

    # 整数编码让三元组可以直接索引 embedding 表。
    return [
        (entity_to_id[t.head], relation_to_id[t.relation], entity_to_id[t.tail])
        for t in triples
    ]


class TransEModel(nn.Module):
    """TransE scoring model using L2 distance."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        # entity_embeddings 和 relation_embeddings 分别学习节点与边的向量表示。
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Xavier 初始化使不同维度的初始方差更稳定，适合小型嵌入模型。
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def score(self, triples: "torch.Tensor") -> "torch.Tensor":
        """Return distance scores; lower means more plausible."""

        # TransE 的核心假设是 h + r 应接近 t，因此使用向量距离作为打分。
        heads = self.entity_embeddings(triples[:, 0])
        relations = self.relation_embeddings(triples[:, 1])
        tails = self.entity_embeddings(triples[:, 2])
        return torch.linalg.vector_norm(heads + relations - tails, dim=1)

    def normalize_entities(self) -> None:
        """Project entity embeddings to the unit ball."""

        # TransE 论文中常对实体向量做归一化，避免通过无限放大范数降低损失。
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(
                self.entity_embeddings.weight.data,
                p=2,
                dim=1,
            )


def corrupt_tail(
    triples: list[tuple[int, int, int]],
    num_entities: int,
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    """Create negative triples by replacing the tail entity."""

    # 负采样让模型学习把真实三元组和破坏三元组分开。
    negatives = []
    positive_set = set(triples)
    for head, relation, tail in triples:
        new_tail = tail
        while new_tail == tail or (head, relation, new_tail) in positive_set:
            new_tail = rng.randrange(num_entities)
        negatives.append((head, relation, new_tail))
    return negatives


def train_once(
    encoded: list[tuple[int, int, int]],
    num_entities: int,
    num_relations: int,
    args: argparse.Namespace,
    config: dict[str, float | int],
) -> tuple[TransEModel, list[dict[str, float | int]]]:
    """Train one TransE configuration."""

    # 每组超参数使用同一随机种子，减少调参比较中的随机性。
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    model = TransEModel(num_entities, num_relations, int(config["embedding_dim"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    margin = float(config["margin"])
    history: list[dict[str, float | int]] = []

    # 小数据集使用全量 batch，避免 batch 随机性掩盖超参数影响。
    positive = torch.tensor(encoded, dtype=torch.long)
    for epoch in range(1, args.epochs + 1):
        negative = torch.tensor(
            corrupt_tail(encoded, num_entities, rng),
            dtype=torch.long,
        )
        positive_score = model.score(positive)
        negative_score = model.score(negative)

        # margin ranking loss 要求正样本距离至少比负样本小 margin。
        loss = F.relu(margin + positive_score - negative_score).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.normalize_entities()

        # 记录少量 epoch 指标，避免 JSON 过大，同时保留收敛趋势。
        if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 4, 1) == 0:
            history.append({"epoch": epoch, "loss": round(float(loss.item()), 4)})
    return model, history


def rank_tails(
    model: TransEModel,
    triples: list[tuple[int, int, int]],
    num_entities: int,
) -> tuple[float, float]:
    """Compute Hits@1 and MRR for tail prediction."""

    # Tail prediction 评估给定 h,r 时正确 t 的排名，适合知识图谱嵌入实验。
    ranks = []
    with torch.no_grad():
        for head, relation, tail in triples:
            candidates = torch.tensor(
                [(head, relation, entity) for entity in range(num_entities)],
                dtype=torch.long,
            )
            scores = model.score(candidates)
            order = torch.argsort(scores).tolist()
            ranks.append(order.index(tail) + 1)

    # Hits@1 看第一名是否正确，MRR 更细致地考虑正确答案排名。
    hits_at_1 = sum(rank == 1 for rank in ranks) / len(ranks)
    mrr = sum(1 / rank for rank in ranks) / len(ranks)
    return hits_at_1, mrr


def run_tuning(args: argparse.Namespace) -> dict[str, object]:
    """Run a compact TransE hyperparameter search."""

    # 构建整数化知识图谱，是所有配置共享的数据基础。
    entity_to_id, relation_to_id = build_mappings(TRIPLES)
    encoded = encode_triples(TRIPLES, entity_to_id, relation_to_id)

    # 调参维度覆盖嵌入容量、间隔约束和优化步长三类关键因素。
    search_space = {
        "embedding_dim": [16, 24],
        "margin": [0.5, 1.0],
        "learning_rate": [0.02, 0.05],
    }
    trials = []

    # 每个 trial 都训练模型并记录最终排名指标。
    for values in product(*search_space.values()):
        config = dict(zip(search_space.keys(), values))
        model, history = train_once(
            encoded,
            len(entity_to_id),
            len(relation_to_id),
            args,
            config,
        )
        hits_at_1, mrr = rank_tails(model, encoded, len(entity_to_id))
        trials.append(
            {
                "config": config,
                "hits_at_1": round(hits_at_1, 4),
                "mrr": round(mrr, 4),
                "history": history,
            }
        )

    # 先看 MRR，再看 Hits@1，避免只优化极少数完全正确样本。
    best = max(trials, key=lambda row: (row["mrr"], row["hits_at_1"]))
    return {
        "task": "transe_embedding_tuning",
        "num_entities": len(entity_to_id),
        "num_relations": len(relation_to_id),
        "num_triples": len(encoded),
        "best_result": best,
        "trials": trials,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    # epochs 和 seed 暴露给命令行，便于控制训练预算和复现结果。
    parser = argparse.ArgumentParser(description="Toy TransE tuning experiment")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def main() -> None:
    """Run TransE tuning and save JSON results."""

    # 依赖检查放在入口处，使文件可以被静态阅读和 py_compile。
    require_torch()
    result = run_tuning(parse_args())
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "transe_tuning_results.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
