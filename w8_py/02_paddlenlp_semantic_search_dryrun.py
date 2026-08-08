"""Dry-run semantic search workflow inspired by PaddleNLP neural search.

The default path uses a transparent TF-IDF baseline so the assignment can be
executed without configuring PaddleNLP.  The script records the same objects a
real semantic retrieval experiment should record: corpus, queries, ranking,
Recall@K, MRR, and error cases.
"""

# 标准库实现可以稳定运行，并把检索流程拆开给人工审查。
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path


# 输出目录固定在本周作业目录下，避免和其他周实验混杂。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class SearchDocument:
    """A searchable document in the retrieval corpus."""

    # doc_id 是评价指标判断召回是否正确的依据。
    doc_id: str

    # title 让检索结果更容易被人工阅读。
    title: str

    # text 是参与向量化和排序的主要内容。
    text: str


@dataclass(frozen=True)
class SearchQuery:
    """A query with known relevant documents."""

    # query_id 用于区分多条检索问题。
    query_id: str

    # text 是用户检索输入。
    text: str

    # relevant_doc_ids 是小规模人工标注答案集合。
    relevant_doc_ids: list[str]


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked retrieval result."""

    # rank 从 1 开始，便于计算 MRR 和人工查看。
    rank: int

    # doc_id 与语料文档对应。
    doc_id: str

    # title 用于快速判断命中主题。
    title: str

    # score 是 TF-IDF 余弦相似度。
    score: float


def build_corpus() -> list[SearchDocument]:
    """Create a small PaddleNLP-oriented retrieval corpus."""

    # 文档围绕第八周三个实战任务：语义检索、文档问答和情感分析。
    return [
        SearchDocument(
            doc_id="doc_semantic_search",
            title="Semantic Search",
            text=(
                "Semantic search encodes queries and documents into vectors. "
                "The system retrieves relevant documents by vector similarity "
                "and evaluates ranking quality with Recall@K and MRR."
            ),
        ),
        SearchDocument(
            doc_id="doc_doc_vqa",
            title="Document VQA",
            text=(
                "Document visual question answering combines OCR tokens, page "
                "layout, question understanding and answer extraction from "
                "document evidence."
            ),
        ),
        SearchDocument(
            doc_id="doc_sentiment",
            title="Sentiment Analysis",
            text=(
                "Sentiment analysis fine-tunes a pretrained model to classify "
                "text polarity and records accuracy, precision, recall, F1 and "
                "error examples."
            ),
        ),
        SearchDocument(
            doc_id="doc_ethics",
            title="Ethics",
            text=(
                "Responsible NLP systems should audit bias, privacy, safety, "
                "copyright, data provenance and rollback plans before release."
            ),
        ),
        SearchDocument(
            doc_id="doc_explainability",
            title="Explainability",
            text=(
                "Model explanations use attribution, deletion tests and "
                "counterfactual examples to analyze why predictions change."
            ),
        ),
    ]


def build_queries() -> list[SearchQuery]:
    """Create query examples with relevance labels."""

    # relevance label 是检索实验的监督信号，没有它就只能做主观展示。
    return [
        SearchQuery(
            query_id="q1",
            text="How to evaluate a semantic retrieval ranking?",
            relevant_doc_ids=["doc_semantic_search"],
        ),
        SearchQuery(
            query_id="q2",
            text="Which task uses OCR tokens and page layout?",
            relevant_doc_ids=["doc_doc_vqa"],
        ),
        SearchQuery(
            query_id="q3",
            text="What metrics are recorded for polarity classification?",
            relevant_doc_ids=["doc_sentiment"],
        ),
        SearchQuery(
            query_id="q4",
            text="How can an NLP model be checked for privacy and bias?",
            relevant_doc_ids=["doc_ethics"],
        ),
    ]


def tokenize(text: str) -> list[str]:
    """Tokenize text for the lexical dry-run baseline."""

    # 这里使用英文 token，是因为内置样例采用英文技术短句。
    return re.findall(r"[a-zA-Z][a-zA-Z0-9@_-]*", text.lower())


def compute_idf(corpus: list[SearchDocument]) -> dict[str, float]:
    """Compute smoothed inverse document frequency."""

    # IDF 提高领域关键词权重，例如 OCR、Recall@K、polarity 等。
    document_frequency: dict[str, int] = {}
    for document in corpus:
        for token in set(tokenize(document.title + " " + document.text)):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    # 平滑 IDF 保证所有出现过的词都有稳定权重。
    total = len(corpus)
    return {
        token: math.log((1 + total) / (1 + frequency)) + 1
        for token, frequency in document_frequency.items()
    }


def vectorize(text: str, idf: dict[str, float]) -> dict[str, float]:
    """Build a sparse TF-IDF vector."""

    # 词频体现局部重要性，IDF 体现全局区分度。
    counts: dict[str, int] = {}
    for token in tokenize(text):
        counts[token] = counts.get(token, 0) + 1

    # 对查询中未在语料出现的词给默认权重，避免完全丢弃。
    return {token: count * idf.get(token, 1.0) for token, count in counts.items()}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    """Compute cosine similarity between sparse vectors."""

    # 余弦相似度适合比较短查询和文档向量方向的一致性。
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    # 空向量代表没有可比较词项，直接返回 0。
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def retrieve(
    query: SearchQuery,
    corpus: list[SearchDocument],
    top_k: int,
) -> list[RetrievalResult]:
    """Retrieve top-k documents for one query."""

    # 每次检索都重建小型索引，课堂样例足够；真实系统应缓存向量。
    idf = compute_idf(corpus)
    query_vector = vectorize(query.text, idf)

    # 对所有文档打分并排序，形成可解释的检索列表。
    results: list[RetrievalResult] = []
    for document in corpus:
        doc_vector = vectorize(document.title + " " + document.text, idf)
        score = cosine(query_vector, doc_vector)
        results.append(
            RetrievalResult(
                rank=0,
                doc_id=document.doc_id,
                title=document.title,
                score=round(score, 6),
            )
        )

    # 排名号在排序后再赋值，确保 rank 与得分顺序一致。
    sorted_results = sorted(results, key=lambda item: item.score, reverse=True)
    return [
        RetrievalResult(
            rank=index,
            doc_id=item.doc_id,
            title=item.title,
            score=item.score,
        )
        for index, item in enumerate(sorted_results[:top_k], start=1)
    ]


def compute_metrics(
    queries: list[SearchQuery],
    rankings: dict[str, list[RetrievalResult]],
    top_k: int,
) -> dict[str, float]:
    """Compute Recall@K and MRR over all queries."""

    # Recall@K 检查 top-k 中是否至少包含一个相关文档。
    recall_hits = 0
    reciprocal_ranks: list[float] = []
    for query in queries:
        relevant = set(query.relevant_doc_ids)
        ranked = rankings[query.query_id]
        first_hit_rank = None
        for item in ranked:
            if item.doc_id in relevant:
                first_hit_rank = item.rank
                break
        if first_hit_rank is not None:
            recall_hits += 1
            reciprocal_ranks.append(1 / first_hit_rank)
        else:
            reciprocal_ranks.append(0.0)

    # MRR 对正确结果排位更敏感，能区分第一名命中和第三名命中。
    query_count = len(queries)
    return {
        f"recall_at_{top_k}": round(recall_hits / query_count, 4),
        "mrr": round(sum(reciprocal_ranks) / query_count, 4),
    }


def check_paddlenlp_availability() -> dict[str, object]:
    """Check whether PaddleNLP is importable without installing anything."""

    # 该函数只检查环境状态，不下载模型，也不修改解释器环境。
    status = {"paddle_available": False, "paddlenlp_available": False}
    try:
        import paddle  # type: ignore

        status["paddle_available"] = True
        status["paddle_version"] = getattr(paddle, "__version__", "unknown")
    except ImportError as error:
        status["paddle_error"] = str(error)

    # PaddleNLP 与 PaddlePaddle 分开检查，便于定位缺失依赖。
    try:
        import paddlenlp  # type: ignore

        status["paddlenlp_available"] = True
        status["paddlenlp_version"] = getattr(paddlenlp, "__version__", "unknown")
    except ImportError as error:
        status["paddlenlp_error"] = str(error)
    return status


def run_experiment(top_k: int) -> dict[str, object]:
    """Run the semantic search dry-run experiment."""

    # 构造小型语料和查询集，模拟 PaddleNLP neural search 的核心对象。
    corpus = build_corpus()
    queries = build_queries()
    rankings = {
        query.query_id: retrieve(query, corpus, top_k=top_k) for query in queries
    }

    # 指标与逐条排名同时保存，便于后续分析检索错误。
    metrics = compute_metrics(queries, rankings, top_k=top_k)
    records = []
    for query in queries:
        ranked = rankings[query.query_id]
        records.append(
            {
                "query": asdict(query),
                "ranking": [asdict(item) for item in ranked],
                "hit": any(
                    item.doc_id in set(query.relevant_doc_ids) for item in ranked
                ),
            }
        )
    return {
        "mode": "standard_library_tfidf_dryrun",
        "metrics": metrics,
        "records": records,
        "paddlenlp_environment": check_paddlenlp_availability(),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    # top_k 是检索系统最基础的评价控制变量。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Execute and save the semantic search dry-run."""

    # dry-run 产物可在没有 PaddleNLP 环境时直接用于周报记录。
    args = parse_args()
    result = run_experiment(top_k=args.top_k)

    # JSON 保存完整排名和指标，Markdown 报告由汇总脚本再生成。
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "week8_semantic_search_dryrun.json"
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 打印核心指标，让命令行输出可以直接检查实验是否合理。
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"Artifact written to: {output_path}")


# 主入口避免导入时执行实验，方便其他脚本复用函数。
if __name__ == "__main__":
    main()
