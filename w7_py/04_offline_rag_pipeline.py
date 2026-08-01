"""A fully offline retrieval-augmented generation prototype.

The implementation intentionally avoids API calls.  It demonstrates the RAG
pipeline with sentence chunking, TF-IDF retrieval, evidence prompt assembly,
and an extractive answer baseline.
"""

# 标准库实现便于观察每个环节，而不是把细节隐藏在框架内部。
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


# 实验输出放在本周目录，形成和其他脚本一致的结果结构。
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass(frozen=True)
class Document:
    """A source document before chunking."""

    # doc_id 是证据引用的最小来源单位。
    doc_id: str

    # title 便于在检索结果中快速判断主题。
    title: str

    # text 保存文档正文，本实验使用内置课程资料。
    text: str


@dataclass(frozen=True)
class Chunk:
    """A searchable chunk derived from a document."""

    # chunk_id 由文档编号和窗口编号构成，用于定位证据片段。
    chunk_id: str

    # doc_id 让 chunk 可以追溯回原始文档。
    doc_id: str

    # title 保存来源标题，便于生成带引用的回答。
    title: str

    # text 是参与检索和 prompt 拼接的实际文本。
    text: str


@dataclass(frozen=True)
class RetrievalHit:
    """A retrieved chunk with its similarity score."""

    # chunk 保存被召回的证据。
    chunk: Chunk

    # score 是 TF-IDF 余弦相似度，用于排序和错误分析。
    score: float


def build_documents() -> list[Document]:
    """Build a tiny course knowledge base."""

    # 文档覆盖第七周核心主题：T5、指令微调、PEFT 和 RAG。
    return [
        Document(
            doc_id="doc_t5",
            title="T5 text-to-text",
            text=(
                "T5 represents every NLP task as text-to-text learning. "
                "Classification labels, summaries, translations and answers "
                "are all written as target text. Task prefixes tell the model "
                "which behavior is expected."
            ),
        ),
        Document(
            doc_id="doc_instruction",
            title="Instruction tuning",
            text=(
                "Instruction tuning uses supervised instruction-input-output "
                "examples. The purpose is to make a pretrained language model "
                "follow user requests, respect formats and answer in a more "
                "helpful way."
            ),
        ),
        Document(
            doc_id="doc_lora",
            title="LoRA",
            text=(
                "LoRA freezes base model weights and trains low-rank adapter "
                "matrices. This reduces trainable parameters and storage cost "
                "while keeping most pretrained knowledge unchanged."
            ),
        ),
        Document(
            doc_id="doc_rag",
            title="Retrieval augmented generation",
            text=(
                "RAG retrieves external documents before generation. It is "
                "useful when knowledge changes frequently or answers need "
                "source evidence. The model receives retrieved chunks in the "
                "prompt and should answer based on those chunks."
            ),
        ),
    ]


def split_sentences(text: str) -> list[str]:
    """Split text into coarse sentences."""

    # 课程材料通常句子边界清楚，正则切分足够支持本实验。
    pieces = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def chunk_documents(documents: list[Document], window_size: int) -> list[Chunk]:
    """Create sentence-window chunks from documents."""

    # 窗口切分模拟真实 RAG 中 chunk_size 的选择问题。
    chunks: list[Chunk] = []
    for document in documents:
        sentences = split_sentences(document.text)
        for start in range(0, len(sentences), window_size):
            window = sentences[start:start + window_size]
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}_{start // window_size}",
                    doc_id=document.doc_id,
                    title=document.title,
                    text=" ".join(window),
                )
            )
    return chunks


def tokenize(text: str) -> list[str]:
    """Normalize and tokenize text for lexical retrieval."""

    # TF-IDF 检索需要稳定 token，本实验用英文小写词作为特征。
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())


def compute_idf(chunks: list[Chunk]) -> dict[str, float]:
    """Compute inverse document frequency over chunks."""

    # IDF 降低常见词权重，提高领域关键词的检索影响。
    document_frequency: dict[str, int] = {}
    for chunk in chunks:
        for token in set(tokenize(chunk.text)):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    # 平滑处理避免除零，同时使低频词得分更高。
    total = len(chunks)
    return {
        token: math.log((1 + total) / (1 + frequency)) + 1
        for token, frequency in document_frequency.items()
    }


def vectorize(text: str, idf: dict[str, float]) -> dict[str, float]:
    """Build a TF-IDF vector represented as a sparse dictionary."""

    # 词频统计保留局部重要性，之后乘以全局 IDF。
    counts: dict[str, int] = {}
    for token in tokenize(text):
        counts[token] = counts.get(token, 0) + 1

    # 未见过的查询词没有 IDF，默认权重为 1。
    return {token: count * idf.get(token, 1.0) for token, count in counts.items()}


def cosine_similarity(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    """Compute cosine similarity for sparse vectors."""

    # 余弦相似度关注方向相似，适合短查询与文档片段匹配。
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))

    # 任一向量为空时没有可比较特征，返回 0。
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def retrieve(query: str, chunks: list[Chunk], top_k: int) -> list[RetrievalHit]:
    """Retrieve top-k chunks for a user query."""

    # 检索阶段先构建索引；小语料即时计算即可，大语料应缓存。
    idf = compute_idf(chunks)
    query_vector = vectorize(query, idf)

    # 每个 chunk 都转为稀疏向量，并与查询计算相似度。
    hits: list[RetrievalHit] = []
    for chunk in chunks:
        chunk_vector = vectorize(chunk.text, idf)
        score = cosine_similarity(query_vector, chunk_vector)
        hits.append(RetrievalHit(chunk=chunk, score=round(score, 6)))

    # 按相似度降序排序，取前 top_k 作为 prompt 证据。
    hits.sort(key=lambda item: item.score, reverse=True)
    return hits[:top_k]


def build_prompt(query: str, hits: list[RetrievalHit]) -> str:
    """Assemble an evidence-grounded prompt."""

    # prompt 明确要求基于证据回答，减少无依据扩展。
    evidence_blocks = []
    for index, hit in enumerate(hits, start=1):
        evidence_blocks.append(
            f"[{index}] {hit.chunk.title} "
            f"({hit.chunk.chunk_id}, score={hit.score}): {hit.chunk.text}"
        )
    evidence = "\n".join(evidence_blocks)
    return (
        "请仅根据给定证据回答问题；若证据不足，请说明无法确定。\n\n"
        f"问题：{query}\n\n证据：\n{evidence}\n\n回答："
    )


def extractive_answer(query: str, hits: list[RetrievalHit]) -> str:
    """Generate a transparent extractive answer from retrieved evidence."""

    # 离线脚本没有调用生成模型，因此用最高分证据句作为保守回答。
    if not hits or hits[0].score <= 0:
        return "根据当前知识库无法确定答案。"

    # 查询关键词用于从最高分 chunk 中选择更贴近问题的句子。
    query_tokens = set(tokenize(query))
    best_sentence = ""
    best_overlap = -1
    for sentence in split_sentences(hits[0].chunk.text):
        overlap = len(query_tokens.intersection(tokenize(sentence)))
        if overlap > best_overlap:
            best_sentence = sentence
            best_overlap = overlap

    # 输出保留来源，体现 RAG 相比纯生成的可追溯优势。
    return f"{best_sentence} 来源：{hits[0].chunk.title}。"


def run_queries(queries: list[str], top_k: int) -> list[dict[str, object]]:
    """Run the full offline RAG loop for all queries."""

    # 构建文档、切分、检索、prompt 拼接和答案生成构成完整 RAG 链路。
    documents = build_documents()
    chunks = chunk_documents(documents, window_size=2)

    # 每个问题都保存检索结果和 prompt，便于诊断错误来自召回还是生成。
    records: list[dict[str, object]] = []
    for query in queries:
        hits = retrieve(query, chunks, top_k=top_k)
        records.append(
            {
                "query": query,
                "prompt": build_prompt(query, hits),
                "answer": extractive_answer(query, hits),
                "hits": [
                    {
                        "chunk_id": hit.chunk.chunk_id,
                        "title": hit.chunk.title,
                        "score": hit.score,
                        "text": hit.chunk.text,
                    }
                    for hit in hits
                ],
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    """Parse retrieval experiment options."""

    # top_k 暴露为参数，便于观察召回证据数量对回答的影响。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Execute the offline RAG experiment and save diagnostics."""

    # 课程问题覆盖本周概念，能够检验检索是否命中正确材料。
    args = parse_args()
    queries = [
        "Why is LoRA parameter efficient?",
        "How does RAG use external knowledge?",
        "What does instruction tuning teach a model?",
    ]

    # 运行完整流程并把中间 prompt 也保存下来，方便研究型复盘。
    records = run_queries(queries, top_k=args.top_k)
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "week7_offline_rag_results.json"
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 打印简短结果，便于确认检索主题是否正确。
    for record in records:
        best_hit = record["hits"][0]
        print(
            f"Q: {record['query']}\n"
            f"Best evidence: {best_hit['title']} "
            f"(score={best_hit['score']})\n"
            f"A: {record['answer']}\n"
        )
    print(f"Artifacts written to: {OUTPUT_DIR}")


# 主入口隔离副作用，使函数可被 notebook 或测试脚本复用。
if __name__ == "__main__":
    main()
