# -*- coding: utf-8 -*-
"""Arc-standard dependency parsing demo.

本脚本对应第五周“依存分析”部分。它不依赖深度学习框架，只是先把
transition-based parsing 的状态转移、oracle 和依存边构造写清楚。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


ActionType = Literal["SHIFT", "LEFT_ARC", "RIGHT_ARC"]

# 用 Literal 限定动作集合，可以在类型层面表达解析器的有限状态空间。
# 这对教学代码很有帮助，因为错误动作会更早暴露。


@dataclass(frozen=True)
class Token:
    """A token in a dependency parsing sentence."""

    # token_id 保留原句中的位置编号，form 保留可读词面。
    # 句法边只存 id，输出展示时再映射成词，能避免重复存储。
    token_id: int
    form: str


@dataclass(frozen=True)
class Arc:
    """A labeled dependency arc from head to dependent."""

    # head -> dependent 表示支配关系，label 则说明具体句法功能。
    # 这里没有把概率放进 Arc，因为本脚本展示的是 oracle 解析流程。
    head: int
    dependent: int
    label: str


@dataclass
class ParserState:
    """The mutable state used by an arc-standard parser."""

    # stack、buffer、arcs 是 transition-based parsing 的三个核心状态变量。
    # 它们共同决定下一步动作是否合法，也决定当前部分树是否完整。
    stack: list[Token]
    buffer: list[Token]
    arcs: list[Arc] = field(default_factory=list)

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot for experiment recording."""

        # 保存快照是为了做逐步可视化和错误分析；
        # 依存句法模型调试时，动作级 trace 往往比最终树更有诊断价值。
        return {
            "stack": [token.form for token in self.stack],
            "buffer": [token.form for token in self.buffer],
            "arcs": [
                {"head": arc.head, "dependent": arc.dependent, "label": arc.label}
                for arc in self.arcs
            ],
        }


def toy_sentence() -> tuple[list[Token], dict[int, int], dict[int, str]]:
    """Create a small gold tree for classroom inspection."""

    # 示例句子故意选短句，因为 arc-standard 转移过程需要逐步检查栈和缓冲区。
    # ROOT 设为 0 是依存分析常见约定，便于把主谓核心连到虚根。
    tokens = [
        Token(0, "ROOT"),
        Token(1, "I"),
        Token(2, "enjoy"),
        Token(3, "natural"),
        Token(4, "language"),
        Token(5, "processing"),
    ]

    # gold_heads[token] = head。ROOT 不需要 head，因此没有放入字典。
    # 金标准 head 字典是 oracle 的依据，不是模型预测结果。
    gold_heads = {
        1: 2,
        2: 0,
        3: 5,
        4: 5,
        5: 2,
    }
    # label 与 head 分开存储，便于区分“结构预测”和“关系分类”两个子任务。
    gold_labels = {
        1: "nsubj",
        2: "root",
        3: "amod",
        4: "compound",
        5: "obj",
    }
    return tokens, gold_heads, gold_labels


def build_gold_children(gold_heads: dict[int, int]) -> dict[int, list[int]]:
    """Invert the gold-head mapping to check whether a subtree is complete."""

    # arc-standard parser 在归约一个节点前，需要确认它的所有孩子已经挂接。
    # 因此从 head->children 的反向索引能快速判断“子树是否闭合”。
    children: dict[int, list[int]] = {}
    for dependent, head in gold_heads.items():
        children.setdefault(head, []).append(dependent)
    return children


def dependents_are_attached(
    token_id: int,
    state: ParserState,
    gold_children: dict[int, list[int]],
) -> bool:
    """Check whether all gold children of a token already appear in arcs."""

    # 已经出现在 arcs 中的 dependent 可以视为完成挂接；
    # 若某个孩子尚未挂接就归约父节点，会导致后续再也无法建立这条边。
    attached = {arc.dependent for arc in state.arcs}
    return all(child in attached for child in gold_children.get(token_id, []))


def oracle_action(
    state: ParserState,
    gold_heads: dict[int, int],
    gold_labels: dict[int, str],
    gold_children: dict[int, list[int]],
) -> tuple[ActionType, str | None] | None:
    """Choose the next gold action for an arc-standard parser."""

    # 栈中少于两个元素时无法建立依存边，只能继续 SHIFT。
    # 这个约束直接来自 arc-standard 的动作前置条件。
    if len(state.stack) < 2:
        return ("SHIFT", None) if state.buffer else None

    # 只观察栈顶两个词，是 arc-standard 局部决策的关键特征。
    # 神经依存分析器会把这里扩展成 embedding 特征。
    second_top = state.stack[-2]
    top = state.stack[-1]

    # LEFT_ARC 建立 top -> second_top。必须确保 second_top 的子树已经处理完。
    # ROOT 不能作为 dependent，因此 LEFT_ARC 排除 token_id == 0。
    if (
        second_top.token_id != 0
        and gold_heads.get(second_top.token_id) == top.token_id
        and dependents_are_attached(second_top.token_id, state, gold_children)
    ):
        return ("LEFT_ARC", gold_labels[second_top.token_id])

    # RIGHT_ARC 建立 second_top -> top。这里同样要等 top 的子树完成后再归约。
    # RIGHT_ARC 会弹出栈顶 top，所以必须确认 top 不再需要接受新的孩子。
    if (
        gold_heads.get(top.token_id) == second_top.token_id
        and dependents_are_attached(top.token_id, state, gold_children)
    ):
        return ("RIGHT_ARC", gold_labels[top.token_id])

    # 若当前不能安全归约，就从 buffer 读入新词，以暴露更多右侧上下文。
    if state.buffer:
        return ("SHIFT", None)
    return None


def apply_action(
    state: ParserState,
    action: tuple[ActionType, str | None],
) -> None:
    """Mutate parser state according to the selected action."""

    # action_type 决定状态转移方式，label 只在建立依存边时有意义。
    action_type, label = action
    if action_type == "SHIFT":
        # SHIFT 将 buffer 头部移动到 stack 顶部，相当于向右扫描句子。
        state.stack.append(state.buffer.pop(0))
        return

    # arc 动作必须带 label，否则最终树只有结构没有句法功能。
    if label is None:
        raise ValueError("Arc actions must carry a dependency label.")

    if action_type == "LEFT_ARC":
        # LEFT_ARC 建立栈顶支配次栈顶的边，并删除已经完成的 dependent。
        head = state.stack[-1]
        dependent = state.stack[-2]
        state.arcs.append(Arc(head.token_id, dependent.token_id, label))
        del state.stack[-2]
        return

    if action_type == "RIGHT_ARC":
        # RIGHT_ARC 建立次栈顶支配栈顶的边，并弹出完成的栈顶节点。
        head = state.stack[-2]
        dependent = state.stack[-1]
        state.arcs.append(Arc(head.token_id, dependent.token_id, label))
        state.stack.pop()
        return

    raise ValueError(f"Unsupported action: {action_type}")


def extract_debug_features(state: ParserState) -> dict[str, str | None]:
    """Extract hand-written parser features for observation."""

    # 这些特征是早期神经依存分析器常见输入的简化版：关注栈顶和缓冲区开头。
    # 真实模型还会加入词性、依存标签、左右子节点等更丰富的结构上下文。
    return {
        "stack_1": state.stack[-1].form if len(state.stack) >= 1 else None,
        "stack_2": state.stack[-2].form if len(state.stack) >= 2 else None,
        "buffer_1": state.buffer[0].form if len(state.buffer) >= 1 else None,
        "buffer_2": state.buffer[1].form if len(state.buffer) >= 2 else None,
    }


def run_parser() -> dict[str, object]:
    """Run oracle-guided parsing and save a step-by-step trace."""

    # 初始化状态时，ROOT 在栈中，真实词序列在 buffer 中。
    # 解析目标是把 buffer 清空，并把 stack 归约回只剩 ROOT。
    tokens, gold_heads, gold_labels = toy_sentence()
    state = ParserState(stack=[tokens[0]], buffer=tokens[1:])
    gold_children = build_gold_children(gold_heads)
    trace: list[dict[str, object]] = []

    # 每轮循环都保存动作前后状态，便于在报告中解释每条边如何生成。
    while state.buffer or len(state.stack) > 1:
        before = state.snapshot()
        features = extract_debug_features(state)
        action = oracle_action(state, gold_heads, gold_labels, gold_children)
        if action is None:
            # oracle 提前停止说明金标准树或动作前置条件有矛盾。
            raise RuntimeError("Oracle stopped before parsing finished.")

        apply_action(state, action)
        trace.append(
            {
                "before": before,
                "features": features,
                "action": {"type": action[0], "label": action[1]},
                "after": state.snapshot(),
            }
        )

    # 按 dependent 排序让最终输出与原句词序一致，阅读时更直观。
    arcs = sorted(state.arcs, key=lambda arc: arc.dependent)
    result = {
        "sentence": [token.form for token in tokens[1:]],
        "num_steps": len(trace),
        "transition_trace": trace,
        "final_arcs": [
            {"head": arc.head, "dependent": arc.dependent, "label": arc.label}
            for arc in arcs
        ],
    }
    return result


def main() -> None:
    # 主函数只负责运行演示和落盘 trace；解析逻辑留在 run_parser 中。
    result = run_parser()
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    # JSON trace 能被周报、笔记或后续可视化脚本复用。
    (output_dir / "dependency_parser_trace.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("依存句法分析演示完成，转移过程已写入 outputs/dependency_parser_trace.json")


if __name__ == "__main__":
    main()
