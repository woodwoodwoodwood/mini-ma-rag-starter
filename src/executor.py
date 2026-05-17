"""
Mini-MA-RAG: Plan Executor (Starter Code)

【实验任务】
实现一个计划执行器，负责驱动 Plan 的逐步执行：
1. 循环调用 StepDefiner 决定当前步骤任务
2. 根据任务类型分发到 RAGAgent（question-answering）或 Aggregator（aggregate）
3. 收集所有中间结果
4. 所有步骤完成后调用 Summarizer 汇总结果

【核心要求】
1. 正确处理 question-answering 和 aggregate 两种任务类型
2. 维护 step_outputs 列表，传递给 StepDefiner 作为历史上下文
3. 支持提前终止（某一步 success="No" 且 rating < 3 时停止）

【需要你实现的方法】
- Aggregator.run(): 处理 aggregate 类型任务（无需检索）
- PlanExecutor.execute(): 主入口，编排整个 Plan 的执行

【自测命令】
    python src/executor.py
"""

from typing import List
from src.step_definer import StepDefiner
from src.rag_agent import RAGAgent
from src.summarizer import Summarizer
from src.state import PlanExecState, QAAnswerFormat
from src.llm_client import LLMClient, get_default_client
from src.prompts import QA_SYSTEM_MESSAGE


class Aggregator:
    """
    聚合器：处理 type='aggregate' 的步骤。

    这类步骤不需要外部检索，只需基于历史结果进行推理汇总。

    TODO (可选): 当前实现使用普通字符串生成。
    进阶要求：改用结构化输出（QAAnswerFormat）。
    """

    def __init__(self, client=None):
        self.client = client or get_default_client()

    def run(self, query: str, memory: List[QAAnswerFormat]) -> QAAnswerFormat:
        """
        基于历史结果回答聚合类问题。

        Args:
            query: 聚合问题
            memory: 之前所有步骤的 QAAnswerFormat 结果

        Returns:
            QAAnswerFormat: 聚合答案

        TODO: 请实现以下逻辑：
        1. 将 memory 格式化为文本（每步的答案和置信度）
        2. 构造 prompt，要求 LLM 基于历史结果回答 query
        3. 调用 self.client.chat() 生成答案（或 chat_structured 输出 QAAnswerFormat）
        4. 返回 QAAnswerFormat

        提示：
        - 聚合类问题如 "Compare A and B"、"Which one is larger?"
        - 不需要检索，所有信息都在 memory 中
        """
        # ====== 你的代码开始 ======

        # 步骤 1: 格式化 memory
        memory_text = ...  # TODO

        # 步骤 2: 构造 prompt
        prompt = ...  # TODO

        # 步骤 3: 调用 LLM
        raw = ...  # TODO

        # 步骤 4: 返回 QAAnswerFormat
        return QAAnswerFormat(
            analysis="Aggregated from previous steps",
            answer=raw,
            success="Yes",
            rating=8,
        )

        # ====== 你的代码结束 ======


class PlanExecutor:
    """
    计划执行器：驱动 Plan 的逐步执行。

    【Phase 3 实验目标】
    完成本类的实现，使其能够正确执行多步 Plan 并汇总结果。

    核心循环逻辑：
    for each step in plan:
        task = step_definer.define(plan, step_index, history)
        if task.type == 'question-answering':
            result = rag_agent.run(task.task)
        else:
            result = aggregator.run(task.task, history)
        history.append(result)
    summary = summarizer.summarize(...)
    """

    def __init__(
        self,
        rag_agent: RAGAgent,
        step_definer: StepDefiner = None,
        summarizer: Summarizer = None,
    ):
        self.rag_agent = rag_agent
        self.step_definer = step_definer or StepDefiner()
        self.summarizer = summarizer or Summarizer()
        self.aggregator = Aggregator()

    def execute(self, original_question: str, plan: List[str]) -> PlanExecState:
        """
        执行计划并返回完整状态。

        Args:
            original_question: 原始问题
            plan: 子任务列表

        Returns:
            PlanExecState: 包含所有中间结果和最终汇总

        TODO: 请实现以下逻辑：
        1. 初始化 PlanExecState（所有累加字段为空列表，plan_summary 为 None，stop 为 False）
        2. 遍历 plan 中的每一步：
           a. 调用 self.step_definer.define(plan, step_idx, step_outputs)
           b. 将 task 追加到 state["step_question"]
           c. 根据 task.type 分发：
              - "aggregate" → self.aggregator.run(task.task, step_outputs)
              - 其他（默认 "question-answering"）→ self.rag_agent.run(task.task)
           d. 将结果追加到 state["step_output"]
           e. 将 doc_ids 和 notes 追加到对应字段（aggregate 类型为空列表）
           f. 检查终止条件：如果 result.success == "No" 且 result.rating < 3，提前 break
        3. 调用 self.summarizer.summarize(original_question, plan, step_outputs)
        4. 将 summary 写入 state["plan_summary"] ，设置 stop=True
        5. 返回 state

        提示：
        - 建议维护一个局部变量 step_outputs（List[QAAnswerFormat]），方便传给 StepDefiner
        - 打印日志有助于调试（如 print(f"[Step {idx}] ...")）
        """
        # ====== 你的代码开始 ======

        # 步骤 1: 初始化状态
        state = ...  # TODO

        # 步骤 2: 遍历执行每一步
        step_outputs = []  # 用于传递给 StepDefiner 的历史结果

        for step_idx in range(len(plan)):
            # TODO: 实现循环体
            pass

        # 步骤 3-5: 汇总并返回
        # TODO

        raise NotImplementedError("请实现 execute 方法")

        # ====== 你的代码结束 ======


if __name__ == "__main__":
    """
    【本地测试】
    测试 PlanExecutor 是否能正确执行两步 Plan。
    预期输出：最终答案为 "Chief of Protocol of the United States"
    """
    from src.config import require_llm_config

    print("=" * 60)
    print("PlanExecutor 本地测试")
    print("=" * 60)

    try:
        require_llm_config()
        from src.retriever import get_retriever
        retriever = get_retriever()
        rag = RAGAgent(retriever)
        executor = PlanExecutor(rag)

        question = (
            "What government position was held by the woman who portrayed "
            "Corliss Archer in the film Kiss and Tell?"
        )
        plan = [
            "Who portrayed Corliss Archer in the film Kiss and Tell?",
            "What government position did this actress hold?",
        ]

        print(f"\n原始问题: {question}")
        print(f"Plan: {plan}\n")

        result = executor.execute(question, plan)
        summary = result["plan_summary"]

        print(f"\n=== 执行结果 ===")
        print(f"状态: {summary.output}")
        print(f"答案: {summary.answer}")
        print(f"置信度: {summary.score}")

        if summary.output == "Successful":
            print("\n[通过] PlanExecutor 测试完成")
        else:
            print("\n[警告] 执行未成功，请检查各组件实现")

    except NotImplementedError as e:
        print(f"\n[未完成] {e}")
        print("提示：请先实现 Aggregator.run 和 PlanExecutor.execute 方法")
    except Exception as e:
        print(f"\n[错误] {type(e).__name__}: {e}")
