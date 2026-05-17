"""
Mini-MA-RAG: Summarizer Agent (Starter Code)

【实验任务】
实现一个汇总器，当 Plan Executor 完成所有步骤后，
整合各步骤结果，输出原始问题的最终答案或报告失败原因。

【核心要求】
1. 输出必须是结构化格式（PlanSummaryFormat）
2. 能区分 Successful（成功得到答案）和 Unsuccessful（无法回答）
3. score 应基于各步骤的 rating 综合计算

【需要你实现的方法】
- summarize(): 主入口，汇总所有步骤结果
- _format_memory(): 将步骤输出格式化为文本

【自测命令】
    python src/summarizer.py
"""

from typing import List, Optional
from src.llm_client import LLMClient, get_default_client
from src.prompts import SUMMARIZER_SYSTEM_MESSAGE, SUMMARIZER_HUMAN_MESSAGE
from src.state import PlanSummaryFormat, QAAnswerFormat


class Summarizer:
    """
    汇总器：整合多步执行结果，生成最终答案。

    【Phase 3 实验目标】
    完成本类的实现，使其能够正确判断计划执行是否成功，并给出最终答案。

    输出规则：
    - Successful: 所有步骤成功，或虽部分失败但足以推断答案
    - Unsuccessful: 无法从步骤输出中推断答案
    """

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_default_client()

    def summarize(
        self,
        original_question: str,
        plan: List[str],
        step_outputs: List[QAAnswerFormat],
    ) -> PlanSummaryFormat:
        """
        汇总执行结果。

        Args:
            original_question: 原始问题
            plan: 执行的计划
            step_outputs: 每步的 QA 输出

        Returns:
            PlanSummaryFormat: 包含 output, answer, score

        TODO: 请实现以下逻辑：
        1. 将 plan 列表格式化为字符串 "[step1, step2, ...]"
        2. 调用 _format_memory() 将 step_outputs 格式化为记忆文本
        3. 使用 SUMMARIZER_SYSTEM_MESSAGE 和 SUMMARIZER_HUMAN_MESSAGE 构造 prompt
        4. 调用 self.client.chat_structured() 并传入 output_schema=PlanSummaryFormat
        5. 返回 result

        提示：
        - memory 文本应包含每步的问题、答案、成功状态和置信度
        - Summarizer 的 Prompt 中已经指导了 LLM 如何计算 score
        """
        # ====== 你的代码开始 ======

        plan_str = ...  # TODO
        memory = ...    # TODO (调用 self._format_memory)
        user_message = ...  # TODO

        result = ...  # TODO
        return ...  # TODO

        # ====== 你的代码结束 ======

    def _format_memory(self, plan: List[str], step_outputs: List[QAAnswerFormat]) -> str:
        """
        将步骤输出格式化为记忆文本。

        TODO: 请实现以下逻辑：
        1. 遍历 step_outputs，格式化为：

           Step {idx+1}: {plan[idx]}
             Answer: {output.answer}
             Success: {output.success}
             Rating: {output.rating}/10
             Analysis: {output.analysis}

        提示：
        - 注意使用 idx+1 作为步骤编号（从 1 开始更友好）
        - 每条记录之间用空行分隔，增强可读性
        """
        # ====== 你的代码开始 ======
        raise NotImplementedError("请实现 _format_memory 方法")
        # ====== 你的代码结束 ======


if __name__ == "__main__":
    """
    【本地测试】
    测试 Summarizer 是否能正确汇总两步结果。
    预期输出：output="Successful", answer="Chief of Protocol of the United States"
    """
    from src.config import require_llm_config

    print("=" * 60)
    print("Summarizer 本地测试")
    print("=" * 60)

    try:
        require_llm_config()
        summarizer = Summarizer()

        plan = [
            "Who portrayed Corliss Archer in the film Kiss and Tell?",
            "What government position did this actress hold?",
        ]
        outputs = [
            QAAnswerFormat(analysis="", answer="Shirley Temple", success="Yes", rating=9),
            QAAnswerFormat(analysis="", answer="Chief of Protocol of the United States", success="Yes", rating=8),
        ]
        question = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"

        print(f"\n原始问题: {question}")
        print(f"Plan: {plan}")
        print(f"步骤结果: {[o.answer for o in outputs]}\n")

        result = summarizer.summarize(question, plan, outputs)
        print(f"结果: {result.output}")
        print(f"答案: {result.answer}")
        print(f"置信度: {result.score}")

        if result.output == "Successful" and result.answer:
            print("\n[通过] Summarizer 测试完成")
        else:
            print("\n[警告] 汇总结果异常，请检查 Prompt 设计")

    except NotImplementedError as e:
        print(f"\n[未完成] {e}")
        print("提示：请先实现 _format_memory 和 summarize 方法")
    except Exception as e:
        print(f"\n[错误] {type(e).__name__}: {e}")
