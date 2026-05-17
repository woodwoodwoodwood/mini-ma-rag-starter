"""
Mini-MA-RAG: Planner Agent (Starter Code)

【实验任务】
实现一个基于 LLM 的规划器，将复杂问题分解为可执行的子任务列表。

【核心要求】
1. 使用结构化输出（Pydantic BaseModel）约束 LLM 的输出格式
2. 支持传入历史经验（past_experiences），用于迭代优化
3. Prompt 中必须包含 few-shot 示例，教导 LLM 如何分解问题
4. 分解的步骤必须是无重叠的（non-overlapping）

【需要你实现的方法】
- plan(): 主入口，调用 LLM 生成 PlanFormat
- _format_memory(): 将历史经验格式化为文本

【自测命令】
    python src/planner.py
"""

from typing import List, Optional
from src.llm_client import LLMClient, get_default_client
from src.prompts import PLANNER_SYSTEM_MESSAGE, PLANNER_HUMAN_MESSAGE
from src.state import PlanFormat, PlanExecState


class Planner:
    """
    规划器：基于 LLM 将问题分解为子任务列表。

    【Phase 1 实验目标】
    完成本类的实现，使其能够正确分解单跳和多跳问题。
    """

    def __init__(self, client: Optional[LLMClient] = None):
        # TODO (可选): 如果需要自定义 LLMClient（如不同的 temperature），可以在这里传入
        self.client = client or get_default_client()

    def plan(self, question: str, past_experiences: Optional[List[PlanExecState]] = None) -> List[str]:
        """
        为问题生成执行计划（子任务列表）。

        Args:
            question: 原始问题字符串
            past_experiences: 之前的执行经验，用于让 LLM 避免重复犯错

        Returns:
            子任务列表，如 ["Who is X?", "What did X do?"]

        TODO: 请实现以下逻辑：
        1. 调用 self._format_memory() 将 past_experiences 格式化为字符串
        2. 使用 PLANNER_SYSTEM_MESSAGE 和 PLANNER_HUMAN_MESSAGE 构造 prompt
        3. 调用 self.client.chat_structured() 并传入 output_schema=PlanFormat
        4. 返回 result.step（List[str]）

        提示：
        - 如果 past_experiences 为空或 None，memory 字符串应为 "empty"
        - self.client.chat_structured() 的签名参考 src/llm_client.py
        """
        # ====== 你的代码开始 ======

        # 步骤 1: 格式化历史经验
        memory = ...  # TODO

        # 步骤 2: 构造 user_message（使用 PLANNER_HUMAN_MESSAGE.format）
        user_message = ...  # TODO

        # 步骤 3: 调用 LLM 的结构化输出
        result = ...  # TODO

        # 步骤 4: 返回步骤列表
        return ...  # TODO

        # ====== 你的代码结束 ======

    def _format_memory(self, past_experiences: Optional[List[PlanExecState]]) -> str:
        """
        将历史经验格式化为文本，供 Planner 参考。

        TODO: 请实现以下逻辑：
        1. 如果 past_experiences 为空或 None，返回 "empty"
        2. 否则，遍历 past_experiences，格式化为如下文本：

           Trial 0:
           Plan: [step1, step2]
           Status: Successful Score: 8

           Trial 1:
           Plan: [step1, step2, step3]
           Status: Unsuccessful Score: 0

        提示：
        - PlanExecState 中的 plan_summary 可能为 None，需要处理
        - 使用 ", ".join(plan) 将步骤列表转为字符串
        """
        # ====== 你的代码开始 ======
        raise NotImplementedError("请实现 _format_memory 方法")
        # ====== 你的代码结束 ======


if __name__ == "__main__":
    """
    【本地测试】
    运行此文件测试你的 Planner 实现。
    预期输出：对测试问题生成 2-3 个无重叠的子任务。
    """
    from src.config import require_llm_config

    print("=" * 60)
    print("Planner 本地测试")
    print("=" * 60)

    try:
        require_llm_config()
        planner = Planner()
        test_question = (
            "What government position was held by the woman who portrayed "
            "Corliss Archer in the film Kiss and Tell?"
        )
        print(f"\n输入问题: {test_question}\n")
        steps = planner.plan(test_question)
        print("生成的 Plan:")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s}")
        print("\n[通过] Planner 测试完成")
    except NotImplementedError as e:
        print(f"\n[未完成] {e}")
        print("提示：请先实现 _format_memory 和 plan 方法")
    except Exception as e:
        print(f"\n[错误] {type(e).__name__}: {e}")
