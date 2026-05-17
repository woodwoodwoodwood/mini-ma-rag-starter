"""
Mini-MA-RAG: Step Definer Agent (Starter Code)

【实验任务】
实现一个步骤定义器，根据当前 Plan、当前步骤和历史执行结果，
生成当前步骤的具体可执行任务。

【核心要求】
1. 输出的查询必须是自包含的（不能包含 "this actress" 等指代词）
2. 必须将之前步骤的具体结果嵌入到当前查询中
3. 正确区分 "question-answering"（需要检索）和 "aggregate"（仅需汇总）

【需要你实现的方法】
- define(): 主入口，生成 StepTaskFormat
- _format_step_outputs(): 将历史步骤输出格式化为记忆文本

【自测命令】
    python src/step_definer.py
"""

from typing import List, Optional
from src.llm_client import LLMClient, get_default_client
from src.prompts import STEP_DEFINER_SYSTEM_MESSAGE, STEP_DEFINER_HUMAN_MESSAGE
from src.state import StepTaskFormat, QAAnswerFormat


class StepDefiner:
    """
    步骤定义器：将 Plan 中的某一步转化为可执行的具体查询。

    【Phase 3 实验目标】
    完成本类的实现，使 Plan 中的每一步都能被正确改写为具体查询。

    关键规则：
    - 查询中必须包含之前步骤的具体答案（如 "Shirley Temple" 而非 "this actress"）
    - type 必须是 "question-answering" 或 "aggregate" 之一
    """

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_default_client()

    def define(
        self,
        plan: List[str],
        current_step_index: int,
        step_outputs: List[QAAnswerFormat],
    ) -> StepTaskFormat:
        """
        定义当前步骤的任务。

        Args:
            plan: 完整计划列表
            current_step_index: 当前执行到第几步（从 0 开始）
            step_outputs: 之前各步骤的 QAAnswerFormat 结果

        Returns:
            StepTaskFormat: 包含 task_type 和具体 task 描述

        TODO: 请实现以下逻辑：
        1. 获取当前步骤文本：plan[current_step_index]
        2. 将 plan 列表格式化为字符串 "[step1, step2, ...]"
        3. 调用 _format_step_outputs() 将 step_outputs 格式化为记忆文本
        4. 使用 STEP_DEFINER_SYSTEM_MESSAGE 和 STEP_DEFINER_HUMAN_MESSAGE 构造 prompt
        5. 调用 self.client.chat_structured() 并传入 output_schema=StepTaskFormat
        6. 返回 result

        提示：
        - 注意 current_step_index 是 0-based
        - memory 文本应包含之前每一步的答案和置信度
        """
        # ====== 你的代码开始 ======

        # 步骤 1-3: 准备输入
        cur_step = ...  # TODO
        plan_str = ...  # TODO
        memory = ...    # TODO (调用 self._format_step_outputs)

        # 步骤 4: 构造 user_message
        user_message = ...  # TODO (使用 STEP_DEFINER_HUMAN_MESSAGE.format)

        # 步骤 5: 调用结构化输出
        result = ...  # TODO

        # 步骤 6: 返回
        return ...  # TODO

        # ====== 你的代码结束 ======

    def _format_step_outputs(self, plan: List[str], step_outputs: List[QAAnswerFormat]) -> str:
        """
        将历史步骤输出格式化为记忆文本。

        TODO: 请实现以下逻辑：
        1. 如果 step_outputs 为空，返回 "None"
        2. 否则，遍历 step_outputs，格式化为：

           Task: {plan[idx]}
           Answer: {output.answer}
           Confident: {output.rating}/10

        提示：
        - plan 和 step_outputs 是按索引一一对应的
        - 每条记录之间用换行分隔
        """
        # ====== 你的代码开始 ======
        raise NotImplementedError("请实现 _format_step_outputs 方法")
        # ====== 你的代码结束 ======


if __name__ == "__main__":
    """
    【本地测试】
    测试 StepDefiner 是否能正确改写第二步查询。
    预期输出：task.type="question-answering"
             task.task 中包含 "Shirley Temple" 而不是 "this actress"
    """
    from src.config import require_llm_config

    print("=" * 60)
    print("StepDefiner 本地测试")
    print("=" * 60)

    try:
        require_llm_config()
        sd = StepDefiner()
        plan = [
            "Who portrayed Corliss Archer in the film Kiss and Tell?",
            "What government position did this actress hold?",
        ]
        # 模拟第一步已完成
        prev_outputs = [QAAnswerFormat(analysis="", answer="Shirley Temple", success="Yes", rating=9)]

        print(f"\nPlan: {plan}")
        print(f"Previous output: {prev_outputs[0].answer}\n")

        task = sd.define(plan, current_step_index=1, step_outputs=prev_outputs)
        print(f"任务类型: {task.type}")
        print(f"任务内容: {task.task}")

        # 验收检查
        if "Shirley Temple" in task.task:
            print("\n[通过] 查询中正确包含了前一步的具体结果")
        else:
            print("\n[警告] 查询中未包含 'Shirley Temple'，请检查 Prompt 设计")

        if task.type.lower() in {"question-answering", "aggregate"}:
            print("[通过] 任务类型合法")
        else:
            print(f"[错误] 任务类型非法: {task.type}")

    except NotImplementedError as e:
        print(f"\n[未完成] {e}")
        print("提示：请先实现 _format_step_outputs 和 define 方法")
    except Exception as e:
        print(f"\n[错误] {type(e).__name__}: {e}")
