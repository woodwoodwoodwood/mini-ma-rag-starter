"""
Mini-MA-RAG: LangGraph Workflow

【实验任务】
使用 LangGraph 定义 Agent-RAG 的完整工作流，包括：
1. Planner Node: 生成/优化执行计划
2. Executor Node: 执行计划并汇总结果
3. 可选迭代反思：当答案质量不高时重新规划

【核心要求】
1. 正确构建 StateGraph(AgentState)
2. planner_node 和 executor_node 正确读写状态
3. 支持条件边实现迭代反思
4. run() 方法作为外部调用的统一入口

【自测命令】
    python graph/workflow.py
"""

from typing import Optional

from langgraph.graph import StateGraph, START, END

from src.state import AgentState
from src.planner import Planner
from src.executor import PlanExecutor
from src.config import MAX_ITERATIONS


class AgentWorkflow:
    """Agent-RAG 的 LangGraph 工作流。

    工作流结构:
        START → planner_node → executor_node → END
                                    ↑              │
                                    └──(低分重试)──┘

    状态说明:
    - planner_node:  读取 original_question 和 past_exp，写入 plan
    - executor_node: 读取 plan，执行后通过 past_exp 累积结果，写入 final_answer
    - 条件边:        根据 plan_summary.score 和 iteration 决定是否重新规划

    关键字段 (AgentState):
    - original_question: 原始问题
    - plan:              Planner 生成的子任务列表
    - past_exp:          历史执行经验 (List[PlanExecState]，通过 operator.add 累加)
    - final_answer:      最终答案
    - iteration:         当前迭代次数
    """

    def __init__(self, plan_executor: PlanExecutor, planner: Optional[Planner] = None):
        """
        Args:
            plan_executor: PlanExecutor 实例（已注入 RAGAgent, StepDefiner, Summarizer）
            planner:       Planner 实例，如果不传则使用默认构造
        """
        self.executor = plan_executor
        self.planner = planner or Planner()
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建 LangGraph 状态图并编译。

        节点:
            planner  - 调用 Planner.plan() 生成子任务列表
            executor - 调用 PlanExecutor.execute() 执行计划并汇总

        边:
            START    → planner      (无条件)
            planner  → executor     (无条件)
            executor → END          (条件: score >= 5 或 iteration >= MAX_ITERATIONS)
            executor → planner      (条件: score < 5 且 iteration < MAX_ITERATIONS)
        """
        graph = StateGraph(AgentState)

        # ============================================================
        # Planner 节点
        # ============================================================
        def planner_node(state: AgentState) -> dict:
            """生成或优化执行计划。

            首次执行：仅基于 original_question 生成计划。
            迭代重试：将 past_exp (历史经验) 传给 Planner，帮助其改进计划。
            """
            question = state["original_question"]
            past = state.get("past_exp", []) or []

            plan = self.planner.plan(
                question,
                past_experiences=past if past else None,
            )
            return {"plan": plan}

        # ============================================================
        # Executor 节点
        # ============================================================
        def executor_node(state: AgentState) -> dict:
            """执行计划并汇总结果。

            1. 调用 PlanExecutor.execute(question, plan)
            2. 从 PlanExecState 中提取 plan_summary 作为最终答案
            3. 通过 past_exp (reducer: operator.add) 累积执行经验
            4. 递增 iteration 计数
            """
            question = state["original_question"]
            plan = state["plan"]

            result = self.executor.execute(question, plan)

            # 从 PlanExecState 提取最终答案
            summary = result.get("plan_summary")
            final_answer = summary.answer if summary else ""

            return {
                "past_exp": [result],
                "final_answer": final_answer,
                "iteration": state.get("iteration", 0) + 1,
            }

        # ============================================================
        # 条件路由: 根据执行质量决定是否重新规划
        # ============================================================
        def should_replan(state: AgentState) -> str:
            """判断是否需要重新规划。

            规则:
            - 如果 iteration >= MAX_ITERATIONS，无条件结束
            - 如果最近一次 plan_summary.score < 5，返回 planner 重新规划
            - 否则结束

            注意: 返回 END (langgraph 内置常量) 表示终止图执行。
            """
            iteration = state.get("iteration", 0)
            if iteration >= MAX_ITERATIONS:
                return END

            past = state.get("past_exp", [])
            if past:
                last_result = past[-1]
                summary = last_result.get("plan_summary")
                if summary is not None and summary.score < 5:
                    return "planner"

            return END

        # ============================================================
        # 组装图
        # ============================================================
        graph.add_node("planner", planner_node)
        graph.add_node("executor", executor_node)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "executor")
        graph.add_conditional_edges(
            "executor",
            should_replan,
            {"planner": "planner", END: END},
        )

        return graph.compile()

    def run(self, question: str) -> dict:
        """运行工作流并返回最终状态。

        Args:
            question: 用户的原始问题

        Returns:
            AgentState 字典，包含以下关键字段:
            - final_answer (str):  最终答案
            - plan (List[str]):    执行的计划
            - past_exp (List[PlanExecState]): 每次执行的历史记录
            - iteration (int):     实际迭代次数
        """
        initial_state: AgentState = {
            "original_question": question,
            "plan": [],
            "past_exp": [],
            "final_answer": "",
            "iteration": 0,
        }
        return self.graph.invoke(initial_state)


# ================================================================
# 本地测试
# ================================================================
if __name__ == "__main__":
    """
    测试 AgentWorkflow 是否能正确编排 Planner → Executor 流程。
    预期输出: 最终答案包含 "Chief of Protocol of the United States"
    """
    from src.config import require_llm_config

    print("=" * 60)
    print("AgentWorkflow 本地测试")
    print("=" * 60)

    try:
        require_llm_config()
        from src.retriever import get_retriever
        from src.rag_agent import RAGAgent

        # 1. 初始化各组件
        retriever = get_retriever()
        rag = RAGAgent(retriever)
        executor = PlanExecutor(rag)
        workflow = AgentWorkflow(executor)

        # 2. 测试问题 (多跳推理)
        question = (
            "What government position was held by the woman who portrayed "
            "Corliss Archer in the film Kiss and Tell?"
        )
        print(f"\n原始问题: {question}\n")

        # 3. 运行工作流
        result = workflow.run(question)

        # 4. 打印结果
        print(f"=== 工作流执行结果 ===")
        print(f"Plan: {result.get('plan', [])}")
        print(f"迭代次数: {result.get('iteration', 0)}")
        print(f"最终答案: {result.get('final_answer', 'N/A')}")

        # 5. 打印每轮执行详情
        past = result.get("past_exp", [])
        for i, exp in enumerate(past):
            summary = exp.get("plan_summary")
            if summary:
                print(f"\n--- Trial {i} ---")
                print(f"  状态: {summary.output}")
                print(f"  答案: {summary.answer}")
                print(f"  评分: {summary.score}/10")

        final = result.get("final_answer", "")
        if final and final != "N/A":
            print("\n[通过] AgentWorkflow 测试完成")
        else:
            print("\n[警告] 工作流执行完成但可能未得到有效答案，请检查各组件实现")

    except NotImplementedError as e:
        print(f"\n[未完成] {e}")
        print("提示：请先完成 Planner、RAGAgent、PlanExecutor 等依赖组件的实现")
    except Exception as e:
        print(f"\n[错误] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
