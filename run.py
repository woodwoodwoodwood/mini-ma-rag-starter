"""
Mini-MA-RAG: 主入口脚本

【实验任务】
实现两个主函数，支持在测试集上批量运行 Agent-RAG 和 Naive RAG，
并通过命令行参数切换模式、指定输出文件等。

【使用示例】
    python run.py --mode agent --output results_agent.jsonl
    python run.py --mode naive --output results_naive.jsonl
    python run.py --mode agent --start 0 --end 5 --output results_sample.jsonl
    python run.py --mode agent --input data/test_set_hard.jsonl --output results_hard.jsonl
"""

import argparse
import json
import os
import sys
from typing import Optional

from tqdm import tqdm

from src.config import require_llm_config, TEST_SET_PATH
from src.retriever import get_retriever
from src.rag_agent import RAGAgent
from src.executor import PlanExecutor
from graph.workflow import AgentWorkflow
from baselines.naive_rag import NaiveRAG


# ================================================================
# Agent-RAG 批量运行
# ================================================================

def run_agent_rag(
    output_path: str,
    start: int = 0,
    end: Optional[int] = None,
    test_set_path: Optional[str] = None,
) -> None:
    """在测试集上批量运行 Agent-RAG，结果写入 JSONL 文件。

    输出格式（每行一个 JSON 对象）:
    {
        "id": "q001",
        "question": "...",
        "ground_truth": "...",
        "predicted": "...",
        "status": "Successful",
        "score": 9,
        "plan": ["step1", "step2"],
        "step_outputs": [
            {"step": "...", "success": "Yes", "rating": 10},
            ...
        ]
    }
    """
    require_llm_config()

    # 1. 加载测试集
    test_path = test_set_path or TEST_SET_PATH
    with open(test_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    items = items[start:end] if end is not None else items[start:]
    print(f"加载测试集: {test_path}，共 {len(items)} 条（[{start}:{end}]）")

    # 2. 初始化组件
    print("初始化 Agent-RAG 组件...")
    retriever = get_retriever()
    rag = RAGAgent(retriever)
    executor = PlanExecutor(rag)
    workflow = AgentWorkflow(executor)
    print("初始化完成\n")

    # 3. 遍历执行
    results = []
    for item in tqdm(items, desc="Agent-RAG"):
        question = item["question"]
        ground_truth = item.get("answer", "")

        try:
            state = workflow.run(question)

            # 从 AgentState 提取关键信息
            plan = state.get("plan", [])
            final_answer = state.get("final_answer", "")
            past_exp = state.get("past_exp", [])

            # 最近一次执行的状态
            last_exp = past_exp[-1] if past_exp else {}
            summary = last_exp.get("plan_summary")
            status = summary.output if summary else "Unknown"
            score = summary.score if summary else 0

            # 提取每步的输出详情
            step_outputs_raw = last_exp.get("step_output", [])
            step_questions_raw = last_exp.get("step_question", [])
            step_outputs = []
            for idx, step_out in enumerate(step_outputs_raw):
                step_info = {
                    "step": step_questions_raw[idx].task if idx < len(step_questions_raw) else f"Step {idx+1}",
                    "success": step_out.success,
                    "rating": step_out.rating,
                }
                step_outputs.append(step_info)

            result = {
                "id": item["id"],
                "question": question,
                "ground_truth": ground_truth,
                "predicted": final_answer,
                "status": status,
                "score": score,
                "plan": plan,
                "step_outputs": step_outputs,
            }
        except Exception as e:
            result = {
                "id": item["id"],
                "question": question,
                "ground_truth": ground_truth,
                "predicted": f"[ERROR] {type(e).__name__}: {e}",
                "status": "Error",
                "score": 0,
                "plan": [],
                "step_outputs": [],
            }

        results.append(result)

    # 4. 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 5. 简要统计
    succeeded = sum(1 for r in results if r["status"] == "Successful")
    print(f"\n完成！结果已写入: {output_path}")
    print(f"共 {len(results)} 条 | Successful: {succeeded} | Unsuccessful/Error: {len(results) - succeeded}")


# ================================================================
# Naive RAG 批量运行
# ================================================================

def run_naive_rag(
    output_path: str,
    start: int = 0,
    end: Optional[int] = None,
    test_set_path: Optional[str] = None,
) -> None:
    """在测试集上批量运行 Naive RAG，结果写入 JSONL 文件。

    输出格式（每行一个 JSON 对象）:
    {
        "id": "q001",
        "question": "...",
        "ground_truth": "...",
        "predicted": "...",
        "documents": [...],
        "doc_ids": [...]
    }
    """
    require_llm_config()

    # 1. 加载测试集
    test_path = test_set_path or TEST_SET_PATH
    with open(test_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    items = items[start:end] if end is not None else items[start:]
    print(f"加载测试集: {test_path}，共 {len(items)} 条（[{start}:{end}]）")

    # 2. 初始化组件
    print("初始化 Naive RAG...")
    retriever = get_retriever()
    naive = NaiveRAG(retriever)
    print("初始化完成\n")

    # 3. 遍历执行
    results = []
    for item in tqdm(items, desc="Naive RAG"):
        question = item["question"]
        ground_truth = item.get("answer", "")

        try:
            ans = naive.answer(question)
            result = {
                "id": item["id"],
                "question": question,
                "ground_truth": ground_truth,
                "predicted": ans.get("answer", ""),
                "documents": ans.get("documents", []),
                "doc_ids": ans.get("doc_ids", []),
            }
        except Exception as e:
            result = {
                "id": item["id"],
                "question": question,
                "ground_truth": ground_truth,
                "predicted": f"[ERROR] {type(e).__name__}: {e}",
                "documents": [],
                "doc_ids": [],
            }

        results.append(result)

    # 4. 写入文件
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n完成！结果已写入: {output_path}")
    print(f"共 {len(results)} 条")


# ================================================================
# 命令行入口
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mini-MA-RAG: 在测试集上运行 RAG 实验"
    )
    parser.add_argument(
        "--mode",
        choices=["agent", "naive"],
        default="agent",
        help="运行模式: agent (Agent-RAG) 或 naive (Naive RAG 基线)",
    )
    parser.add_argument(
        "--output",
        default="results.jsonl",
        help="结果输出文件路径 (默认: results.jsonl)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="测试集起始索引 (默认: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="测试集结束索引 (默认: 全部)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="自定义测试集路径 (默认: data/test_set.jsonl)",
    )

    args = parser.parse_args()

    if args.mode == "agent":
        run_agent_rag(args.output, args.start, args.end, args.input)
    else:
        run_naive_rag(args.output, args.start, args.end, args.input)
