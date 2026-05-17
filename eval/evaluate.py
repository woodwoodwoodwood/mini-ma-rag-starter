"""
Mini-MA-RAG: 评估模块

支持两种评估方式：
1. LLM-as-Judge: 用 GPT-4 判断预测答案与标准答案是否语义等价
2. Exact Match / 包含匹配: 简单字符串匹配

输入格式：jsonl，每行包含 {"question", "ground_truth", "predicted", ...}
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import List, Dict
from tqdm import tqdm

from src.llm_client import LLMClient, get_default_client
from src.prompts import JUDGE_SYSTEM_MESSAGE, JUDGE_HUMAN_MESSAGE


class Evaluator:
    """评估器"""

    def __init__(self, judge_client: LLMClient = None, use_llm_judge: bool = True):
        self.use_llm_judge = use_llm_judge
        self.judge = judge_client or get_default_client()

    def evaluate_file(self, result_path: str) -> Dict:
        """评估结果文件"""
        items = []
        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line.strip()))

        correct = 0
        total = len(items)
        details = []

        for item in tqdm(items, desc="Evaluating"):
            gt = item.get("ground_truth", "").strip()
            pred = item.get("predicted", "").strip()
            question = item.get("question", "")

            if self.use_llm_judge:
                is_correct = self._llm_judge(question, gt, pred)
            else:
                is_correct = self._exact_match(gt, pred)

            if is_correct:
                correct += 1

            details.append({
                "id": item.get("id", ""),
                "question": question,
                "ground_truth": gt,
                "predicted": pred,
                "correct": is_correct,
            })

        accuracy = correct / total if total > 0 else 0.0
        return {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "details": details,
        }

    def _llm_judge(self, question: str, ground_truth: str, predicted: str) -> bool:
        """使用 LLM 判断语义等价性"""
        if not predicted or predicted.strip().lower() in {"n/a", "none", "", "unsuccessful"}:
            return False

        user_message = JUDGE_HUMAN_MESSAGE.format(
            question=question,
            ground_truth=ground_truth,
            predicted=predicted,
        )
        try:
            response = self.judge.chat(
                system_message=JUDGE_SYSTEM_MESSAGE,
                user_message=user_message,
                temperature=0.0,
            )
            return "CORRECT" in response.upper()
        except Exception as e:
            print(f"[WARN] LLM Judge 失败: {e}, fallback 到精确匹配")
            return self._exact_match(ground_truth, predicted)

    def _exact_match(self, ground_truth: str, predicted: str) -> bool:
        """精确匹配（忽略大小写和标点）"""
        def normalize(s: str) -> str:
            return s.lower().strip().rstrip(".").strip()
        return normalize(ground_truth) == normalize(predicted)


def main():
    parser = argparse.ArgumentParser(description="评估 RAG 实验结果")
    parser.add_argument("input", help="结果 jsonl 文件路径")
    parser.add_argument("--output", default="eval_result.json", help="评估报告输出路径")
    parser.add_argument("--no-llm-judge", action="store_true", help="禁用 LLM Judge，使用精确匹配")
    args = parser.parse_args()

    evaluator = Evaluator(use_llm_judge=not args.no_llm_judge)
    result = evaluator.evaluate_file(args.input)

    print(f"\n========== 评估结果 ==========")
    print(f"总计: {result['total']}")
    print(f"正确: {result['correct']}")
    print(f"准确率: {result['accuracy'] * 100:.2f}%")
    print(f"==============================\n")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 详细报告已保存至 {args.output}")


if __name__ == "__main__":
    main()
