"""
Mini-MA-RAG: Naive RAG Baseline (Starter Code)

【实验任务】
实现朴素 RAG 基线，用于与 Agent-RAG 进行对比实验。

【核心要求】
1. 一次性检索固定数量（Top-K）的文档
2. 将检索到的文档原文直接拼接到 prompt 中
3. 调用 LLM 生成答案

【流程】
    用户提问 → 检索 Top-K 文档 → 拼接原文为 context → LLM 生成答案

【自测命令】
    python baselines/naive_rag.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import LLMClient, get_default_client
from src.retriever import Retriever, get_retriever
from src.prompts import NAIVE_RAG_SYSTEM_MESSAGE, NAIVE_RAG_HUMAN_MESSAGE


class NaiveRAG:
    """
    朴素 RAG：一次性检索 + 直接生成，无分解、无抽取、无验证。
    """

    def __init__(self, retriever: Retriever, client=None):
        self.retriever = retriever
        self.client = client or get_default_client()

    def answer(self, question: str, top_k: int = 5) -> dict:
        """
        回答问题。

        Args:
            question: 用户问题
            top_k: 检索文档数量

        Returns:
            dict: {
                "question": question,
                "documents": [...],   # 检索到的文档原文
                "doc_ids": [...],     # 文档 ID
                "answer": str,        # LLM 生成的答案
            }

        TODO: 请实现以下逻辑：
        1. 调用 self.retriever.search(question, top_k=top_k) 检索文档
        2. 将文档和 doc_id 配对，格式化为：
           [doc_id1] text1
           [doc_id2] text2
           ...
        3. 使用 NAIVE_RAG_SYSTEM_MESSAGE 和 NAIVE_RAG_HUMAN_MESSAGE 构造 prompt
        4. 调用 self.client.chat() 生成答案
        5. 返回包含 question/documents/doc_ids/answer 的字典

        提示：
        - 这是最简单的 RAG 实现，没有 Extractor 和 Step Definer
        - 所有文档直接塞进 prompt，注意上下文长度限制
        """
        # ====== 你的代码开始 ======

        # 步骤 1: 检索
        docs, doc_ids = ...  # TODO

        # 步骤 2: 格式化 context
        context = ...  # TODO

        # 步骤 3-4: 构造 prompt 并生成答案
        user_message = ...  # TODO
        answer_text = ...  # TODO

        # 步骤 5: 返回结果
        return {
            "question": question,
            "documents": ...,
            "doc_ids": ...,
            "answer": ...,
        }

        # ====== 你的代码结束 ======


def run_naive_rag_on_dataset(test_path: str, output_path: str, top_k: int = 5):
    """
    在测试集上批量运行 Naive RAG。

    TODO: 请实现以下逻辑：
    1. 加载测试集（jsonl 格式，每行包含 id, question, answer）
    2. 对每个问题调用 NaiveRAG.answer()
    3. 收集结果并写入 output_path（jsonl 格式）

    提示：
    - 使用 json.loads() 读取每行
    - 结果格式参考 Agent-RAG 的 run.py
    """
    # ====== 你的代码开始 ======
    raise NotImplementedError("请实现 run_naive_rag_on_dataset 函数")
    # ====== 你的代码结束 ======


if __name__ == "__main__":
    """
    【本地测试】
    测试 NaiveRAG 是否能回答单跳问题。
    预期输出：对 "Who portrayed Corliss Archer..." 给出 "Shirley Temple"
    """
    from src.config import require_llm_config

    print("=" * 60)
    print("NaiveRAG 本地测试")
    print("=" * 60)

    try:
        require_llm_config()
        retriever = get_retriever()
        rag = NaiveRAG(retriever)

        question = "Who portrayed Corliss Archer in the film Kiss and Tell?"
        print(f"\n问题: {question}\n")

        result = rag.answer(question)
        print(f"检索到 {len(result['documents'])} 篇文档")
        print(f"答案: {result['answer']}")

        if result["answer"]:
            print("\n[通过] NaiveRAG 测试完成")
        else:
            print("\n[警告] 答案为空")

    except NotImplementedError as e:
        print(f"\n[未完成] {e}")
        print("提示：请先实现 NaiveRAG.answer 方法")
    except Exception as e:
        print(f"\n[错误] {type(e).__name__}: {e}")
