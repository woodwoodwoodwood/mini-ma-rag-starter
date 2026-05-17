"""
Mini-MA-RAG: RAG Agent (Starter Code)

【实验任务】
实现一个完整的 RAG Agent，包含三个核心环节：
1. Retrieve: 调用 Retriever 检索相关文档
2. Extract: 从每篇文档中抽取与问题相关的信息（Notes）
3. Generate: 基于抽取的 Notes 生成答案

【核心要求】
1. Extractor 必须能有效过滤无关信息，输出 "No related information from this document."
2. QA Agent 的输出必须是结构化格式（QAAnswerFormat），包含 analysis / answer / success / rating
3. Generate 时要把 doc_id 和 note 配对展示，方便溯源

【需要你实现的方法】
- run(): 主入口，编排 retrieve → extract → generate
- _extract(): 对单篇文档抽取信息
- _generate(): 基于 notes 生成答案

【自测命令】
    python src/rag_agent.py
"""

from typing import List, Optional
from src.llm_client import LLMClient, get_default_client
from src.retriever import Retriever
from src.prompts import EXTRACTOR_SYSTEM_MESSAGE, EXTRACTOR_HUMAN_MESSAGE, QA_SYSTEM_MESSAGE, QA_HUMAN_MESSAGE
from src.state import RagState, QAAnswerFormat, ExtractResultFormat


class RAGAgent:
    """
    RAG Agent：执行"检索 → 抽取 → 生成"的完整流程。

    【Phase 2 实验目标】
    完成本类的实现，使其能够正确回答单个子问题。

    与 Naive RAG 的关键区别：
    - Naive RAG: 检索文档 → 直接把原文塞进 prompt → 生成答案
    - RAG Agent: 检索文档 → 先提炼 notes → 把 notes 塞进 prompt → 生成答案
    """

    def __init__(
        self,
        retriever: Retriever,
        client: Optional[LLMClient] = None,
    ):
        self.retriever = retriever
        self.client = client or get_default_client()

    def run(self, question: str) -> RagState:
        """
        执行完整的 RAG 流程。

        Returns:
            RagState 包含 documents, doc_ids, notes, final_raw_answer

        TODO: 请实现以下逻辑：
        1. 调用 self.retriever.search(question) 检索 Top-K 文档
        2. 对每篇文档调用 self._extract(doc, question) 抽取 notes
        3. 调用 self._generate(question, notes, doc_ids) 生成答案
        4. 组装并返回 RagState

        提示：
        - retriever.search() 返回 (list_docs, list_doc_ids)
        - notes 列表应与 documents 列表按索引一一对应
        """
        # ====== 你的代码开始 ======

        # 步骤 1: 检索文档
        docs, doc_ids = ...  # TODO

        # 步骤 2: 抽取 notes（对每篇文档分别调用 _extract）
        notes = ...  # TODO

        # 步骤 3: 生成答案
        answer = ...  # TODO (调用 self._generate)

        # 步骤 4: 返回 RagState
        return {
            "question": question,
            "documents": ...,
            "doc_ids": ...,
            "notes": ...,
            "final_raw_answer": ...,
        }

        # ====== 你的代码结束 ======

    def _extract(self, passage: str, question: str) -> str:
        """
        从单篇文档中抽取与问题相关的信息。

        TODO: 请实现以下逻辑：
        1. 如果 passage 为空或太短（<10 字符），直接返回 "No related information from this document."
        2. 使用 EXTRACTOR_SYSTEM_MESSAGE 和 EXTRACTOR_HUMAN_MESSAGE 构造 prompt
        3. 调用 self.client.chat_structured() 并传入 output_schema=ExtractResultFormat
        4. 返回 result.notes

        提示：
        - EXTRACTOR_HUMAN_MESSAGE 需要两个参数：passage 和 question
        - 如果结构化输出失败（某些 API 不支持），可以 fallback 到 self.client.chat()
        - 参考 src/llm_client.py 中的 chat_structured 和 chat 方法
        """
        # ====== 你的代码开始 ======
        raise NotImplementedError("请实现 _extract 方法")
        # ====== 你的代码结束 ======

    def _generate(self, question: str, notes: List[str], doc_ids: List[str]) -> QAAnswerFormat:
        """
        基于抽取的 notes 生成最终答案。

        TODO: 请实现以下逻辑：
        1. 将 notes 和 doc_ids 配对，格式化为：
           doc_{doc_id}: {note}
           doc_{doc_id}: {note}
           ...
        2. 使用 QA_SYSTEM_MESSAGE 和 QA_HUMAN_MESSAGE 构造 prompt
        3. 调用 self.client.chat_structured() 并传入 output_schema=QAAnswerFormat
        4. 返回 result

        提示：
        - 配对展示有助于 LLM 区分不同文档的信息来源
        - QA_HUMAN_MESSAGE 需要两个参数：context 和 question
        """
        # ====== 你的代码开始 ======
        raise NotImplementedError("请实现 _generate 方法")
        # ====== 你的代码结束 ======


if __name__ == "__main__":
    """
    【本地测试】
    测试 RAGAgent 的完整链路。
    预期输出：正确回答 "Who portrayed Corliss Archer...?"，
             且 Extractor 能从无关文档中过滤出 "No related information..."
    """
    from src.config import require_llm_config

    print("=" * 60)
    print("RAGAgent 本地测试")
    print("=" * 60)

    try:
        require_llm_config()
        from src.retriever import get_retriever
        retriever = get_retriever()
        rag = RAGAgent(retriever)

        question = "Who portrayed Corliss Archer in the film Kiss and Tell?"
        print(f"\n问题: {question}\n")

        result = rag.run(question)
        answer = result["final_raw_answer"]

        print(f"检索到 {len(result['documents'])} 篇文档")
        print(f"Notes: {result['notes']}")
        print(f"\n答案: {answer.answer}")
        print(f"成功: {answer.success}")
        print(f"置信度: {answer.rating}/10")
        print(f"分析: {answer.analysis}")

        if answer.answer and answer.success.lower() == "yes":
            print("\n[通过] RAGAgent 测试完成")
        else:
            print("\n[警告] 答案为空或 success=No，请检查实现")

    except NotImplementedError as e:
        print(f"\n[未完成] {e}")
        print("提示：请先实现 run、_extract 和 _generate 方法")
    except Exception as e:
        print(f"\n[错误] {type(e).__name__}: {e}")
