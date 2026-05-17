"""
Mini-MA-RAG: 全量状态定义模块

本模块使用 TypedDict 和 Pydantic BaseModel 定义系统中流转的所有数据结构。
LangGraph 的状态传播依赖这些类型定义。
"""

from typing import List, Annotated, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator


# ===================== Pydantic 结构化输出格式 =====================

class PlanFormat(BaseModel):
    """Planner 的结构化输出格式"""
    analysis: str = Field(description="对问题的分析，Think step-by-step")
    step: List[str] = Field(description="分解后的子任务列表，按执行顺序排列")


class StepTaskFormat(BaseModel):
    """Step Definer 的结构化输出格式"""
    type: str = Field(description="任务类型，必须是 'question-answering' 或 'aggregate' 之一")
    task: str = Field(description="当前步骤的具体执行任务，需包含之前步骤的所有关键信息")


class ExtractResultFormat(BaseModel):
    """Extractor 的结构化输出格式"""
    notes: str = Field(description="从文档中抽取的与问题相关的要点摘要，若无相关信息则输出 'No related information from this document.'")


class QAAnswerFormat(BaseModel):
    """QA Agent 的结构化输出格式"""
    analysis: str = Field(description="对问题和证据的思考过程")
    answer: str = Field(description="简洁的答案")
    success: str = Field(description="是否能回答问题，只能是 'Yes' 或 'No'")
    rating: int = Field(default=0, description="置信度评分，0-10，证据越充分分值越高")


class PlanSummaryFormat(BaseModel):
    """Summarizer 的结构化输出格式"""
    output: str = Field(description="计划执行结果：'Successful' 或 'Unsuccessful'")
    answer: str = Field(description="原始问题的最终答案。如果 Unsuccessful，填 'N/A'")
    score: int = Field(description="最终置信度评分 0-10。Unsuccessful 时为 0")


# ===================== TypedDict 图状态 =====================

class RagState(TypedDict):
    """RAG Agent 子图的状态"""
    question: str                    # 当前子问题
    documents: List[str]             # 检索到的原始文档列表
    doc_ids: List[str]               # 文档 ID 列表
    notes: List[str]                 # 从文档中抽取的 notes
    final_raw_answer: Optional[QAAnswerFormat]  # QA Agent 生成的答案


class PlanExecState(TypedDict):
    """Plan Executor 的状态：执行一个 Plan 的全过程"""
    original_question: str
    plan: List[str]                                  # 当前 Plan 的步骤列表
    step_question: Annotated[List[StepTaskFormat], operator.add]   # 每步的 Step Definer 输出
    step_output: Annotated[List[QAAnswerFormat], operator.add]     # 每步的执行结果
    step_docs_ids: Annotated[List[List[str]], operator.add]        # 每步检索到的文档 ID
    step_notes: Annotated[List[List[str]], operator.add]           # 每步的 notes
    plan_summary: Optional[PlanSummaryFormat]        # 最终汇总结果
    stop: bool = False                               # 是否终止执行


class AgentState(TypedDict):
    """顶层 LangGraph 的状态"""
    original_question: str
    plan: List[str]
    past_exp: Annotated[List[PlanExecState], operator.add]   # 历史执行经验（用于迭代优化）
    final_answer: str
    iteration: int = 0                                       # 当前迭代轮数
