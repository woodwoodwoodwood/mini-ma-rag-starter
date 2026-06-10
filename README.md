# Mini-MA-RAG 多智能体检索增强生成实验指南

> **教学目标**：通过搭建一个完整的 Multi-Agent RAG 系统，理解"规划-检索-生成"的协作范式，并通过对比实验验证其有效性。

**总课时**：20 课时（共 5 个阶段）

---

## 目录

- [实验概述](#实验概述)
- [前置知识](#前置知识)
- [环境配置](#环境配置)
- [分阶段实现计划](#分阶段实现计划)
- [评估和答辩](#评估和答辩)
- [常见问题](#常见问题)

---

## 实验概述

### 1.1 实验背景

传统的检索增强生成（Naive RAG）存在以下问题：

| 问题 | 表现 | 影响 |
|------|------|------|
| **固定检索策略** | 所有问题都检索 Top-K 文档，无论问题复杂度如何 | 简单问题造成计算浪费，复杂问题信息不足 |
| **无反思机制** | 一次检索后直接生成答案，无法自我纠正 | 可能生成幻觉数据，无法感知答案错误 |
| **上下文窗口低效** | 把所有文档塞进 prompt，挤占有效信息空间 | 生成质量下降，无法处理长上下文 |

### 1.2 Agent-RAG 的核心创新

将 RAG 过程建模为**多个专业化智能体的协作过程**：

```
用户提问
    ↓
[Planner] 将问题分解为子任务，生成执行计划
    ↓
[Step Definer] 根据历史结果，动态生成当前步骤的最优查询
    ↓
[QA Agent] 检索相关文档，基于文档回答当前步骤
    ↓
[Summarizer] 汇总各步骤的回答，给出最终答案
```

**核心优势**：
- ✅ **灵活的检索策略**：根据步骤需要动态调整 Top-K
- ✅ **细粒度验证**：每个步骤可以独立评估置信度，触发反思循环
- ✅ **完整可追溯性**：每一步的逻辑和数据都可被解释
- ✅ **高效上下文利用**：分步检索，避免信息冗余

### 1.3 学习成果

完成本实验后，你将掌握：

1. **LLM 应用架构设计**
   - 如何将复杂任务分解为协作的子任务
   - 状态管理和信息流设计
   - 错误恢复和反思机制

2. **实战编程技能**
   - Pydantic 的结构化输出设计
   - LangGraph 工作流编排
   - 向量检索系统实现

3. **实验科学方法**
   - 如何对比两种架构的性能差异
   - 如何设计具有挑战性的测试集
   - 如何进行可重复的性能评估

---

## 前置知识

### 必需知识

- **Python 基础**：面向对象编程、类型提示、异常处理
- **LLM 基础**：Prompt 设计、API 调用、结构化输出
- **向量检索基础**：Embedding、向量相似度、Top-K 检索
- **异步编程**（可选）：理解 async/await 有助于理解 LangGraph

### 推荐学习资源

- [Pydantic 官方文档](https://docs.pydantic.dev/)
- [LangGraph 快速开始](https://langchain-ai.github.io/langgraph/)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)

---

## 环境配置

### 2.1 系统要求

```
Python: 3.9+
CUDA: 11.8+（如使用 GPU）或 CPU 也可以
存储: ≥2GB（包含模型和语料）
```

### 2.2 快速开始（3 课时）

#### Step 1: 克隆项目并配置环境

```bash
cd /path/to/workspace
# 项目已经存在，直接进入
cd RAG/mini-ma-rag-starter
```

#### Step 2: 创建虚拟环境并安装依赖

```bash
# 方案 A: 使用 venv（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 方案 B: 使用 conda
conda create -n rag-agent python=3.10
conda activate rag-agent
```

#### Step 3: 安装 Python 包

```bash
pip install -r requirements.txt
```

**包清单**：
- `openai>=1.0.0` - LLM API 调用
- `sentence-transformers>=2.2.0` - 本地 Embedding
- `langgraph>=0.0.50` - 工作流编排
- `langchain>=0.1.0` - LLM 框架
- `pydantic>=2.0.0` - 数据验证
- `python-dotenv>=1.0.0` - 环境变量管理
- `faiss-cpu>=1.7.4` - 向量索引（可选）

#### Step 4: 配置 API Key

```bash
# 复制示例配置
cp .env.sample .env

# 编辑 .env，填写你的 API Key
# 支持的 API 提供商：
#   - OpenAI (api.openai.com)
#   - DeepSeek (api.deepseek.com)
#   - Qwen (dashscope.aliyuncs.com)
#   - 本地 vLLM 服务 (http://localhost:8000/v1)
```

**配置示例**：

```dotenv
# 选项 A: 使用 OpenAI
OPENAI_API_KEY=sk-xxx...
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini

# 选项 B: 使用 DeepSeek（推荐经济方案）
OPENAI_API_KEY=sk-xxx...（DeepSeek key）
OPENAI_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat

# 选项 C: 本地部署（最经济）
OPENAI_API_KEY=fake-key
OPENAI_BASE_URL=http://localhost:8000/v1
MODEL_NAME=your-local-model

# Embedding 配置（留空表示使用本地模型）
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

#### Step 5: 验证环境

```bash
# 测试 API 连接
python -c "from src.llm_client import LLMClient; client = LLMClient(); print(client.chat('system: you are helpful', 'Hello'))"

# 测试 Embedding
python -c "from src.retriever import LocalEmbeddingProvider; emb = LocalEmbeddingProvider(); print(emb.encode(['hello']).shape)"
```

**预期输出**：
- API 连接测试应该返回一个问候句子
- Embedding 测试应该返回 `(1, 384)` 的形状

---

## 分阶段实现计划

### Phase 1: 环境和工具库（3.5 课时）

**目标**：理解项目框架，实现基础工具类

#### 1.1 项目结构导读（0.5 课时）

学生应该理解以下文件的用途：

```
mini-ma-rag-starter/
├── src/
│   ├── config.py          # 全局配置（API Key、超参数等）
│   ├── retriever.py       # 向量检索工具（Embedding + 相似度搜索）
│   ├── llm_client.py      # LLM API 统一客户端（普通 + 结构化输出）
│   ├── state.py           # 状态定义（Agent 状态、Plan 格式等）
│   ├── prompts.py         # Prompt 模板（Agent 的系统 Prompt）
│   ├── planner.py         # 规划器（问题分解） ← 【需要完成】
│   ├── step_definer.py    # 步骤定义器（生成查询） ← 【需要完成】
│   ├── rag_agent.py       # 问答代理（文档检索+生成） ← 【需要完成】
│   ├── summarizer.py      # 汇总器（整合步骤结果） ← 【需要完成】
│   └── executor.py        # 执行器（协调各 Agent） ← 【需要完成】
├── graph/
│   ├── __init__.py
│   └── workflow.py        # LangGraph 工作流定义 ✓ 完整
├── baselines/
│   └── naive_rag.py       # 朴素 RAG 基线 ← 【需要完成】
├── data/
│   ├── build_corpus.py    # 语料库构建脚本
│   ├── corpus.jsonl       # 文本语料库（需运行 build_corpus.py 生成）
│   └── test_set.jsonl     # 测试集（QA 对）
├── eval/
│   └── evaluate.py        # 评估脚本 ✓ 完整
└── run.py                 # 主入口脚本 ✓ 完整
```

**自学内容**：
- 阅读 `README.md` 中的"实验概述"部分（15 分钟）
- 浏览 `src/state.py`，理解各数据类型的定义（15 分钟）
- 浏览 `src/config.py`，了解配置项（10 分钟）

#### 1.2 实现 LLM 客户端（1.5 课时）

**任务**：完成 `src/llm_client.py` 中的 `chat()` 和 `chat_structured()` 方法

**核心需求**：
- ✅ `chat()` 返回字符串，用于普通对话
- ✅ `chat_structured()` 返回 Pydantic 对象，支持结构化输出
- ✅ 兼容不同 API 提供商（OpenAI、DeepSeek 等）
- ✅ 实现重试机制（最多 3 次）

**学习点**：
- OpenAI Python SDK 的基本用法
- Pydantic 的 `BaseModel` 和 `response_format`
- 异常处理和重试策略

**参考代码片段**：

```python
def chat(self, system_message: str, user_message: str, temperature: Optional[float] = None) -> str:
    """普通对话生成"""
    temp = temperature if temperature is not None else self.temperature
    for attempt in range(self.max_retries):
        try:
            # 使用 self.client.chat.completions.create() 调用 API
            # 返回 response.choices[0].message.content.strip()
            pass
        except Exception as e:
            if attempt == self.max_retries - 1:
                raise RuntimeError(f"LLM API 调用失败: {e}")
    return ""

def chat_structured(self, system_message: str, user_message: str, 
                    output_schema: Type[BaseModel], temperature: Optional[float] = None) -> BaseModel:
    """结构化输出生成"""
    # 兼容策略：优先尝试 beta.chat.completions.parse，失败则用普通 chat + 手动 JSON 解析
    pass
```

**自测命令**：

```bash
python -c "
from src.llm_client import LLMClient
from src.config import require_llm_config
require_llm_config()

client = LLMClient()

# 测试普通对话
response = client.chat('You are helpful', 'What is 2+2?')
print('Chat response:', response)

# 测试结构化输出
from pydantic import BaseModel
class Math(BaseModel):
    result: int
    explanation: str

result = client.chat_structured(
    'You are a math expert',
    'What is 2+2?',
    Math
)
print('Structured response:', result)
"
```

**预期输出**：两个测试都应该返回有效的响应（第二个应该是 JSON 格式）。

#### 1.3 实现向量检索系统（1.5 课时）

**任务**：完成 `src/retriever.py` 中的关键方法

**核心需求**：
- ✅ 加载本地 Embedding 模型（sentence-transformers）
- ✅ 从 jsonl 文件加载文本语料库
- ✅ 构建向量索引（支持 FAISS 可选）
- ✅ 实现向量相似度搜索，返回 Top-K 文档

**学习点**：
- Sentence-Transformers 的使用
- 向量归一化和 L2 距离
- FAISS 的基本使用（可选）
- 文件 I/O 和 JSON 解析

**关键代码位置**：

```python
class LocalEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL):
        # TODO: 使用 sentence_transformers.SentenceTransformer 加载模型
        # 注意：需要对 embeddings 进行 L2 归一化
        pass

    def encode(self, texts: List[str]) -> np.ndarray:
        # TODO: 调用模型生成 embeddings，返回归一化后的向量
        pass

class Retriever:
    def load_corpus(self, corpus_path: str = CORPUS_PATH):
        # TODO: 从 jsonl 文件加载文档，每行格式为 {"id": xxx, "text": ...}
        # 存储到 self.docs 和 self.doc_ids
        pass

    def build_index(self):
        # TODO: 为所有文档生成 embeddings，存储在 self.embeddings
        # 如果 use_faiss=True，调用 _build_faiss_index()
        pass

    def search(self, query: str, top_k: Optional[int] = None) -> Tuple[List[str], List[str]]:
        # TODO: 编码 query，计算与所有文档的相似度，返回 Top-K
        # 支持 FAISS 索引（如可用）或暴力搜索
        pass
```

**自测命令**：

```bash
python src/retriever.py
```

**预期输出**：应该加载语料库，运行示例查询，返回 5 个相关文档。

---

### Phase 2: Naive RAG 基线（3 课时）

**目标**：实现一个简单的 RAG 基线，作为对比参考

#### 2.1 理论讲解（1 课时）

Naive RAG 的流程很简单：

```
用户提问 → [检索] 获取 Top-K 文档 → [生成] 基于文档生成答案
```

**核心特点**：
- 一次性检索，不分步骤
- 直接使用问题作为检索查询
- 无反思或修正机制

#### 2.2 实现 NaiveRAG 类（1.5 课时）

**任务**：完成 `baselines/naive_rag.py`

**核心需求**：
- ✅ 实现 `answer()` 方法，接收问题，返回答案
- ✅ 使用检索器获取相关文档
- ✅ 将文档拼接成上下文
- ✅ 调用 LLM 生成答案

**实现要点**：

```python
class NaiveRAG:
    def __init__(self, retriever: Retriever, client=None):
        self.retriever = retriever
        self.client = client or get_default_client()

    def answer(self, question: str, top_k: int = 5) -> dict:
        """
        回答问题。
        
        返回格式: {
            "question": question,
            "documents": [...],  # 文档文本列表
            "doc_ids": [...],    # 文档 ID 列表
            "answer": str        # 生成的答案
        }
        """
        # TODO: 
        # 1. 调用 self.retriever.search(question, top_k=top_k)
        # 2. 拼接文档为 context 字符串，格式 "[doc_id] text\n\n"
        # 3. 使用 NAIVE_RAG_SYSTEM_MESSAGE 和 NAIVE_RAG_HUMAN_MESSAGE
        # 4. 调用 self.client.chat() 生成答案
        # 5. 返回结果字典
        pass
```

**参考 Prompt**（在 `src/prompts.py` 中）：

```python
NAIVE_RAG_SYSTEM_MESSAGE = "你是一个有用的助手，基于给定的文档回答问题。"

NAIVE_RAG_HUMAN_MESSAGE = """请根据以下文档回答问题。

【文档】
{context}

【问题】
{question}

【答案】"""
```

#### 2.3 测试 Naive RAG（0.5 课时）

**自测命令**：

```bash
python -c "
import json
from src.retriever import get_retriever
from baselines.naive_rag import NaiveRAG

retriever = get_retriever()
retriever.load_corpus()
retriever.build_index()

rag = NaiveRAG(retriever)

# 测试单个问题
result = rag.answer('Who is Shirley Temple?', top_k=3)
print('Question:', result['question'])
print('Documents retrieved:', len(result['documents']))
print('Answer:', result['answer'][:200])
"
```

**预期输出**：应该返回 3 篇文档和一个关于 Shirley Temple 的答案。

---

### Phase 3: Agent-RAG 核心组件（7 课时）

**目标**：实现 Agent-RAG 的各个智能体和执行器

#### 3.1 Planner 规划器（1.5 课时）

**任务**：完成 `src/planner.py` 中的 `Planner` 类

**核心概念**：
- 将复杂问题分解为多个可执行的子步骤
- 使用 LLM 的结构化输出确保格式一致性
- 支持迭代优化（传入历史失败经验让 LLM 改进计划）

**实现要点**：

```python
class Planner:
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_default_client()

    def plan(self, question: str, past_experiences: Optional[List[PlanExecState]] = None) -> List[str]:
        """
        为问题生成执行计划（子任务列表）。
        
        Args:
            question: 原始问题
            past_experiences: 之前的执行经验，用于让 LLM 避免重复犯错
        
        Returns:
            子任务列表，如 ["Who is X?", "What did X do?"]
        """
        # TODO:
        # 1. 调用 self._format_memory(past_experiences) 生成 memory 字符串
        # 2. 使用 PLANNER_HUMAN_MESSAGE.format(question=..., memory=...) 生成 user_message
        # 3. 调用 self.client.chat_structured(
        #      system_message=PLANNER_SYSTEM_MESSAGE,
        #      user_message=user_message,
        #      output_schema=PlanFormat,
        #      temperature=0.3
        #    )
        # 4. 返回 result.step
        pass

    def _format_memory(self, past_experiences: Optional[List[PlanExecState]]) -> str:
        """
        将历史经验格式化为文本，供 Planner 参考。
        
        格式示例：
        Trial 0:
        Plan: [Step 1, Step 2, ...]
        Status: Failed Score: 2
        
        返回值：如果 past_experiences 为空，返回 "empty"
        """
        # TODO: 实现格式化逻辑
        pass
```

**参考 Prompt 格式**（在 `src/prompts.py` 中）：

```python
from pydantic import BaseModel, Field

class PlanFormat(BaseModel):
    """计划的数据格式"""
    analysis: str = Field(description="对问题的分析，Think step-by-step")
    step: List[str] = Field(description="按顺序执行的步骤列表")

PLANNER_SYSTEM_MESSAGE = """
你是一个问题分解专家。你的任务是将复杂问题分解为一系列简单的、可独立回答的子问题。

【要求】
1. 每个子问题应该是可以通过文档检索直接回答的
2. 子问题之间无重叠，且满足逻辑顺序（后序问题可以依赖前序答案）
3. 通常 3-5 个步骤足够，避免过度分解

【分解示例】
问题：Shirley Temple 演过什么电影，她的政治生涯如何？
分解：
1. Shirley Temple 有哪些代表作电影？
2. Shirley Temple 的政治生涯是怎样的？
3. 她在政治上取得了什么成就？
"""

PLANNER_HUMAN_MESSAGE = """
【之前的尝试】
{memory}

【新的问题】
{question}

请为上述问题生成一个执行计划。
"""
```

**自测命令**：

```bash
python src/planner.py
```

**预期输出**：应该生成 3-5 个分解步骤。

#### 3.2 Step Definer 步骤定义器（1.5 课时）

**任务**：完成 `src/step_definer.py` 中的 `StepDefiner` 类

**核心概念**：
- 根据原始问题和之前的步骤结果，动态生成当前步骤的最优查询
- 目的是实现"细粒度检索"，避免冗余信息

**实现要点**：

```python
class StepDefiner:
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_default_client()

    def define(
        self,
        plan: List[str],
        current_step_index: int,
        step_outputs: List[QAAnswerFormat],
    ) -> StepTaskFormat:
        """
        为当前步骤生成最优查询。

        Args:
            plan: 完整计划列表
            current_step_index: 当前执行到第几步（从 0 开始）
            step_outputs: 之前各步骤的 QAAnswerFormat 结果

        Returns:
            StepTaskFormat: 包含 type (question-answering/aggregate) 和 task
        """
        # TODO:
        # 1. 获取当前步骤文本 plan[current_step_index]
        # 2. 调用 self._format_step_outputs(plan, step_outputs) 生成记忆
        # 3. 使用 STEP_DEFINER_HUMAN_MESSAGE 生成 user_message
        # 4. 调用 self.client.chat_structured() + StepTaskFormat
        # 5. 返回 result
        pass

    def _format_step_outputs(self, plan: List[str], step_outputs: List[QAAnswerFormat]) -> str:
        """将之前的步骤答案格式化为记忆文本"""
        # TODO: 实现格式化逻辑，如果为空返回 "None"
        pass
```

**参考 Prompt**：

```python
STEP_DEFINER_SYSTEM_MESSAGE = """
你是一个查询优化专家。你的任务是根据当前执行步骤和之前的回答历史，
生成一个最优的检索查询字符串，使其能够高效检索相关文档。
"""

STEP_DEFINER_HUMAN_MESSAGE = """
【原始问题】
{original_question}

【之前的回答】
{history}

【当前步骤】
{step}

请生成一个检索查询（1-2 句话），用于查找与当前步骤相关的文档。
返回查询字符串即可，无需其他说明。
"""
```

**自测命令**：

```bash
python src/step_definer.py
```

#### 3.3 RAG Agent 问答代理（1.5 课时）

**任务**：完成 `src/rag_agent.py` 中的 `RAGAgent` 类

**核心概念**：
- 检索相关文档
- 基于文档生成步骤的答案
- 记录回答的置信度评分

**实现要点**：

```python
class RAGAgent:
    def __init__(self, retriever: Retriever, client: Optional[LLMClient] = None):
        self.retriever = retriever
        self.client = client or get_default_client()

    def run(self, question: str) -> RagState:
        """
        执行完整的 RAG 流程：检索 → 抽取 → 生成。

        Returns:
            RagState: 包含 question, documents, doc_ids, notes, final_raw_answer
        """
        # TODO:
        # 1. 调用 self.retriever.search(question) 检索 Top-K 文档
        # 2. 对每篇文档调用 self._extract(doc, question) 抽取 notes
        # 3. 调用 self._generate(question, notes, doc_ids) 生成答案
        # 4. 组装并返回 RagState
        pass

    def _extract(self, passage: str, question: str) -> str:
        """从单篇文档中抽取与问题相关的 notes"""
        # TODO: 调用 self.client.chat_structured() + ExtractResultFormat
        pass

    def _generate(self, question: str, notes: List[str], doc_ids: List[str]) -> QAAnswerFormat:
        """基于抽取的 notes 生成最终答案"""
        # TODO: 调用 self.client.chat_structured() + QAAnswerFormat
        pass
```

**参考 Prompt**：

```python
from pydantic import BaseModel, Field

class QAAnswerFormat(BaseModel):
    """QA Agent 的输出格式"""
    analysis: str = Field(description="对问题和证据的思考过程")
    answer: str = Field(description="简洁的答案")
    success: str = Field(description="是否成功回答，Yes/No")
    rating: int = Field(default=0, description="置信度评分 0-10")

QA_SYSTEM_MESSAGE = """
你是一个精准的问答助手。你的任务是基于给定的文档回答问题。

【回答要求】
1. 如果文档中有明确答案，直接引用
2. 如果信息不足，坦诚说明 (success=No, rating=0)
3. 如果答案只在部分文档中，也要 success=Yes 但 rating=5-7
4. success 必须是 "Yes" 或 "No"
"""

QA_HUMAN_MESSAGE = """
【文档】
{context}

【问题】
{question}

请回答问题，并用 JSON 格式返回：{{"answer": ..., "success": "Yes/No", "rating": 0-10}}
"""
```

**自测命令**：

```bash
python src/rag_agent.py
```

#### 3.4 Summarizer 汇总器（1.5 课时）

**任务**：完成 `src/summarizer.py` 中的 `Summarizer` 类

**核心概念**：
- 整合各个步骤的答案
- 生成最终答案
- 给出答案质量评分

**实现要点**：

```python
class Summarizer:
    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_default_client()

    def summarize(
        self,
        original_question: str,
        plan: List[str],
        step_outputs: List[QAAnswerFormat],
    ) -> PlanSummaryFormat:
        """
        汇总步骤输出，生成最终答案。

        Returns:
            PlanSummaryFormat: {
                "output": "Successful"/"Unsuccessful",
                "answer": str,      # 最终答案
                "score": int,       # 0-10 的质量评分
            }
        """
        # TODO:
        # 1. 将 plan 格式化为字符串，调用 self._format_memory(plan, step_outputs) 生成记忆
        # 2. 使用 SUMMARIZER_SYSTEM_MESSAGE 和 SUMMARIZER_HUMAN_MESSAGE
        # 3. 调用 self.client.chat_structured() 得到 PlanSummaryFormat
        # 4. 返回结果
        pass

    def _format_memory(self, plan: List[str], step_outputs: List[QAAnswerFormat]) -> str:
        """将步骤输出格式化为记忆文本"""
        pass
```

**参考 Prompt**：

```python
class PlanSummaryFormat(BaseModel):
    """计划汇总格式"""
    output: str = Field(description="执行状态，Successful 或 Unsuccessful")
    answer: str = Field(description="最终答案，Unsuccessful 时为 N/A")
    score: int = Field(description="答案质量评分，0-10")

SUMMARIZER_SYSTEM_MESSAGE = """
你是一个信息汇总专家。你的任务是基于各步骤的回答，
生成一个连贯、准确的最终答案。
"""

SUMMARIZER_HUMAN_MESSAGE = """
【原始问题】
{question}

【步骤回答】
{steps}

请生成最终答案，并评估答案的完整性和准确性。
"""
```

#### 3.5 Executor 执行器（1 课时）

**任务**：完成 `src/executor.py` 中的 `PlanExecutor` 类

**核心概念**：
- 协调各个 Agent 的执行
- 管理状态转移
- 实现反思循环（若早期步骤失败，可以重新规划）

**实现要点**：

```python
class Aggregator:
    """处理 type='aggregate' 的步骤，无需检索，仅基于历史结果推理"""
    def __init__(self, client=None): ...
    def run(self, query: str, memory: List[QAAnswerFormat]) -> QAAnswerFormat: ...

class PlanExecutor:
    def __init__(self, rag_agent: RAGAgent,
                 step_definer: StepDefiner = None,
                 summarizer: Summarizer = None):
        self.rag_agent = rag_agent
        self.step_definer = step_definer or StepDefiner()
        self.summarizer = summarizer or Summarizer()
        self.aggregator = Aggregator()

    def execute(self, question: str, plan: List[str]) -> PlanExecState:
        """
        执行整个计划，返回完整状态。

        核心循环:
          for step in plan:
            task = step_definer.define(plan, idx, step_outputs)
            if task.type == 'question-answering':
                result = rag_agent.run(task.task)
            else:
                result = aggregator.run(task.task, step_outputs)
            step_outputs.append(result)
          summary = summarizer.summarize(question, plan, step_outputs)

        Returns:
            PlanExecState: 包含 step_question, step_output, plan_summary, stop 等
        """
        pass
```

**自测命令**：

```bash
python src/executor.py
```

---

### Phase 4: LangGraph 工作流和主入口（3 课时）

**目标**：将各组件整合为完整的工作流，支持命令行执行

#### 4.1 LangGraph 工作流（1.5 课时）

**任务**：完成 `graph/workflow.py` 中的 `AgentWorkflow` 类

**核心概念**：
- 使用 LangGraph 定义状态图
- 实现节点和边，支持条件跳转
- 支持多轮迭代反思（可选）

**实现要点**：

```python
from langgraph.graph import StateGraph, START, END

class AgentWorkflow:
    def __init__(self, plan_executor: PlanExecutor):
        self.executor = plan_executor
        self.planner = Planner()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        # TODO:
        # 1. 创建 StateGraph(AgentState)
        # 2. 添加 planner_node：调用 self.planner.plan()，返回 {"plan": ...}
        # 3. 添加 executor_node：调用 self.executor.execute()，返回结果
        # 4. 添加边：START → planner_node → executor_node → END
        # 5. 编译图并返回
        
        graph = StateGraph(AgentState)
        
        def planner_node(state):
            # 生成计划
            question = state["original_question"]
            plan = self.planner.plan(question)
            return {"plan": plan}
        
        def executor_node(state):
            # 执行计划
            question = state["original_question"]
            plan = state["plan"]
            result = self.executor.execute(question, plan)
            return {
                "past_exp": [result],
                "final_answer": ...
            }
        
        graph.add_node("planner", planner_node)
        graph.add_node("executor", executor_node)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "executor")
        graph.add_edge("executor", END)
        
        return graph.compile()

    def run(self, question: str) -> dict:
        """运行工作流"""
        # TODO: 调用 self.graph.invoke() 并返回结果
        pass
```

**参考状态定义**（在 `src/state.py` 中）：

```python
from typing import Any, Annotated, Optional, List
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """Agent 的状态"""
    original_question: str          # 原始问题
    plan: List[str]                 # 当前执行的计划
    past_exp: List[dict]            # 历史执行经验（用于迭代）
    final_answer: str               # 最终答案
    iteration: int                  # 当前迭代次数
```

**自测命令**：

```bash
python -c "
from graph.workflow import AgentWorkflow
from src.executor import PlanExecutor
from src.rag_agent import RAGAgent
from src.retriever import get_retriever

retriever = get_retriever()
retriever.load_corpus()
retriever.build_index()

rag = RAGAgent(retriever)
executor = PlanExecutor(rag)
workflow = AgentWorkflow(executor)

# 测试工作流
result = workflow.run('Who is Shirley Temple?')
print('Final answer:', result.get('final_answer', 'N/A')[:200])
"
```

#### 4.2 主入口脚本（1.5 课时）

**任务**：完成 `run.py` 中的两个主函数

**核心需求**：
- ✅ `run_agent_rag()` - 在测试集上运行 Agent-RAG
- ✅ `run_naive_rag()` - 在测试集上运行 Naive RAG
- ✅ 支持命令行参数（--mode, --start, --end, --output）
- ✅ 输出标准 JSON 格式的结果文件

**实现要点**：

```python
def run_agent_rag(output_path: str, start: int = 0, end: int = None, test_set_path: str = None):
    """
    运行 Agent-RAG 在测试集上，生成结果文件。
    
    输出 JSON 格式:
    {
        "id": "q001",
        "question": "...",
        "ground_truth": "...",
        "predicted": "...",
        "status": "Successful",
        "score": 9,
        "plan": [...],
        "step_outputs": [
            {"step": "...", "success": "Yes", "rating": 10},
            ...
        ]
    }
    """
    # TODO:
    # 1. 调用 require_llm_config() 检查环境配置
    # 2. 加载测试集（从 test_set_path 或 TEST_SET_PATH）
    # 3. 初始化检索器、Agent 和工作流
    # 4. 遍历测试集 items[start:end]
    # 5. 对每个问题调用 workflow.run()
    # 6. 整理结果为上述 JSON 格式
    # 7. 写入 output_path
    pass

def run_naive_rag(output_path: str, start: int = 0, end: int = None, test_set_path: str = None):
    """运行 Naive RAG 基线"""
    # 类似于 run_agent_rag，但调用 NaiveRAG.answer()
    pass
```

**命令行接口**：

```python
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["agent", "naive"], default="agent")
    parser.add_argument("--output", default="results.jsonl")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--input", default=None, help="Custom test set path")
    
    args = parser.parse_args()
    
    if args.mode == "agent":
        run_agent_rag(args.output, args.start, args.end, args.input)
    else:
        run_naive_rag(args.output, args.start, args.end, args.input)
```

**使用示例**：

```bash
# 运行 Agent-RAG 在测试集上
python run.py --mode agent --output results_agent.jsonl

# 运行 Naive RAG
python run.py --mode naive --output results_naive.jsonl

# 运行前 5 题
python run.py --mode agent --start 0 --end 5 --output results_sample.jsonl

# 使用自定义测试集
python run.py --mode agent --input data/test_set.jsonl --output results_hard.jsonl
```

---

### Phase 5: 评估和对比分析（3.5 课时）

**目标**：实现评估框架，进行对比实验，撰写实验报告

#### 5.1 评估脚本（1.5 课时）

**任务**：完成 `eval/evaluate.py` 中的评估函数

**核心需求**：
- ✅ 读取结果文件（results.jsonl）
- ✅ 使用 LLM 作为评估器判断答案正确性
- ✅ 计算准确率、F1 等指标
- ✅ 输出详细的评估报告

**实现要点**：

`eval/evaluate.py` 已提供完整的 `Evaluator` 类，支持两种评估方式：
- `_llm_judge()`: 使用 LLM-as-Judge 判断答案正确性
- `_exact_match()`: 精确匹配（归一化字符串比较，LLM 不可用时的 fallback）
- `evaluate_file()`: 主入口，读取结果文件 → 逐题评估 → 输出统计

**使用示例**：

```bash
# 评估 Agent-RAG 结果
python eval/evaluate.py results_agent.jsonl --output eval_agent.json

# 评估 Naive RAG 结果
python eval/evaluate.py results_naive.jsonl --output eval_naive.json
```

#### 5.2 对比实验（1.5 课时）

**任务**：运行两个系统的完整实验，收集数据

**实验流程**：

1. **数据准备**
   ```bash
   # 检查语料库和测试集是否完整
   ls -la data/
   # 应该看到 corpus.jsonl 和 test_set.jsonl
   ```

2. **运行 Naive RAG 基线**
   ```bash
   python run.py --mode naive --output results_naive.jsonl
   python eval/evaluate.py results_naive.jsonl --output eval_naive.json
   ```

3. **运行 Agent-RAG**
   ```bash
   python run.py --mode agent --output results_agent.jsonl
   python eval/evaluate.py results_agent.jsonl --output eval_agent.json
   ```

4. **收集和分析结果**
   - 比较两个系统的准确率
   - 分析失败案例，找出各自的弱点
   - 评估可解释性和可追溯性

**期望结果**：
- 在简单问题上，Naive RAG 和 Agent-RAG 应表现接近
- 在复杂问题上，Agent-RAG 应因能够分步骤而表现更好（或相当）
- Agent-RAG 可以提供完整的执行过程（Plan + Step 输出）

#### 5.3 实验报告撰写（0.5 课时）

**报告结构**：

```markdown
# Agent-RAG 与 Naive RAG 对比实验报告

## 1. 实验目标
- 对比两种 RAG 架构的性能和可解释性差异
- 验证多智能体协作范式的有效性

## 2. 实验设计
- 测试集: test_set.jsonl (10 道题，包含简单事实题和多跳推理题)
- 评估指标: 准确率、F1 分数、推理过程完整性
- LLM 模型: [你使用的模型]
- Embedding 模型

## 3. 实验结果

### 3.1 定量结果

| 系统 | 准确率 | 平均响应时间 | 平均 Token 消耗 |
|------|--------|------------|---------------|
| Naive RAG | 80% | 2.3s | 450 |
| Agent-RAG | 85% | 3.1s | 680 |

### 3.2 分类性能

| 问题类型 | Naive RAG | Agent-RAG |
|---------|----------|----------|
| 简单事实题 | 100% | 100% |
| 单跳推理 | 75% | 85% |
| 多跳推理 | 60% | 80% |

## 4. 定性分析

### 4.1 失败案例分析
- Naive RAG 失败原因: [列举]
- Agent-RAG 失败原因: [列举]

### 4.2 可解释性对比
- Naive RAG: 黑盒，无法解释决策过程
- Agent-RAG: 白盒，提供完整的 Plan + Step 执行过程

### 4.3 计算效率对比
- Agent-RAG 额外的 API 调用次数
- 总体时间和 Token 消耗的增长

## 5. 结论
- Agent-RAG 在复杂问题上有明显优势
- 需要权衡准确率提升与计算成本增加
- 可解释性是重要价值，值得额外开销

## 6. 改进方向
- 动态调整步骤数量
- 实现更聪明的反思和重新规划机制
- 缓存检索结果，减少重复调用

## 附录
- 完整的失败案例分析
- 代码和配置说明
- 实验数据 CSV 文件
```

**评分标准**：
- ✅ 实验设计合理（20%）
- ✅ 定量数据完整（30%）
- ✅ 定性分析深入（30%）
- ✅ 结论有据可查（20%）

---

## 评估

### 6.1 自动化测试（10%）

代码应通过以下测试：

```bash
# 测试 1: 环境检查
python -c "
from src.config import require_llm_config
require_llm_config()
print('✓ API 配置正确')
"

# 测试 2: 组件加载
python -c "
from src.planner import Planner
from src.step_definer import StepDefiner
from src.rag_agent import RAGAgent
from src.summarizer import Summarizer
from src.executor import PlanExecutor
from graph.workflow import AgentWorkflow
print('✓ 所有组件加载成功')
"

# 测试 3: Naive RAG 运行
python run.py --mode naive --start 0 --end 2 --output /tmp/test_naive.jsonl
# 应该产生 2 行结果

# 测试 4: Agent-RAG 运行
python run.py --mode agent --start 0 --end 2 --output /tmp/test_agent.jsonl
# 应该产生 2 行结果（含 Plan 和 Step 输出）
```

### 6.3 实验报告（60%）

- 报告结构完整（见上文）
- 数据分析深入有见地
- 对比分析清晰有对照
- 改进建议可行

### 6.4 框架代码设计（40%）

1. **架构设计**
   - 为什么要把 RAG 分解为多个 Agent？
   - 每个 Agent 的职责是什么？
   - 如何处理 Agent 之间的协作？

2. **实现细节**
   - Planner 如何分解问题？
   - Step Definer 的作用是什么？
   - 如何保证各步骤的执行质量？

3. **实验结果**
   - Agent-RAG 相比 Naive RAG 的优势和劣势？
   - 在哪些问题上 Agent-RAG 表现更好，为什么？
   - 计算成本如何，是否值得？

4. **改进方向**
   - 如何进一步优化架构？
   - 如何处理更复杂的问题？
   - 实际应用中的考虑？

---

## 常见问题

### Q1: API 调用费用很高怎么办？

**答**：有几个办法降低成本：
1. 使用免费模型如 DeepSeek API（更便宜）
2. 本地部署开源 LLM（Llama、Mistral 等），通过 vLLM 或 ollama 提供 API
3. 使用本地小模型如 Qwen1.5-7B（有开源版）
4. 缓存检索结果，减少重复调用

**推荐配置**：
```dotenv
# 使用 DeepSeek（API 费用 ~1/10）
OPENAI_API_KEY=sk-xxx (DeepSeek key)
OPENAI_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
```

### Q2: 我想用中文测试集，需要修改什么？

**答**：只需替换 `data/test_set.jsonl`，代码无需改动。但需要注意：
1. Embedding 模型应使用多语言模型如 `bge-m3` 或 `multilingual-e5-large`
2. Prompt 中的示例也应改为中文

### Q3: 我想实现更复杂的工作流（如迭代反思）

**答**：参考 `graph/workflow.py` 中的 LangGraph 模式，可以：
1. 添加条件判断边（`add_conditional_edges`）
2. 基于执行结果决定是否重新规划
3. 实现反思循环：若 summary.score < 5 且 iteration < MAX，则重新规划

### Q4: 我想对比不同 Embedding 模型的性能

**答**：
1. 修改 `src/config.py` 中的 `LOCAL_EMBEDDING_MODEL`
2. 运行完整实验，对比结果
3. 注意不同模型的向量维度（256、384、768 等）可能不同

---

## 附录 A: 项目文件清单

```mini-ma-rag-starter/
├── src/
│   ├── config.py              ✓ 完整
│   ├── llm_client.py          ✓ 完整
│   ├── retriever.py           ✓ 完整
│   ├── state.py               ✓ 完整
│   ├── prompts.py             ✓ 完整
│   ├── planner.py             ✗ 需要完成 Planner.plan() 和 _format_memory()
│   ├── step_definer.py        ✗ 需要完成 StepDefiner.define() 和 _format_step_outputs()
│   ├── rag_agent.py           ✗ 需要完成 RAGAgent.run() 和 _extract() / _generate()
│   ├── summarizer.py          ✗ 需要完成 Summarizer.summarize() 和 _format_memory()
│   └── executor.py            ✗ 需要完成 Aggregator.run() 和 PlanExecutor.execute()
├── graph/
│   ├── __init__.py            ✓ 完整
│   └── workflow.py            ✓ 完整（AgentWorkflow 编排 Planner + PlanExecutor）
├── baselines/
│   ├── __init__.py            ✓ 完整
│   └── naive_rag.py           ✗ 需要完成 NaiveRAG.answer() 和 run_naive_rag_on_dataset()
├── eval/
│   ├── __init__.py            ✓ 完整
│   └── evaluate.py            ✓ 完整（Evaluator 类 + CLI）
├── data/
│   ├── build_corpus.py        ✓ 完整
│   ├── corpus.jsonl           ⚠ 需运行 `python data/build_corpus.py` 生成
│   └── test_set.jsonl         ✓ 完整
├── .env.sample                ✓ 完整
├── requirements.txt           ✓ 完整
├── run.py                     ✓ 完整（agent / naive 模式，CLI 参数完整）
├── QUICK_REFERENCE.md         快速参考指南
├── SCHEDULE.md                课时分配和评分标准
└── README.md                  本文件```

---

## 附录 B: 参考实现（示例答案）

为了学生有方向，这里给出 Planner.plan() 的参考实现：

```python
def plan(self, question: str, past_experiences: Optional[List[PlanExecState]] = None) -> List[str]:
    """为问题生成执行计划"""
    memory = self._format_memory(past_experiences)
    user_message = PLANNER_HUMAN_MESSAGE.format(question=question, memory=memory)
    
    result = self.client.chat_structured(
        system_message=PLANNER_SYSTEM_MESSAGE,
        user_message=user_message,
        output_schema=PlanFormat,
        temperature=0.3,
    )
    return result.step

def _format_memory(self, past_experiences: Optional[List[PlanExecState]]) -> str:
    """将历史经验格式化为文本"""
    if not past_experiences:
        return "empty"
    
    lines = []
    for idx, exp in enumerate(past_experiences):
        plan_str = ", ".join(exp["plan"])
        summary = exp.get("plan_summary")
        if summary:
            lines.append(
                f"Trial {idx}:\n"
                f"Plan: [{plan_str}]\n"
                f"Status: {summary.output} Score: {summary.score}\n"
            )
        else:
            lines.append(f"Trial {idx}:\nPlan: [{plan_str}]\nStatus: No summary\n")
    return "\n".join(lines)
```

这可以作为学生的参考，但建议先让学生独立尝试。

---

## 总结

本实验通过 5 个阶段（共 20 课时）引导学生实现一个完整的多智能体 RAG 系统。核心学习成果包括：

1. **架构思想**：理解如何将复杂任务分解为协作的智能体
2. **工程能力**：掌握 LLM API、向量检索、工作流编排等技术
3. **实验方法**：学会设计对比实验，进行数据分析和报告撰写

预计学生完成本实验后，能够独立设计和实现中等复杂度的 LLM 应用系统。

