# Mini-MA-RAG 实验课时分配表

## 概览

| 阶段 | 主题 | 课时 | 难度 | 核心目标 |
|------|------|------|------|---------|
| Phase 1 | 环境配置与基础工具 | 3 | ⭐ | 配置开发环境，实现 LLM 客户端和向量检索 |
| Phase 2 | Naive RAG 基线 | 2.5 | ⭐ | 实现简单 RAG 系统作为对比基线 |
| Phase 3 | Agent-RAG 核心组件 | 7 | ⭐⭐⭐ | 实现 Planner/StepDefiner/Agent/Summarizer 等 |
| Phase 4 | 工作流和主入口 | 2.5 | ⭐⭐ | 整合各组件为完整工作流，支持命令行运行 |
| Phase 5 | 评估和对比分析 | 3.5 | ⭐⭐ | 运行实验，收集数据，撰写报告 |
| **总计** | | **18.5** | | |

---

## 详细课时计划

### Phase 1: 环境配置与基础工具（3 课时）

#### 周期 1.1: 项目结构和环境配置（1 课时）

**教学内容**：
- 讲解项目目录结构和各文件用途（15 分钟）
- 指导学生安装依赖和配置 API Key（30 分钟）
- 学生自主测试环境配置（15 分钟）

**学习成果**：
- 理解项目框架
- 环境配置正确无误
- 能成功调用 LLM API 和本地 Embedding

**作业**：运行环境检查脚本，截图验证

---

#### 周期 1.2: 实现 LLM 客户端（1 课时）

**教学内容**：
- 讲解 OpenAI Python SDK 的基本使用（15 分钟）
- 讲解 Pydantic 的结构化输出（15 分钟）
- 学生独立实现 `chat()` 和 `chat_structured()` 方法（20 分钟）
- 代码审查和纠错（10 分钟）

**关键代码位置**：
- `src/llm_client.py` 中的 `LLMClient` 类

**学习成果**：
- 理解 LLM API 的调用方式
- 掌握 Pydantic 的使用
- 实现错误处理和重试机制

**自测命令**：
```bash
python -c "from src.llm_client import LLMClient; 
client = LLMClient(); 
print(client.chat('You are helpful', 'Hello'))"
```

---

#### 周期 1.3: 实现向量检索系统（1 课时）

**教学内容**：
- 讲解 Embedding 的原理和应用（10 分钟）
- 讲解 Sentence-Transformers 库的使用（15 分钟）
- 讲解向量相似度搜索（余弦相似度、L2 距离）（10 分钟）
- 学生实现 `Retriever.load_corpus()`, `build_index()`, `search()` 方法（20 分钟）
- 性能优化讨论（FAISS 可选）（5 分钟）

**关键代码位置**：
- `src/retriever.py` 中的 `LocalEmbeddingProvider` 和 `Retriever` 类

**学习成果**：
- 理解 Embedding 向量的含义
- 掌握向量相似度计算
- 能够构建和查询向量索引

**自测命令**：
```bash
python src/retriever.py
```

---

### Phase 2: Naive RAG 基线（2.5 课时）

#### 周期 2.1: Naive RAG 设计和实现（1.5 课时）

**教学内容**：
- 讲解 Naive RAG 的工作流（10 分钟）
- 讲解 Prompt 设计原理（10 分钟）
- 学生实现 `NaiveRAG.answer()` 方法（40 分钟）
- 代码审查和优化讨论（10 分钟）

**关键代码位置**：
- `baselines/naive_rag.py` 中的 `NaiveRAG` 类
- `src/prompts.py` 中的 `NAIVE_RAG_SYSTEM_MESSAGE` 等

**学习成果**：
- 理解传统 RAG 的架构和局限
- 掌握文档拼接和 Prompt 构造
- 能够实现基本的 QA 系统

**自测命令**：
```bash
python -c "from baselines.naive_rag import NaiveRAG; 
from src.retriever import get_retriever; 
retriever = get_retriever(); 
retriever.load_corpus(); 
retriever.build_index(); 
rag = NaiveRAG(retriever); 
result = rag.answer('Who is Shirley Temple?'); 
print(result['answer'][:200])"
```

---

#### 周期 2.2: Naive RAG 测试和优化（1 课时）

**教学内容**：
- 在小型测试集上运行 Naive RAG（15 分钟）
- 分析失败案例，讨论改进方向（20 分钟）
- 学生尝试优化 Prompt，改进表现（20 分钟）
- 性能对比讨论（5 分钟）

**学习成果**：
- 了解 Naive RAG 的实际表现
- 识别其局限性（固定检索、无反思）
- 为 Agent-RAG 做准备

---

### Phase 3: Agent-RAG 核心组件（7 课时）

#### 周期 3.1: Planner 规划器（1.5 课时）

**教学内容**：
- 讲解问题分解的重要性（10 分钟）
- 讲解 Few-shot Prompting 和结构化输出（15 分钟）
- 讲解 `PlanFormat` 数据结构（5 分钟）
- 学生实现 `Planner.plan()` 和 `_format_memory()` 方法（40 分钟）
- 测试和调试（10 分钟）

**关键代码位置**：
- `src/planner.py` 中的 `Planner` 类
- `src/prompts.py` 中的 `PLANNER_SYSTEM_MESSAGE`

**学习成果**：
- 理解问题分解策略
- 掌握 Few-shot Prompting 技巧
- 理解迭代优化的机制

**自测命令**：
```bash
python src/planner.py
```

---

#### 周期 3.2: Step Definer 步骤定义器（1.5 课时）

**教学内容**：
- 讲解查询优化的概念（10 分钟）
- 讲解如何利用历史信息优化查询（10 分钟）
- 学生实现 `StepDefiner.define_step()` 方法（40 分钟）
- 代码审查（10 分钟）

**关键代码位置**：
- `src/step_definer.py` 中的 `StepDefiner` 类

**学习成果**：
- 理解动态查询生成的好处
- 掌握上下文信息的利用
- 理解"细粒度检索"的优势

---

#### 周期 3.3: RAG Agent 问答代理（1.5 课时）

**教学内容**：
- 讲解问答代理的设计（10 分钟）
- 讲解置信度评分的意义（10 分钟）
- 讲解 `StepOutput` 数据格式（5 分钟）
- 学生实现 `RAGAgent.answer_step()` 方法（45 分钟）

**关键代码位置**：
- `src/rag_agent.py` 中的 `RAGAgent` 类

**学习成果**：
- 理解单步问答的设计
- 掌握置信度评估
- 理解故障检测的重要性

---

#### 周期 3.4: Summarizer 汇总器（1 课时）

**教学内容**：
- 讲解多步输出的整合方法（10 分钟）
- 讲解 `PlanSummary` 数据格式（5 分钟）
- 学生实现 `Summarizer.summarize()` 方法（35 分钟）
- 测试（10 分钟）

**关键代码位置**：
- `src/summarizer.py` 中的 `Summarizer` 类

**学习成果**：
- 理解信息整合的逻辑
- 掌握质量评分的方法

---

#### 周期 3.5: Executor 执行器（1.5 课时）

**教学内容**：
- 讲解执行器的协调职责（10 分钟）
- 讲解状态管理和信息流（15 分钟）
- 讲解可选的反思机制（迭代规划）（10 分钟）
- 学生实现 `PlanExecutor.execute()` 方法（40 分钟）

**关键代码位置**：
- `src/executor.py` 中的 `PlanExecutor` 类

**学习成果**：
- 理解多个 Agent 的协作方式
- 掌握状态管理和错误处理
- 理解反思循环的设计

---

### Phase 4: 工作流和主入口（2.5 课时）

#### 周期 4.1: LangGraph 工作流（1 课时）

**教学内容**：
- 讲解 LangGraph 的基本概念（节点、边、状态）（15 分钟）
- 讲解 `AgentState` 的定义（10 分钟）
- 讲解工作流的编译和执行（10 分钟）
- 学生实现 `AgentWorkflow._build_graph()` 和 `run()` 方法（20 分钟）

**关键代码位置**：
- `graph/workflow.py` 中的 `AgentWorkflow` 类

**学习成果**：
- 理解图结构和状态管理
- 掌握 LangGraph 的使用
- 理解工作流的可视化和调试

---

#### 周期 4.2: 主入口脚本（1.5 课时）

**教学内容**：
- 讲解命令行接口的设计（10 分钟）
- 讲解结果文件的格式规范（10 分钟）
- 学生实现 `run_agent_rag()` 和 `run_naive_rag()` 函数（50 分钟）
- 端到端测试（10 分钟）

**关键代码位置**：
- `run.py` 中的主函数

**学习成果**：
- 掌握完整的系统集成
- 理解数据流和结果格式
- 能够进行端到端的测试

**测试命令**：
```bash
python run.py --mode agent --start 0 --end 2 --output /tmp/test.jsonl
python run.py --mode naive --start 0 --end 2 --output /tmp/test.jsonl
```

---

### Phase 5: 评估和对比分析（3.5 课时）

#### 周期 5.1: 评估脚本实现（1.5 课时）

**教学内容**：
- 讲解 LLM 作为评估器的方法（15 分钟）
- 讲解评估指标的设计（准确率、F1 等）（10 分钟）
- 学生实现 `LLMEvaluator` 和 `evaluate_results()` 函数（45 分钟）
- 测试（10 分钟）

**关键代码位置**：
- `eval/evaluate.py` 中的评估函数

**学习成果**：
- 理解如何用 LLM 进行自动评估
- 掌握评估指标的计算
- 能够生成评估报告

---

#### 周期 5.2: 对比实验执行（1 课时）

**教学内容**：
- 讲解实验设计的最佳实践（10 分钟）
- 指导学生运行完整的实验（30 分钟）
  - 运行 Naive RAG
  - 运行 Agent-RAG
  - 评估两者的结果
- 结果分析和讨论（15 分钟）
- 数据收集和整理（5 分钟）

**预期输出**：
- `results_agent.jsonl` 和 `results_naive.jsonl`
- `eval_agent.json` 和 `eval_naive.json`
- 对比数据表格

---

#### 周期 5.3: 报告撰写和答辩准备（1 课时）

**教学内容**：
- 讲解报告的结构和要求（15 分钟）
- 讲解数据可视化（图表、表格）（10 分钟）
- 学生撰写实验报告（30 分钟）
- 答辩准备指导（5 分钟）

**报告要求**：
- 总长度：2000-3000 字
- 包含：目标、设计、结果、分析、结论
- 包含至少 2 个对比表格或图表
- 包含失败案例分析

---

## 课堂进度表（建议）

### 第 1 周（3 课时）
- **第 1-2 课**：Phase 1.1 + 1.2（环境和 LLM 客户端）
- **第 3 课**：Phase 1.3（向量检索）

### 第 2 周（2.5 课时）
- **第 4-5 课**：Phase 2.1（Naive RAG 实现）
- **第 6 课**：Phase 2.2（Naive RAG 优化和对比）+ Phase 3.1 开始

### 第 3-4 周（6.5 课时）
- **第 7-8 课**：Phase 3.1（Planner）
- **第 9-10 课**：Phase 3.2（Step Definer）
- **第 11-12 课**：Phase 3.3（RAG Agent）
- **第 13 课**：Phase 3.4 + 3.5（Summarizer + Executor）

### 第 5 周（2.5 课时）
- **第 14-15 课**：Phase 4.1 + 4.2（工作流和主入口）
- **第 16 课**：端到端测试

### 第 6 周（3.5 课时）
- **第 17 课**：Phase 5.1（评估脚本）
- **第 18 课**：Phase 5.2（对比实验）
- **第 19 课**：Phase 5.3（报告和答辩）

---

## 检查点和里程碑

| 阶段 | 检查点 | 预期产出 | 验收标准 |
|------|--------|---------|---------|
| Phase 1 | 1.3 | 能运行 `python src/retriever.py` | 正确加载语料库和执行搜索 |
| Phase 2 | 2.2 | `results_naive_baseline.jsonl` | 能成功处理 10+ 个问题 |
| Phase 3 | 3.5 | `results_agent_preliminary.jsonl` | 包含完整的 Plan 和 Step 输出 |
| Phase 4 | 4.2 | 能用 `python run.py` 命令运行系统 | 支持 --mode/--output/--input 参数 |
| Phase 5 | 5.3 | 最终实验报告和评估数据 | 包含定量对比和定性分析 |

---

## 学习资源清单

### 必读
- Pydantic 文档：https://docs.pydantic.dev/
- LangGraph 快速开始：https://langchain-ai.github.io/langgraph/
- Sentence Transformers 文档：https://www.sbert.net/
- OpenAI API 参考：https://platform.openai.com/docs/api-reference

### 推荐
- RAG 综述论文：https://arxiv.org/abs/2312.10997
- LLM Agent 设计模式：https://lilianweng.github.io/posts/2023-06-23-agent/
- Prompt Engineering 指南：https://platform.openai.com/docs/guides/prompt-engineering

### 可选
- 向量检索论文：https://arxiv.org/abs/2401.04245
- 多 Agent 协作：https://arxiv.org/abs/2308.03762

---

## 答辩评分细则（总分 100）

| 维度 | 权重 | 评分标准 |
|------|------|---------|
| **代码完整性** | 15% | 所有模块完整、可运行、无明显 bug |
| **代码质量** | 15% | 代码风格规范、注释清晰、错误处理完善 |
| **实验设计** | 20% | 测试集合理、参数配置一致、实验可重复 |
| **定量结果** | 20% | 数据完整、准确率计算正确、对比清晰 |
| **定性分析** | 20% | 失败案例分析深入、见解独特、有改进建议 |
| **答辩表现** | 10% | 理解深刻、表达清晰、对提问回答有据可查 |

---

## 常见时间陷阱

⚠️ 学生容易在以下环节超时：

1. **API 配置调试**（1-2 课时）
   - 预防：提前准备好 API Key，提供配置示例

2. **Embedding 模型下载**（0.5-1 课时）
   - 预防：提前下载或使用镜像源

3. **LLM 调用的 JSON 解析**（1-2 课时）
   - 预防：提供 fallback 机制，讲解常见错误

4. **向量检索调试**（1 课时）
   - 预防：提供测试数据和调试脚本

5. **工作流集成测试**（1-2 课时）
   - 预防：提供完整的端到端示例

---

## 教师备注

### 课前准备
- [ ] 准备示例 API Key（或购买相关 API 额度）
- [ ] 提前下载 Embedding 模型
- [ ] 准备测试数据集（corpus.jsonl 和 test_set.jsonl）
- [ ] 确认学生的网络可以访问 API

### 课中
- [ ] 准备屏幕共享的演示代码
- [ ] 准备故障排除的 checklist
- [ ] 记录学生的常见问题，为后续课程调整

### 课后
- [ ] 收集学生的代码并进行审查
- [ ] 记录进度，调整后续课程安排
- [ ] 为进度落后的学生提供补课机会

