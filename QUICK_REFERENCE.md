# Mini-MA-RAG 实验快速参考指南

## 快速开始（5 分钟）

```bash
# 1. 进入项目目录
cd RAG/mini-ma-rag-starter

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key
cp .env.sample .env
# 编辑 .env，填写你的 OPENAI_API_KEY 等

# 5. 测试环境
python -c "from src.llm_client import LLMClient; LLMClient().chat('sys', 'hello')"
```

---

## 项目文件速查表

### 📋 核心模块（需要学生实现）

| 文件 | 类/函数 | 任务描述 | 难度 |
|------|--------|--------|------|
| `src/planner.py` | `Planner.plan()` | 问题分解规划 | ⭐⭐ |
| `src/planner.py` | `Planner._format_memory()` | 格式化历史经验 | ⭐ |
| `src/step_definer.py` | `StepDefiner.define()` | 步骤查询优化 | ⭐⭐ |
| `src/step_definer.py` | `StepDefiner._format_step_outputs()` | 格式化步骤历史 | ⭐ |
| `src/rag_agent.py` | `RAGAgent.run()` | 完整 RAG 流程 | ⭐⭐ |
| `src/rag_agent.py` | `RAGAgent._extract()` | 文档信息抽取 | ⭐ |
| `src/rag_agent.py` | `RAGAgent._generate()` | 基于 notes 生成答案 | ⭐ |
| `src/summarizer.py` | `Summarizer.summarize()` | 结果汇总 | ⭐ |
| `src/summarizer.py` | `Summarizer._format_memory()` | 格式化步骤输出 | ⭐ |
| `src/executor.py` | `Aggregator.run()` | 聚合类步骤处理 | ⭐ |
| `src/executor.py` | `PlanExecutor.execute()` | 执行协调 | ⭐⭐ |
| `baselines/naive_rag.py` | `NaiveRAG.answer()` | 简单 RAG 回答 | ⭐ |
| `baselines/naive_rag.py` | `run_naive_rag_on_dataset()` | Naive RAG 批处理 | ⭐ |

### ✅ 已完成模块（可直接使用）

| 文件 | 说明 |
|------|------|
| `src/llm_client.py` | LLM API 客户端（chat + 结构化输出 + 重试） |
| `src/retriever.py` | 向量检索（SBERT + FAISS + 暴力搜索） |
| `graph/workflow.py` | LangGraph 工作流编排（AgentWorkflow） |
| `run.py` | 主入口脚本（agent / naive 模式） |
| `eval/evaluate.py` | 评估脚本（LLM-as-Judge + 精确匹配） |

### 📝 参考文件（已完成，可参考）

| 文件 | 说明 |
|------|------|
| `src/config.py` | 全局配置 |
| `src/state.py` | 数据类型定义 |
| `src/prompts.py` | Prompt 模板 |
| `data/corpus.jsonl` | 文本语料库（需运行 `python data/build_corpus.py` 生成） |
| `data/test_set.jsonl` | 测试集 |
| `.env.sample` | 环境变量示例 |

---

## 数据格式速查

### Embedding 向量格式
```python
# shape: (n_docs, embedding_dim)
# 例: (1000, 384) - 1000 篇文档，每个 384 维向量
embeddings = np.array([[0.1, 0.2, ..., 0.384]])
```

### 计划格式（PlanFormat）
```python
{
    "analysis": "Think step-by-step...",
    "step": [
        "Who is Shirley Temple?",
        "What government positions did she hold?",
        "When did she serve as Chief of Protocol?"
    ]
}
```

### 步骤输出格式（QAAnswerFormat）
```python
{
    "analysis": "思考过程...",
    "answer": "Shirley Temple served as Chief of Protocol from 1976-1977.",
    "success": "Yes",  # "Yes" 或 "No"
    "rating": 9        # 0-10 的置信度评分
}
```

### 汇总格式（PlanSummaryFormat）
```python
{
    "output": "Successful",  # "Successful" 或 "Unsuccessful"
    "answer": "Shirley Temple served as Chief of Protocol of the United States from 1976-1977.",
    "score": 9              # 0-10 的质量评分
}
```

### 最终结果格式
```python
{
    "id": "q001",
    "question": "Who is Shirley Temple?",
    "ground_truth": "She was an actress and diplomat.",
    "predicted": "Shirley Temple was an American actress and diplomat...",
    "status": "Successful",
    "score": 9,
    "plan": ["步骤1", "步骤2", ...],
    "step_outputs": [
        {"step": "...", "success": "Yes", "rating": 9},
        ...
    ]
}
```

---

## 常用命令速查

### 环境检查
```bash
# 检查 Python 版本
python --version

# 检查依赖
pip list | grep -E "openai|sentence-transformers|langgraph"

# 检查 API 连接
python -c "from src.config import require_llm_config; require_llm_config()"
```

### 组件测试
```bash
# 测试 LLM 客户端
python -c "from src.llm_client import LLMClient; 
           client = LLMClient(); 
           print(client.chat('sys: you are helpful', 'hello'))"

# 测试 Embedding
python src/retriever.py

# 测试 Planner
python src/planner.py

# 测试 Naive RAG
python -c "from baselines.naive_rag import NaiveRAG; 
           from src.retriever import get_retriever; 
           r = get_retriever(); r.load_corpus(); r.build_index(); 
           rag = NaiveRAG(r); 
           print(rag.answer('Who is Shirley Temple?')['answer'][:100])"
```

### 实验运行
```bash
# 运行 Naive RAG（所有测试集）
python run.py --mode naive --output results_naive.jsonl

# 运行 Agent-RAG（所有测试集）
python run.py --mode agent --output results_agent.jsonl

# 运行前 5 题进行快速测试
python run.py --mode agent --start 0 --end 5 --output test_results.jsonl

# 使用自定义测试集
python run.py --mode agent --input data/test_set.jsonl --output results_hard.jsonl

# 评估结果
python eval/evaluate.py results_agent.jsonl --output eval_agent.json
python eval/evaluate.py results_naive.jsonl --output eval_naive.json
```

### 结果分析
```bash
# 查看结果样本
head -1 results_agent.jsonl | python -m json.tool

# 统计结果数量
wc -l results_agent.jsonl

# 查看评估分数
python -c "import json; 
           with open('eval_agent.json') as f: 
               data = json.load(f); 
               print(f\"准确率: {data['accuracy']:.1%}\")"
```

---

## 常见错误排查

### ❌ "未检测到 OPENAI_API_KEY"
**原因**：.env 文件未配置或路径错误

**解决**：
```bash
cp .env.sample .env
# 编辑 .env，确保 OPENAI_API_KEY 不为空
echo "OPENAI_API_KEY=$YOUR_KEY" >> .env
```

### ❌ "模型加载超时"
**原因**：Sentence-Transformers 在从 HuggingFace 下载模型

**解决**：
```bash
# 方案 A: 使用镜像源
export HF_ENDPOINT=https://huggingface-mirror.com

# 方案 B: 提前下载
python -c "from sentence_transformers import SentenceTransformer; 
           SentenceTransformer('all-MiniLM-L6-v2')"

# 方案 C: 指定本地路径
# 在 .env 中设置 LOCAL_EMBEDDING_MODEL=/path/to/model
```

### ❌ "JSON 解析失败"
**原因**：某些 API（如 DeepSeek）不支持 `response_format`

**解决**：无需手动修复，`chat_structured()` 会自动回退到普通生成 + 手动解析

### ❌ "检索结果不相关"
**原因**：TOP_K 过小或 Embedding 模型不适合

**解决**：
```bash
# 在 .env 中调整
RETRIEVAL_TOP_K=5  # 增加到 5 或 10

# 或使用更强大的 Embedding 模型
LOCAL_EMBEDDING_MODEL=bge-m3  # 更好的多语言支持
```

### ❌ "Agent-RAG 响应缓慢"
**原因**：多次 API 调用

**解决**：
```bash
# 减少 TOP_K
RETRIEVAL_TOP_K=2

# 或使用更快的模型
MODEL_NAME=gpt-3.5-turbo  # 而不是 gpt-4o-mini
```

---

## 性能基准参考

### 预期性能（基于示例代码）

| 系统 | 准确率 | 平均时间 | Token 消耗 |
|------|--------|---------|-----------|
| Naive RAG | 80-90% | 2-3s | 300-500 |
| Agent-RAG | 85-95% | 4-6s | 800-1200 |

**注**：实际结果取决于模型、测试集和 Embedding 质量

### 成本估计（使用 OpenAI API）

| 模型 | 输入价格 | 输出价格 | 推荐用途 |
|------|---------|---------|---------|
| gpt-3.5-turbo | $0.5/1M | $1.5/1M | 快速原型 |
| gpt-4o-mini | $0.15/1M | $0.6/1M | **推荐** |
| gpt-4o | $2.5/1M | $10/1M | 高精度任务 |

**成本计算示例**：
```
运行 100 题的 Agent-RAG：
- 平均 1000 tokens/题 × 100 题 = 100K tokens
- 成本 ≈ $0.015 (使用 gpt-4o-mini)
```

---

## 调试技巧

### 1. 打印调试信息
```python
# 在关键位置加入 print
import json
print(f"[DEBUG] 检索结果: {json.dumps(docs[:1], ensure_ascii=False)}")
print(f"[DEBUG] 生成的计划: {plan}")
print(f"[DEBUG] 步骤置信度: {rating}/10")
```

### 2. 使用 Python debugger
```python
import pdb; pdb.set_trace()  # 在代码中设置断点
# 或使用 VS Code 的 Debug 功能
```

### 3. 验证输入输出格式
```python
from pydantic import ValidationError
try:
    result = PlanFormat(step=["step1", "step2"])
    print("✓ 格式正确:", result)
except ValidationError as e:
    print("✗ 格式错误:", e)
```

### 4. 比较两个系统的差异
```bash
# 在相同的 5 题上运行两个系统
python run.py --mode naive --start 0 --end 5 --output test_naive.jsonl
python run.py --mode agent --start 0 --end 5 --output test_agent.jsonl

# 手动对比结果
python -c "
import json
with open('test_naive.jsonl') as f:
    naive = [json.loads(line) for line in f]
with open('test_agent.jsonl') as f:
    agent = [json.loads(line) for line in f]

for n, a in zip(naive, agent):
    print(f\"Q: {n['question'][:50]}...\")
    print(f\"Naive: {n['predicted'][:80]}...\")
    print(f\"Agent: {a['predicted'][:80]}...\")
    print()
"
```

---

## 学习进度自检清单

### Phase 1: 基础工具
- [ ] 环境配置成功，无 import 错误
- [ ] LLM 客户端能正确调用 API
- [ ] 向量检索能正确加载和查询语料库
- [ ] 能独立解决常见的配置问题

### Phase 2: Naive RAG
- [ ] 实现了 `NaiveRAG.answer()` 方法
- [ ] 能在 10+ 个问题上成功运行
- [ ] 理解 Naive RAG 的局限性
- [ ] 能分析几个失败案例

### Phase 3: Agent-RAG
- [ ] 实现了所有 5 个组件（Planner 等）
- [ ] 每个组件都能独立测试成功
- [ ] 理解各组件的职责和协作方式
- [ ] 能解释为什么需要多步骤

### Phase 4: 工作流
- [ ] 实现了完整的 LangGraph 工作流
- [ ] `python run.py` 命令能成功运行
- [ ] 理解 StateGraph 和节点的概念
- [ ] 能调试工作流的执行过程

### Phase 5: 评估
- [ ] 运行了完整的对比实验
- [ ] 生成了评估报告
- [ ] 撰写了实验报告（≥2000 字）
- [ ] 能用数据支持你的结论

---

## 推荐进度跟踪

**周一**：
- 完成 Phase 1（环境 + 基础工具）
- 里程碑：`python src/retriever.py` 能正常运行

**周二-三**：
- 完成 Phase 2（Naive RAG）+ Phase 3 前半（Planner + Step Definer）
- 里程碑：`python src/planner.py` 能生成有效的计划

**周四**：
- 完成 Phase 3 后半（RAG Agent + Summarizer + Executor）
- 里程碑：单个问题能完整执行完整的 Agent-RAG 流程

**周五**：
- 完成 Phase 4（工作流 + 主入口）
- 里程碑：`python run.py --mode agent` 能处理整个测试集

**周末**：
- 完成 Phase 5（评估 + 报告）
- 里程碑：生成对比报告和评估数据

---

## 额外资源

### 在线工具
- **JSON 验证**：https://jsonlint.com/
- **Prompt 测试**：https://platform.openai.com/playground
- **LLM 成本计算器**：https://www.tokencounter.com/

### 示例和模板
- **示例答案代码**：见 `mini-ma-rag/` 目录（完整实现）
- **Prompt 模板**：`src/prompts.py`（包含系统 Prompt 和 Human Prompt）
- **测试数据**：`data/test_set.jsonl`

### 相关论文和博客
- RAG 综述：https://arxiv.org/abs/2312.10997
- Agent 设计：https://lilianweng.github.io/posts/2023-06-23-agent/
- Prompt 工程：https://platform.openai.com/docs/guides/prompt-engineering

