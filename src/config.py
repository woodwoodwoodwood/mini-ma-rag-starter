"""
Mini-MA-RAG: 配置管理模块
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# LLM 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "/mnt/4090/data/jianglei/models/Qwen/Qwen3-0.6B")

# Embedding 配置（API 方式）
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", OPENAI_API_KEY)
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", OPENAI_BASE_URL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# 本地 Embedding 配置（当不使用 API 时回退）
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# 检索配置
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))   # 最大迭代反思轮数

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
TEST_SET_PATH = os.path.join(DATA_DIR, "test_set.jsonl")


def require_llm_config():
    """检查 LLM 配置是否完整"""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "[ERROR] 未检测到 OPENAI_API_KEY。\n"
            "请复制 .env.sample 为 .env，并填写你的 API Key。\n"
            "支持 OpenAI、DeepSeek、Qwen 等任何兼容 OpenAI 格式的 API。"
        )
