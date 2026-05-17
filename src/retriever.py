"""
Mini-MA-RAG: 检索器模块

支持两种模式：
1. 本地 Sentence-Transformers（默认，无需 API Key，CPU 可跑）
2. OpenAI Embedding API（可选）

向量检索：小数据(<10000)用 numpy 暴力搜索，大数据可接入 FAISS。
"""

import json
import os
import numpy as np
from typing import List, Tuple, Optional

from src.config import (
    LOCAL_EMBEDDING_MODEL,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    TOP_K,
    DATA_DIR,
    CORPUS_PATH,
)


class EmbeddingProvider:
    """Embedding 提供者接口"""

    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class LocalEmbeddingProvider(EmbeddingProvider):
    """基于 sentence-transformers 的本地 Embedding"""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model.eval()

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # L2 归一化（使内积等价于余弦相似度）
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms


class APIEmbeddingProvider(EmbeddingProvider):
    """基于 OpenAI API 的 Embedding"""

    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def encode(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=texts)
        embeddings = np.array([item.embedding for item in response.data])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms


class Retriever:
    """
    统一的检索器。
    加载语料库，建立向量索引，支持 add/search。
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        top_k: int = TOP_K,
        use_faiss: bool = False,
    ):
        self.top_k = top_k
        self.use_faiss = use_faiss
        self.docs: List[dict] = []          # 原始文档列表
        self.doc_ids: List[str] = []        # 文档 ID 列表
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index = None

        # 初始化 Embedding 提供者
        if embedding_provider is not None:
            self.embedder = embedding_provider
        elif EMBEDDING_API_KEY:
            self.embedder = APIEmbeddingProvider(EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL)
        else:
            print("[INFO] 使用本地 Embedding 模型:", LOCAL_EMBEDDING_MODEL)
            self.embedder = LocalEmbeddingProvider()

    def load_corpus(self, corpus_path: str = CORPUS_PATH):
        """从 jsonl 文件加载语料库"""
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"语料库文件不存在: {corpus_path}\n"
                f"请运行 python -c \"from data.build_corpus import build; build()\" 生成测试语料。"
            )
        self.docs = []
        self.doc_ids = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                self.docs.append(item)
                self.doc_ids.append(str(item.get("id", len(self.doc_ids))))
        print(f"[INFO] 加载语料库: {len(self.docs)} 篇文档")

    def build_index(self):
        """为语料库构建向量索引"""
        if not self.docs:
            raise ValueError("请先调用 load_corpus() 加载语料")
        texts = [doc["text"] for doc in self.docs]
        print("[INFO] 正在编码语料库...")
        self.embeddings = self.embedder.encode(texts)
        print(f"[INFO] 索引维度: {self.embeddings.shape}")

        if self.use_faiss:
            self._build_faiss_index()

    def _build_faiss_index(self):
        import faiss
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.embeddings.astype(np.float32))
        print("[INFO] FAISS 索引构建完成")

    def search(self, query: str, top_k: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """
        检索与 query 最相关的文档。
        返回: (文档文本列表, 文档ID列表)
        """
        k = top_k or self.top_k
        query_emb = self.embedder.encode([query])

        if self.use_faiss and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(query_emb.astype(np.float32), k)
            indices = indices[0]
        else:
            # 暴力搜索：计算内积（因已归一化，等价于余弦相似度）
            similarities = np.dot(self.embeddings, query_emb[0])
            indices = np.argsort(similarities)[::-1][:k]

        docs = []
        doc_ids = []
        for idx in indices:
            idx = int(idx)
            docs.append(self.docs[idx]["text"])
            doc_ids.append(self.doc_ids[idx])
        return docs, doc_ids

    def add_documents(self, documents: List[dict]):
        """动态添加文档（用于在线扩展）"""
        start_id = len(self.docs)
        for i, doc in enumerate(documents):
            self.docs.append(doc)
            self.doc_ids.append(str(doc.get("id", start_id + i)))
        texts = [d["text"] for d in documents]
        new_embs = self.embedder.encode(texts)
        if self.embeddings is None:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])
        if self.use_faiss and self.faiss_index is not None:
            self.faiss_index.add(new_embs.astype(np.float32))


def get_retriever(use_faiss: bool = False) -> Retriever:
    """工厂函数：获取配置好的 Retriever"""
    retriever = Retriever(use_faiss=use_faiss)
    retriever.load_corpus()
    retriever.build_index()
    return retriever
