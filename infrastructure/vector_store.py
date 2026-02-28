import os
import numpy as np
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
from core.interfaces import IVectorStore

class FaissVectorStore(IVectorStore):
    """
    基于 FAISS 和 Sentence-Transformers 的工业级本地向量存储。
    使用 CPU 索引，处理万级文档极快。
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # 加载轻量级语义向量嵌入模型
        print(f"🧠 正在加载 Embeddings 模型: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []

    def add_documents(self, documents: List[str]):
        if not documents: return
        # 语义向量化
        embeddings = self.model.encode(documents)
        self.index.add(np.array(embeddings).astype('float32'))
        self.documents.extend(documents)
        print(f"📚 RAG-FAISS: 已完成 {len(documents)} 条文档条目的语义索引")

    def query(self, text: str, top_k: int = 3) -> List[str]:
        if not self.documents: return []
        
        # 将搜索词向量化
        query_vector = self.model.encode([text])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), top_k)
        
        results = []
        for i in indices[0]:
            if i != -1 and i < len(self.documents):
                results.append(self.documents[i])
        return results
