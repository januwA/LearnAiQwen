import os
from pathlib import Path
from typing import Dict, List
from core.interfaces import IVectorStore

class RagEngine:
    """
    应用层：处理索引逻辑和检索流程。
    """
    def __init__(
        self,
        vector_store: IVectorStore,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        min_score: float = 0.12,
    ):
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, max(0, chunk_size - 1))
        self.min_score = min_score

    def _iter_chunks(self, text: str):
        step = max(1, self.chunk_size - self.chunk_overlap)
        for start in range(0, len(text), step):
            end = start + self.chunk_size
            yield start, end, text[start:end]

    def index_project(self, directory: str = "."):
        """
        自动扫描并索引项目中的代码文件
        """
        docs: List[Dict[str, str | int]] = []
        base_dir = Path(directory).resolve()
        ignored_tokens = {".git", ".venv", "__pycache__", ".cache"}

        # 只索引具有代表意义的文件，保留元数据供检索后引用
        for root, _, files in os.walk(directory):
            if any(token in root for token in ignored_tokens):
                continue
            for file in files:
                if file.endswith((".py", ".toml", ".md")):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            rel_path = str(Path(path).resolve().relative_to(base_dir))
                            for idx, (start, end, chunk) in enumerate(self._iter_chunks(content), start=1):
                                text = chunk.strip()
                                if not text:
                                    continue
                                docs.append({
                                    "text": text,
                                    "file": file,
                                    "path": rel_path,
                                    "chunk_id": idx,
                                    "start": start,
                                    "end": min(end, len(content)),
                                })
                    except OSError as exc:
                        print(f"⚠️ 跳过文件失败: {path} ({exc})")
        self.vector_store.add_documents(docs)

    def get_related_context(self, query: str) -> str:
        results = self.vector_store.query(query, top_k=5) # 增加 top_k 提高召回
        filtered = [r for r in results if float(r.get("score", 0.0)) >= self.min_score]
        
        if not filtered:
            # 尝试关键词硬匹配兜底
            keywords = [k for k in query.split() if len(k) > 2]
            hard_matches = []
            if keywords:
                for doc in self.vector_store.documents:
                    text = str(doc.get("text", "")).lower()
                    if any(k.lower() in text for k in keywords):
                        hard_matches.append(doc)
                        if len(hard_matches) >= 3: break
            
            if hard_matches:
                filtered = hard_matches
                prefix = "\n⚠️ [RAG 兜底] 语义检索未命中，已切换至关键词硬匹配模式：\n"
            else:
                return "\n[通知] RAG 扫描完成：未发现与此请求直接相关的本地代码片段。请基于常识或已分析的内容回答。"
        else:
            prefix = "\n--- 📚 RAG 检索到的参考代码 (语义+关键词混合) ---\n"

        context = prefix
        for i, doc in enumerate(filtered, start=1):
            score = float(doc.get("score", 0.0))
            path = doc.get("path", "<unknown>")
            chunk_id = doc.get("chunk_id", "?")
            text = str(doc.get("text", "")).strip()
            context += (
                f"\n[参考 {i}] 路径: {path} | 相似度: {score:.3f}\n"
                f"{text}\n"
            )
        context += "\n----------------------------------------"
        return context
