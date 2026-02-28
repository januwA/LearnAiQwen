import json
import math
import re
from collections import Counter
from pathlib import Path
import numpy as np
import faiss
from typing import Any, Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from core.interfaces import IVectorStore

class FaissVectorStore(IVectorStore):
    """
    基于 FAISS 和 Sentence-Transformers 的工业级本地向量存储。
    使用 CPU 索引，处理万级文档极快。
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", local_files_only: bool = False):
        # 加载轻量级语义向量嵌入模型
        print(f"🧠 正在加载 Embeddings 模型: {model_name}...")
        self.model = SentenceTransformer(model_name, local_files_only=local_files_only)
        self.dimension = self.model.get_sentence_embedding_dimension()
        # 使用余弦相似度（先归一化，再用内积）
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents: List[Dict[str, Any]] = []
        self.doc_token_counts: List[Counter[str]] = []
        self.df: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0

    def _tokenize(self, text: str) -> List[str]:
        # 支持英文/数字 token 与中文连续词块
        pattern = r"[A-Za-z_]\w+|\d+|[\u4e00-\u9fff]+"
        return [t.lower() for t in re.findall(pattern, text)]

    def _rebuild_sparse_stats(self) -> None:
        self.doc_token_counts = []
        self.df = {}
        total_len = 0
        for doc in self.documents:
            tokens = self._tokenize(str(doc.get("text", "")))
            cnt = Counter(tokens)
            self.doc_token_counts.append(cnt)
            total_len += len(tokens)
            for tok in cnt.keys():
                self.df[tok] = self.df.get(tok, 0) + 1
        self.avg_doc_len = (total_len / len(self.documents)) if self.documents else 0.0

    def _normalize_entries(self, documents: List[Dict[str, Any] | str]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in documents:
            if isinstance(item, str):
                normalized.append({"text": item})
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            normalized.append(item)
        return normalized

    def add_documents(self, documents: List[Dict[str, Any] | str]):
        entries = self._normalize_entries(documents)
        if not entries:
            return

        texts = [entry["text"] for entry in entries]
        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents.extend(entries)
        self._rebuild_sparse_stats()
        print(f"📚 RAG-FAISS: 已完成 {len(documents)} 条文档条目的语义索引")

    def _bm25_score_for_doc(self, doc_idx: int, q_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> float:
        if not self.documents or not q_tokens or self.avg_doc_len <= 0:
            return 0.0
        cnt = self.doc_token_counts[doc_idx]
        doc_len = sum(cnt.values())
        score = 0.0
        n_docs = len(self.documents)
        for tok in q_tokens:
            tf = cnt.get(tok, 0)
            if tf == 0:
                continue
            df = self.df.get(tok, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * doc_len / self.avg_doc_len)
            score += idf * ((tf * (k1 + 1)) / max(denom, 1e-9))
        return score

    def _dense_candidates(self, text: str, top_n: int) -> Tuple[Dict[int, float], Dict[int, int]]:
        query_vector = self.model.encode([text])
        query_vector = np.array(query_vector).astype("float32")
        faiss.normalize_L2(query_vector)
        top_n = min(top_n, len(self.documents))
        scores, indices = self.index.search(query_vector, top_n)
        dense_scores: Dict[int, float] = {}
        dense_ranks: Dict[int, int] = {}
        for rank, idx in enumerate(indices[0], start=1):
            if idx == -1:
                continue
            dense_scores[int(idx)] = float(scores[0][rank - 1])
            dense_ranks[int(idx)] = rank
        return dense_scores, dense_ranks

    def _sparse_candidates(self, q_tokens: List[str], top_n: int) -> Tuple[Dict[int, float], Dict[int, int]]:
        sparse_scores: Dict[int, float] = {}
        if not q_tokens:
            return sparse_scores, {}
        for i in range(len(self.documents)):
            s = self._bm25_score_for_doc(i, q_tokens)
            if s > 0:
                sparse_scores[i] = s
        sorted_items = sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        sparse_ranks = {idx: rank for rank, (idx, _) in enumerate(sorted_items, start=1)}
        sparse_scores = {idx: score for idx, score in sorted_items}
        return sparse_scores, sparse_ranks

    def query(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.documents:
            return []
        dense_top_n = max(20, top_k * 8)
        sparse_top_n = max(20, top_k * 8)
        q_tokens = self._tokenize(text)

        dense_scores, dense_ranks = self._dense_candidates(text, dense_top_n)
        sparse_scores, sparse_ranks = self._sparse_candidates(q_tokens, sparse_top_n)

        candidate_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        if not candidate_ids:
            return []

        dense_vals = list(dense_scores.values()) or [0.0]
        sparse_vals = list(sparse_scores.values()) or [0.0]
        dmin, dmax = min(dense_vals), max(dense_vals)
        smin, smax = min(sparse_vals), max(sparse_vals)

        ranked: List[Tuple[int, float, float, float, float]] = []
        for idx in candidate_ids:
            d_raw = dense_scores.get(idx, 0.0)
            s_raw = sparse_scores.get(idx, 0.0)
            d_norm = (d_raw - dmin) / (dmax - dmin + 1e-9) if dmax > dmin else 0.0
            s_norm = (s_raw - smin) / (smax - smin + 1e-9) if smax > smin else 0.0
            # 轻量重排特征：词汇覆盖率
            q_set = set(q_tokens)
            doc_set = set(self.doc_token_counts[idx].keys()) if idx < len(self.doc_token_counts) else set()
            overlap = (len(q_set & doc_set) / len(q_set)) if q_set else 0.0
            # RRF 融合，稳住密集/稀疏排名
            rrf = 0.0
            if idx in dense_ranks:
                rrf += 1.0 / (60 + dense_ranks[idx])
            if idx in sparse_ranks:
                rrf += 1.0 / (60 + sparse_ranks[idx])
            score = 0.50 * d_norm + 0.30 * s_norm + 0.15 * overlap + 2.5 * rrf
            ranked.append((idx, score, d_raw, s_raw, overlap))

        ranked.sort(key=lambda x: x[1], reverse=True)
        results: List[Dict[str, Any]] = []
        for idx, score, d_raw, s_raw, overlap in ranked[:top_k]:
            entry = dict(self.documents[idx])
            entry["score"] = float(score)
            entry["dense_score"] = float(d_raw)
            entry["sparse_score"] = float(s_raw)
            entry["overlap"] = float(overlap)
            results.append(entry)
        return results

    def save(self, directory: str) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target / "index.faiss"))
        (target / "documents.json").write_text(
            json.dumps(self.documents, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, directory: str) -> bool:
        target = Path(directory)
        index_path = target / "index.faiss"
        docs_path = target / "documents.json"

        if not index_path.exists() or not docs_path.exists():
            return False

        self.index = faiss.read_index(str(index_path))
        loaded = json.loads(docs_path.read_text(encoding="utf-8"))
        # 兼容旧格式：documents 可能是字符串数组
        if loaded and isinstance(loaded[0], str):
            self.documents = [{"text": item} for item in loaded]
        else:
            self.documents = loaded
        self._rebuild_sparse_stats()
        return True
