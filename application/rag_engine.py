import os
from typing import List
from core.interfaces import IVectorStore

class RagEngine:
    """
    应用层：处理索引逻辑和检索流程。
    """
    def __init__(self, vector_store: IVectorStore):
        self.vector_store = vector_store

    def index_project(self, directory: str = "."):
        """
        自动扫描并索引项目中的代码文件
        """
        docs = []
        # 只索引具有代表意义的文件
        for root, _, files in os.walk(directory):
            if ".git" in root or ".venv" in root: continue
            for file in files:
                if file.endswith((".py", ".toml", ".md")):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            # 简单分块：按行或按函数分（演示用简单截断）
                            docs.append(f"文件: {file}\n路径: {path}\n内容摘要: {content[:500]}")
                    except:
                        pass
        self.vector_store.add_documents(docs)

    def get_related_context(self, query: str) -> str:
        # 下一步改进：在这里引入流程图中的【相关度检查】
        results = self.vector_store.query(query, top_k=3)
        if not results:
            return "\n[通知] RAG 扫描完成：未发现与此请求直接相关的本地代码片段。请基于常识或已分析的内容回答。"
        
        context = "\n--- 📚 RAG 检索到的参考代码 (标注来源) ---\n"
        for i, doc in enumerate(results):
            context += f"\n[参考 {i+1}]:\n{doc}\n"
        context += "\n----------------------------------------"
        return context
