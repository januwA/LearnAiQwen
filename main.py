import argparse
import os
import sys
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from infrastructure.llm_service import QwenService
from infrastructure.storage_service import StorageService
from application.chat_app import ChatApp

EMBEDDING_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"

def _find_cached_embedding_snapshot() -> Path | None:
    home = Path(os.environ.get("USERPROFILE", ""))
    local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
    roots = [
        home / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots",
        local_app_data / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots",
    ]
    candidates = []
    for root in roots:
        if root.exists():
            candidates.extend([p for p in root.iterdir() if p.is_dir()])
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def _resolve_embedding_model_path(requested_path: str) -> str:
    local_path = Path(requested_path).resolve()
    if local_path.exists():
        return str(local_path)

    cached = _find_cached_embedding_snapshot()
    if cached is not None:
        print(f"[System] 使用本地 HF 缓存 Embeddings 模型: {cached}")
        return str(cached)

    print("[System] 未找到本地 Embeddings 模型，正在自动下载...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=EMBEDDING_REPO_ID,
        local_dir=str(local_path),
    )
    print(f"[System] Embeddings 模型已下载到: {downloaded}")
    return str(Path(downloaded).resolve())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("-y", "--yes", action="store_true")
    parser.add_argument("--model-path", type=str, default="./qwen2.5-3b")
    parser.add_argument("--embedding-model-path", type=str, default="./models/all-MiniLM-L6-v2")
    parser.add_argument("--index-dir", type=str, default=".cache/rag")
    parser.add_argument("--reindex", action="store_true", help="强制重建 RAG 索引")
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="离线模式（默认开启）：仅从本地目录加载模型，不访问 HF Hub",
    )
    args = parser.parse_args()

    local_model_path = str(Path(args.model_path).resolve())
    if not Path(local_model_path).exists():
        raise FileNotFoundError(f"本地模型目录不存在: {local_model_path}")
    embedding_model_path = _resolve_embedding_model_path(args.embedding_model_path)

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    use_4bit = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory / 1024**3 < 8)

    # 基础组件实例化
    storage = StorageService()
    
    # 2. 高级 RAG 初始化 (使用 FAISS 工业级实现)
    from infrastructure.vector_store import FaissVectorStore
    from application.rag_engine import RagEngine
    vector_store = FaissVectorStore(
        model_name=embedding_model_path,
        local_files_only=True,
    )
    rag_engine = RagEngine(vector_store)
    index_dir = str(Path(args.index_dir).resolve())
    loaded = (not args.reindex) and vector_store.load(index_dir)
    if loaded:
        print(f"🔭 [System] 已加载本地 RAG 索引: {index_dir}")
    else:
        print("🔭 [System] 正在构建项目语义索引 (RAG)...")
        rag_engine.index_project(".")
        vector_store.save(index_dir)
        print(f"✅ [System] RAG 索引已保存到: {index_dir}")
    
    llm = QwenService(local_model_path, use_4bit=use_4bit)
    
    # 3. 注入 RAG 到 ChatApp
    app = ChatApp(
        llm,
        storage,
        rag_engine=rag_engine,
        auto_approve=args.yes,
        collect_feedback=(args.task is None),
    )

    from infrastructure.tools import (
        WebSearchTool, 
        DateTimeTool, 
        ListCurrentDirTool, 
        FileAnalysisTool, 
        PlanTool, 
        GitStatusTool,
        FileEditTool,
        SystemContextTool,
        PythonReplTool,
        ImageAnalysisTool,
        RagSearchTool
    )
    app.register_tool(WebSearchTool()) # 增加联网搜索
    app.register_tool(DateTimeTool())
    app.register_tool(ListCurrentDirTool())
    app.register_tool(FileAnalysisTool())
    app.register_tool(GitStatusTool())
    app.register_tool(PlanTool(storage))
    app.register_tool(FileEditTool())
    app.register_tool(SystemContextTool())
    app.register_tool(PythonReplTool())
    app.register_tool(ImageAnalysisTool())
    app.register_tool(RagSearchTool(rag_engine))

    if args.task:
        app.run(args.task)
        sys.exit(0)

    print("\n[System] 助手已就绪。输入 '/help' 查看指令，输入内容开始对话。\n")
    while True:
        try:
            user_input = input("👤 You: ").strip()
            if not user_input: continue
            
            # 优先处理以 / 开头的系统命令
            if app.handle_command(user_input):
                continue
                
            # 普通对话
            app.run(user_input)
            
        except KeyboardInterrupt:
            print("\n[System] 正在退出...")
            break
        except Exception as e:
            print(f"\n[Error] 运行时异常: {e}")

if __name__ == "__main__":
    main()
