import argparse
import os
import sys
from pathlib import Path

# 解析参数以确定是否启用 verbose 模式（必须在导入其他库之前）
# 这样可以在导入 transformers/sentence-transformers 之前就设置好日志级别
_verbose = "-v" in sys.argv or "--verbose" in sys.argv
if not _verbose:
    # 抑制第三方库的日志和进度条
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from huggingface_hub import snapshot_download
from infrastructure.llm_service import QwenService
from infrastructure.storage_service import StorageService
from application.chat_app import ChatApp

EMBEDDING_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"

def _get_config_dir() -> Path:
    """获取配置目录，统一使用用户家目录"""
    # 所有平台都使用 ~/.ai_qwen
    if sys.platform == "win32":
        return Path(os.environ.get("USERPROFILE", "~")) / ".ai_qwen"
    else:
        return Path.home() / ".ai_qwen"

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

def _resolve_embedding_model_path(requested_path: str, verbose: bool = False) -> str:
    local_path = Path(requested_path).resolve()
    if local_path.exists():
        return str(local_path)

    cached = _find_cached_embedding_snapshot()
    if cached is not None:
        if verbose:
            print(f"[System] 使用本地 HF 缓存 Embeddings 模型: {cached}")
        return str(cached)

    if verbose:
        print("[System] 未找到本地 Embeddings 模型，正在自动下载...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=EMBEDDING_REPO_ID,
        local_dir=str(local_path),
    )
    if verbose:
        print(f"[System] Embeddings 模型已下载到: {downloaded}")
    return str(Path(downloaded).resolve())

def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="任务描述（也可通过 stdin 管道输入）")
    parser.add_argument("-y", "--yes", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true", help="开启调试模式，显示详细日志")
    # 模型路径使用当前脚本所在目录作为基准，支持从任意目录运行
    script_dir = Path(__file__).parent.resolve()
    parser.add_argument("--model-path", type=str, default=str(script_dir / "qwen2.5-3b"))
    parser.add_argument("--embedding-model-path", type=str, default=str(script_dir / "models/all-MiniLM-L6-v2"))
    parser.add_argument("--reindex", action="store_true", help="强制重建 RAG 索引")
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="离线模式（默认开启）：仅从本地目录加载模型，不访问 HF Hub",
    )
    args = parser.parse_args()

    # 设置日志级别
    if not args.verbose:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("faiss").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    else:
        logging.basicConfig(level=logging.DEBUG)

    # 支持从 stdin 管道读取任务
    task = args.task
    if not task and not sys.stdin.isatty():
        task = sys.stdin.read().strip()

    if not task:
        # 没有任务时进入交互模式
        task = None

    local_model_path = str(Path(args.model_path).resolve())
    if not Path(local_model_path).exists():
        raise FileNotFoundError(f"本地模型目录不存在: {local_model_path}")
    embedding_model_path = _resolve_embedding_model_path(args.embedding_model_path, verbose=args.verbose)

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    use_4bit = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory / 1024**3 < 8)

    # 获取配置目录
    config_dir = _get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"[System] 配置目录: {config_dir}")

    # 基础组件实例化
    storage = StorageService(db_path=str(config_dir / "agent_memory.db"))

    # 2. 高级 RAG 初始化 (使用 FAISS 工业级实现)
    from infrastructure.vector_store import FaissVectorStore
    from application.rag_engine import RagEngine
    vector_store = FaissVectorStore(
        model_name=embedding_model_path,
        local_files_only=True,
        verbose=args.verbose,
    )
    rag_engine = RagEngine(vector_store)
    # RAG 索引放在配置目录下，按项目路径哈希区分不同项目
    import hashlib
    project_hash = hashlib.md5(str(Path.cwd()).encode()).hexdigest()[:8]
    index_dir = config_dir / "rag" / project_hash
    index_dir.mkdir(parents=True, exist_ok=True)
    loaded = (not args.reindex) and vector_store.load(str(index_dir))
    if loaded:
        if args.verbose:
            print(f"🔭 [System] 已加载本地 RAG 索引: {index_dir}")
    else:
        if args.verbose:
            print("🔭 [System] 正在构建项目语义索引 (RAG)...")
        rag_engine.index_project(".")
        vector_store.save(str(index_dir))
        if args.verbose:
            print(f"✅ [System] RAG 索引已保存到: {index_dir}")

    llm = QwenService(local_model_path, use_4bit=use_4bit)

    # 3. 注入 RAG 到 ChatApp
    app = ChatApp(
        llm,
        storage,
        rag_engine=rag_engine,
        auto_approve=args.yes,
        quiet=bool(task and args.yes and not args.verbose),
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
        RagSearchTool,
        ShellCommandTool,
        FileSearchTool,
        CodeSearchTool,
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
    app.register_tool(ShellCommandTool()) # 新增：通用 Shell 命令工具
    app.register_tool(FileSearchTool())   # 新增：文件搜索工具
    app.register_tool(CodeSearchTool())   # 新增：代码内容搜索工具

    if task:
        result = app.run(task, stream=not (args.yes and not args.verbose))
        if args.yes and not args.verbose:
            print(result)
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
