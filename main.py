import argparse
import sys
import torch
from infrastructure.llm_service import QwenService
from infrastructure.storage_service import StorageService
from infrastructure.tools import DateTimeTool, ListCurrentDirTool, FileAnalysisTool, PlanTool
from application.chat_app import ChatApp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args()

    local_model_path = r"d:\ajanuw\ai_qwen\qwen2.5-3b"
    use_4bit = torch.cuda.is_available() and (torch.cuda.get_device_properties(0).total_memory / 1024**3 < 8)

    # 基础组件实例化
    storage = StorageService()
    
    # 2. 高级 RAG 初始化 (使用 FAISS 工业级实现)
    from infrastructure.vector_store import FaissVectorStore
    from application.rag_engine import RagEngine
    vector_store = FaissVectorStore() # 自动加载语义模型
    rag_engine = RagEngine(vector_store)
    print("🔭 [System] 异步构建全项目语义索引 (RAG)...")
    rag_engine.index_project(".") 
    
    llm = QwenService(local_model_path, use_4bit=use_4bit)
    
    # 3. 注入 RAG 到 ChatApp
    app = ChatApp(llm, storage, rag_engine=rag_engine, auto_approve=args.yes)

    # 注册扩展后的全能工具集
    from infrastructure.tools import WebSearchTool, DateTimeTool, ListCurrentDirTool, FileAnalysisTool, PlanTool
    app.register_tool(WebSearchTool()) # 增加联网搜索
    app.register_tool(DateTimeTool())
    app.register_tool(ListCurrentDirTool())
    app.register_tool(FileAnalysisTool())
    app.register_tool(PlanTool(storage))

    if args.task:
        app.run(args.task)
        sys.exit(0)

    print("\n💡 输入 'exit' 退出，输入 'clear' 清空历史\n")
    while True:
        user_input = input("👤 You: ").strip()
        if user_input.lower() in ['exit', 'quit']: break
        if user_input.lower() == 'clear':
            app.clear_history()
            continue
        if user_input: app.run(user_input)

if __name__ == "__main__":
    main()
