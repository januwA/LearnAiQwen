import os
import json
import subprocess
from typing import Dict, Any, List
from core.interfaces import ITool
from infrastructure.storage_service import StorageService
from duckduckgo_search import DDGS
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

class WebSearchTool(ITool):
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "进行实时联网搜索，获取最新信息。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"}
                    },
                    "required": ["query"]
                }
            }
        }
    def execute(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                if not results: return "未找到相关搜索结果。"
                formatted = "\n\n".join([f"标题: {r['title']}\n摘要: {r['body']}" for r in results])
                return f"🔍 Web 搜索结果:\n{formatted}"
        except Exception as e:
            return f"搜索失败: {str(e)}"

class DateTimeTool(ITool):
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_current_datetime",
                "description": "获取当前的日期和时间",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        }
    def execute(self, **kwargs) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class ListCurrentDirTool(ITool):
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "list_current_dir",
                "description": "列出当前工作目录下的详细内容（区分文件和目录）",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        }
    def execute(self, **kwargs) -> str:
        try:
            items = os.listdir('.')
            res = []
            for item in items:
                prefix = "[DIR]" if os.path.isdir(item) else "[FILE]"
                size = "" if os.path.isdir(item) else f" ({os.path.getsize(item) // 1024}KB)"
                res.append(f"{prefix} {item}{size}")
            return "当前目录详细内容:\n" + "\n".join(res)
        except Exception as e:
            return f"错误: {str(e)}"

class FileAnalysisTool(ITool):
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "file_analysis",
                "description": "分析文件内容的专业工具",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["read_lines", "get_info"]},
                        "path": {"type": "string"}
                    },
                    "required": ["action", "path"]
                }
            }
        }
    def execute(self, action: str, path: str, **kwargs) -> str:
        root = os.path.abspath(".")
        abs_path = os.path.abspath(path)
        if os.path.commonpath([root, abs_path]) != root:
            return "❌ 仅允许读取当前项目目录内文件。"
        if os.path.isdir(abs_path):
            if action == "get_info":
                items = os.listdir(abs_path)
                files = sum(1 for x in items if os.path.isfile(os.path.join(abs_path, x)))
                dirs = sum(1 for x in items if os.path.isdir(os.path.join(abs_path, x)))
                return f"目录: {abs_path}\n子目录数: {dirs}\n文件数: {files}\n总条目: {len(items)}"
            return f"❌ '{path}' 是目录。请提供具体文件名。"
        try:
            stat = os.stat(abs_path)
            with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if action == "get_info":
                return f"文件: {abs_path}\n大小: {stat.st_size} bytes\n行数: {len(content.splitlines())}"
            return f"--- 文件内容 ({abs_path}) ---\n{content[:3000]}"
        except Exception as e:
            return f"失败: {str(e)}"

class GitStatusTool(ITool):
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "git_status",
                "description": "获取当前仓库分支和工作区变更状态（git status）。",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        }

    def execute(self, **kwargs) -> str:
        try:
            cp = subprocess.run(
                ["git", "status", "--short", "--branch"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            if cp.returncode != 0:
                err = cp.stderr.strip() or "git status 执行失败"
                return f"❌ {err}"
            output = cp.stdout.strip()
            return output if output else "工作区干净，无变更。"
        except Exception as e:
            return f"❌ git 状态获取失败: {e}"

class PlanTool(ITool):
    def __init__(self, storage: StorageService):
        self.storage = storage
        self.console = Console()

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "manage_plan",
                "description": "Agent 核心计划管理。复杂任务前必须 set，每步结束必须 update。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["set", "update", "show"]},
                        "tasks": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string", "description": "步骤内容"},
                                    "status": {"type": "string", "enum": ["pending", "working", "done"]}
                                }
                            }
                        }
                    },
                    "required": ["action"]
                }
            }
        }

    def _render_table(self, title: str):
        tasks = self.storage.get_tasks()
        if not tasks: return
        table = Table(title=f"📋 {title}", border_style="magenta", header_style="bold cyan")
        table.add_column("状态", style="bold")
        table.add_column("任务步骤", style="white")
        for t in tasks:
            status_map = {"pending": "[yellow]⏳ 等待[/]", "working": "[blue]⚙️ 执行[/]", "done": "[green]✅ 完成[/]"}
            table.add_row(status_map.get(t['status'], t['status']), t['content'])
        self.console.print(table)

    def execute(self, action: str, tasks: List[Dict[str, Any]] = None) -> str:
        if action == "set" and tasks:
            self.storage.sync_tasks(tasks)
            self._render_table("任务计划已确权")
            return f"✅ 计划已建立。第一步任务是: '{tasks[0]['content']}'。请立刻开始执行，调用相关工具（如 list_current_dir）。"
            
        elif action == "update" and tasks:
            current = {t["content"]: t["status"] for t in self.storage.get_tasks()}
            for t in tasks: current[t["content"]] = t["status"]
            new_list = [{"content": k, "status": v} for k, v in current.items()]
            self.storage.sync_tasks(new_list)
            self._render_table("任务进度已更新")
            done_count = len([t for t in new_list if t['status'] == 'done'])
            pending = [t['content'] for t in new_list if t['status'] != 'done']
            next_hint = f"下一个待办任务是: '{pending[0]}'" if pending else "所有任务已完成。"
            return f"✅ 进度更新成功（{done_count}/{len(new_list)}）。{next_hint}"
            
        elif action == "show":
            self._render_table("当前待办清单")
            return "已在屏幕显示当前计划清单。"
        return "完成"
