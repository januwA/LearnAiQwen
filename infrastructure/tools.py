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
        # 允许读取系统 Pictures 目录或其他外部目录（如果是绝对路径且用户提供）
        if not os.path.exists(abs_path):
            return f"❌ 文件不存在: {path}"
            
        is_image = abs_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
        
        try:
            stat = os.stat(abs_path)
            if action == "get_info":
                info = f"文件: {abs_path}\n大小: {stat.st_size} bytes"
                if not is_image:
                    try:
                        with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = len(f.read().splitlines())
                            info += f"\n行数: {lines}"
                    except: pass
                return info
            
            if is_image:
                return f"❌ '{path}' 是图像文件，请使用 image_analysis 工具。"

            with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return f"--- 文件内容 ({abs_path}) ---\n{content[:3000]}"
        except Exception as e:
            return f"失败: {str(e)}"

class ImageAnalysisTool(ITool):
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "image_analysis",
                "description": "获取图像的基本信息（尺寸、格式、模式等元数据）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "图片路径"}
                    },
                    "required": ["path"]
                }
            }
        }

    def execute(self, path: str) -> str:
        from PIL import Image
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            return f"❌ 图片不存在: {path}"
        try:
            with Image.open(abs_path) as img:
                info = [
                    f"格式: {img.format}",
                    f"尺寸: {img.width}x{img.height}",
                    f"模式: {img.mode}",
                    f"路径: {abs_path}"
                ]
                if hasattr(img, 'info'):
                    info.append(f"元数据: {json.dumps(img.info, ensure_ascii=False)[:200]}...")
                return "🖼️ 图像分析结果:\n" + "\n".join(info)
        except Exception as e:
            return f"❌ 图像读取失败: {str(e)}"

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

class FileEditTool(ITool):
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "file_edit",
                "description": "对文件进行原子级修改（替换或追加内容）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "文件路径"},
                        "action": {"type": "string", "enum": ["replace", "append"], "description": "操作类型"},
                        "old_content": {"type": "string", "description": "要替换的旧内容（仅 replace 时需要）"},
                        "new_content": {"type": "string", "description": "新内容"}
                    },
                    "required": ["path", "action", "new_content"]
                }
            }
        }

    def execute(self, path: str, action: str, new_content: str, old_content: str = None) -> str:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path) and action == "replace":
            return f"❌ 文件不存在: {path}"
        
        try:
            if action == "replace":
                if old_content is None: return "❌ replace 操作需要 old_content。"
                with open(abs_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if old_content not in content:
                    return f"❌ 在文件中未找到指定的 old_content，请核对内容后重试。"
                new_full_content = content.replace(old_content, new_content, 1)
                with open(abs_path, 'w', encoding='utf-8') as f:
                    f.write(new_full_content)
                return f"✅ 已成功替换文件 '{path}' 中的内容。"
            
            elif action == "append":
                with open(abs_path, 'a', encoding='utf-8') as f:
                    f.write("\n" + new_content)
                return f"✅ 已成功追加内容到文件 '{path}'。"
        except Exception as e:
            return f"❌ 修改文件失败: {str(e)}"

class SystemContextTool(ITool):
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_system_context",
                "description": "获取系统运行环境信息（OS, CPU, 内存, GPU 等）。",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        }

    def execute(self, **kwargs) -> str:
        import platform
        import psutil
        import torch
        
        mem = psutil.virtual_memory()
        cpu_count = psutil.cpu_count(logical=True)
        gpu_info = "N/A"
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_info = f"{gpu_name} ({gpu_mem:.1f}GB)"
            
        info = [
            f"操作系统: {platform.system()} {platform.release()}",
            f"CPU 核心数: {cpu_count}",
            f"内存总量: {mem.total / (1024**3):.1f}GB (可用: {mem.available / (1024**3):.1f}GB)",
            f"GPU 信息: {gpu_info}",
            f"Python 版本: {platform.python_version()}"
        ]
        return "\n".join(info)

class PythonReplTool(ITool):
    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "python_repl",
                "description": "运行 Python 代码并获取输出结果，适用于复杂计算和数据分析。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "要运行的 Python 代码"}
                    },
                    "required": ["code"]
                }
            }
        }

    def execute(self, code: str) -> str:
        # 这里为了演示简单直接调用，实际应与 ChatApp 内部复用逻辑
        import sys
        import subprocess
        cmd = [sys.executable, "-c", code]
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        except Exception as e:
            return f"执行失败: {str(e)}"

class RagSearchTool(ITool):
    def __init__(self, rag_engine: Any):
        self.rag_engine = rag_engine

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": "在本地代码库和文档中进行语义搜索，获取相关的代码片段和逻辑说明。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词或描述性问题"}
                    },
                    "required": ["query"]
                }
            }
        }

    def execute(self, query: str) -> str:
        if not self.rag_engine:
            return "❌ RAG 引擎未初始化，无法进行本地搜索。"
        try:
            context = self.rag_engine.get_related_context(query)
            return context
        except Exception as e:
            return f"❌ RAG 搜索失败: {str(e)}"

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


class ShellCommandTool(ITool):
    """
    执行 Shell 命令的通用工具。
    适用于 git/npm/pip/docker 等命令行操作。
    """

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "shell_command",
                "description": "执行 Shell 命令并获取输出。适用于 git/npm/pip/docker 等命令行操作。危险操作（如 rm -rf）会被拒绝。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "要执行的完整命令"},
                        "cwd": {"type": "string", "description": "工作目录（可选，默认为当前目录）"}
                    },
                    "required": ["command"]
                }
            }
        }

    def _is_dangerous(self, command: str) -> bool:
        """检测危险命令"""
        dangerous_patterns = [
            "rm -rf /", "rm -rf /*", "mkfs", "dd if=",
            ":(){:|:&};:", "chmod -R 777 /", "chown -R"
        ]
        cmd_lower = command.lower()
        return any(pattern in cmd_lower for pattern in dangerous_patterns)

    def execute(self, command: str, cwd: str = None) -> str:
        # 安全检查
        if self._is_dangerous(command):
            return "❌ 拒绝执行危险命令，该操作可能对系统造成破坏。"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd or os.getcwd(),
                timeout=120,
                encoding="utf-8",
                errors="replace"
            )

            output_parts = []
            if result.returncode != 0:
                output_parts.append(f"退出码：{result.returncode}")
            if result.stdout:
                output_parts.append(f"STDOUT:\n{result.stdout.strip()}")
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr.strip()}")

            if not output_parts:
                return "命令执行成功，无输出。"

            return "\n".join(output_parts)

        except subprocess.TimeoutExpired:
            return "❌ 命令执行超时（120 秒），请尝试优化命令或分段执行。"
        except Exception as e:
            return f"❌ 命令执行失败：{str(e)}"


class FileSearchTool(ITool):
    """
    模糊查找文件的工具，支持 glob 模式匹配。
    """

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "file_search",
                "description": "根据文件名或模式搜索文件。支持 glob 通配符（如 *.py、**/test/*.js）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "搜索模式，如 '*.py' 或 '**/*.txt'"},
                        "path": {"type": "string", "description": "搜索起始目录（可选，默认为当前目录）"},
                        "max_results": {"type": "integer", "description": "最大返回结果数（可选，默认 50）"}
                    },
                    "required": ["pattern"]
                }
            }
        }

    def execute(self, pattern: str, path: str = ".", max_results: int = 50) -> str:
        import fnmatch

        search_path = os.path.abspath(path) if path else os.getcwd()
        if not os.path.isdir(search_path):
            return f"❌ 目录不存在：{path}"

        # 处理 glob 模式
        is_recursive = "**" in pattern
        simple_pattern = pattern.replace("**/", "").replace("**", "")

        results = []
        try:
            if is_recursive:
                for root, dirs, files in os.walk(search_path):
                    # 跳过隐藏目录和常见忽略目录
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                    for file in files:
                        if fnmatch.fnmatch(file, simple_pattern) or simple_pattern in file:
                            rel_path = os.path.relpath(os.path.join(root, file), search_path)
                            results.append(rel_path)
                            if len(results) >= max_results:
                                break
                    if len(results) >= max_results:
                        break
            else:
                for file in os.listdir(search_path):
                    if fnmatch.fnmatch(file, pattern) or pattern in file:
                        results.append(file)
                        if len(results) >= max_results:
                            break

            if not results:
                return f"未找到匹配模式 '{pattern}' 的文件。"

            formatted = "\n".join(f"  {r}" for r in results)
            suffix = f"\n... 共 {len(results)}+ 个结果" if len(results) >= max_results else ""
            return f"🔍 搜索结果 (模式：{pattern}):{suffix}\n{formatted}"

        except Exception as e:
            return f"❌ 搜索失败：{str(e)}"


class CodeSearchTool(ITool):
    """
    代码内容搜索工具，使用 grep 或 ripgrep 搜索代码。
    """

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "code_search",
                "description": "在代码文件中搜索包含特定关键词或正则表达式的行。优先使用 ripgrep (rg)，如果没有则使用 grep。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "搜索关键词或正则表达式"},
                        "file_type": {"type": "string", "description": "文件类型过滤（可选），如 'py'、'js'、'ts'"},
                        "path": {"type": "string", "description": "搜索目录（可选，默认为当前目录）"},
                        "max_results": {"type": "integer", "description": "最大返回结果数（可选，默认 30）"}
                    },
                    "required": ["pattern"]
                }
            }
        }

    def execute(self, pattern: str, file_type: str = None, path: str = ".", max_results: int = 30) -> str:
        search_path = os.path.abspath(path) if path else os.getcwd()
        if not os.path.isdir(search_path):
            return f"❌ 目录不存在：{path}"

        # 构建 grep 命令
        use_ripgrep = self._has_ripgrep()

        if use_ripgrep:
            cmd = ["rg", "--line-number", "--color", "never", pattern]
            if file_type:
                cmd.extend(["--type", file_type])
            cmd.extend(["--max-count", str(max_results)])
            cmd.append(search_path)
        else:
            cmd = ["grep", "-rn", "--color=never", pattern]
            if file_type:
                cmd.extend(["--include", f"*.{file_type}"])
            cmd.extend(["-m", str(max_results)])
            cmd.append(search_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                encoding="utf-8",
                errors="replace"
            )

            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= max_results:
                    return f"🔍 代码搜索结果 (关键词：{pattern}):\n{result.stdout.strip()}\n... (共 {max_results}+ 个结果，已截断)"
                return f"🔍 代码搜索结果 (关键词：{pattern}):\n{result.stdout.strip()}"
            elif result.returncode == 1:
                return f"未找到匹配 '{pattern}' 的代码。"
            else:
                error = result.stderr.strip() or "未知错误"
                return f"❌ 搜索失败：{error}"

        except subprocess.TimeoutExpired:
            return "❌ 搜索超时（30 秒），请尝试更精确的关键词。"
        except Exception as e:
            return f"❌ 搜索执行失败：{str(e)}"

    def _has_ripgrep(self) -> bool:
        """检测系统是否安装了 ripgrep"""
        try:
            result = subprocess.run(
                ["rg", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
