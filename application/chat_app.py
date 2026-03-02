import os
import json
import re
import subprocess
import sys
import questionary
from typing import List, Dict, Any
from core.interfaces import ILLMService, ITool
from infrastructure.storage_service import StorageService
from application.rag_engine import RagEngine

from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown

class ChatApp:
    def __init__(
        self,
        llm_service: ILLMService,
        storage: StorageService,
        rag_engine: RagEngine = None,
        auto_approve: bool = False,
        quiet: bool = False,
    ):
        self.llm_service = llm_service
        self.storage = storage
        self.rag_engine = rag_engine
        self.auto_approve = auto_approve
        self.quiet = quiet
        self.tools: Dict[str, ITool] = {}
        self.console = Console()
        self.total_tokens = 0 # 当前会话累计 Token
        
        self.default_prompt = (
            "你是一个本地 Agent，具备工具调用能力。\n"
            "工作原则：\n"
            "1) 需要实时/外部信息时，优先调用工具，不要臆测。\n"
            "2) 简单事实或推理问题可直接回答。\n"
            "3) 当你无法直接获得信息时，不要说“做不到”，而是发起工具调用。\n"
            "4) 如果需要执行代码，可返回 ```python 代码块或调用 python_repl，系统会执行并回传结果。\n"
            "5) 修改代码请使用 file_edit 工具，不要只在回答中展示。"
        )
        self.history = [{"role": "system", "content": self.default_prompt}]
        self.history.extend(self.storage.get_all_messages())

        # Few-shot 示例：帮助小模型理解工具调用格式
        self._add_few_shot_examples()

    def _get_usage_bar(self, turn_tokens: int = 0) -> str:
        """生成 Token 状态栏文本"""
        import psutil
        mem = psutil.Process().memory_info().rss / (1024 * 1024)
        return (
            f"[bold dim][Stats] 本次: {turn_tokens}t | 会话累计: {self.total_tokens}t | "
            f"内存: {mem:.1f}MB[/]"
        )

    def handle_command(self, user_input: str) -> bool:
        """
        处理以 / 开头的系统命令。
        返回 True 表示已作为命令处理，False 表示是普通对话。
        """
        if not user_input.startswith('/'):
            return False

        parts = user_input.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ['/exit', '/quit']:
            self.console.print("[System] 正在退出系统...")
            import sys
            sys.exit(0)
        
        elif cmd == '/clear':
            self.clear_history()
            return True
        
        elif cmd == '/help':
            self._show_help()
            return True
        
        elif cmd == '/tools':
            self._list_tools()
            return True
        
        elif cmd == '/stats':
            self.console.print(Panel(self._get_usage_bar(), title="[System] 当前状态"))
            return True
            
        else:
            self.console.print(f"[System] 未知指令: {cmd}。输入 /help 查看可用指令。")
            return True

    def _show_help(self):
        help_text = (
            "可用系统指令:\n"
            "  /help    - 显示此帮助信息\n"
            "  /clear   - 清空当前对话历史和数据库记忆\n"
            "  /tools   - 列出当前已注册的所有工具及其功能\n"
            "  /stats   - 显示当前会话的 Token 和资源使用统计\n"
            "  /exit    - 退出程序\n"
        )
        self.console.print(Panel(help_text, title="[System] 指令列表"))

    def _list_tools(self):
        tools_text = ""
        for name, tool in self.tools.items():
            desc = tool.metadata["function"]["description"]
            tools_text += f"- [bold cyan]{name}[/]: {desc}\n"
        self.console.print(Panel(tools_text, title="[System] 已注册工具集"))

    def _decide_strategy(self, prompt: str) -> Dict[str, bool]:
        decide_prompt = (
            "你是任务分析引擎。根据用户输入判断是否需要使用检索(RAG)或外部工具。\n"
            "可用能力：\n"
            "- RAG: 针对本项目代码库/文档的语义检索。\n"
            "- Tools: 包括文件分析(读文件/信息)、图像分析(元数据)、网络搜索、Git、日期、Python执行、修改文件。\n"
            "规则：\n"
            "1. 提到具体文件路径、分析图片、执行代码、获取时间、搜索互联网，必须 use_tools=true。\n"
            "2. 询问关于本项目代码的逻辑、结构、实现，建议 use_rag=true。\n"
            "仅返回 JSON 格式：\n"
            '{"use_rag": boolean, "use_tools": boolean}\n'
            f"用户输入: {prompt}"
        )
        raw = self.llm_service.generate_response([{"role": "user", "content": decide_prompt}]).strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return {"use_rag": False, "use_tools": True}
        try:
            obj = json.loads(m.group(0))
            return {
                "use_rag": bool(obj.get("use_rag", False)),
                "use_tools": bool(obj.get("use_tools", False)),
            }
        except json.JSONDecodeError:
            return {"use_rag": False, "use_tools": True}

    def _looks_local_access_refusal(self, text: str) -> bool:
        patterns = [
            r"无法直接访问",
            r"无法访问.*本地",
            r"不能访问.*目录",
            r"无法获取.*日期",
            r"无法获取.*时间",
            r"无法.*当前.*时间",
            r"无法.*当前.*日期",
            r"我无法获取",
            r"我不能获取",
            r"cannot access.*local",
            r"can't access.*local",
            r"i cannot access",
            r"i can't access",
            r"i cannot get",
            r"i can't get",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    def _extract_python_blocks(self, text: str) -> List[str]:
        return re.findall(r"```python\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)

    def _execute_python(self, code: str) -> str:
        cmd = [sys.executable, "-c", code]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=os.getcwd(),
                timeout=30,
            )
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or "").strip()
            err = (exc.stderr or "").strip()
            return (
                "exit_code=124\n\n"
                f"stdout:\n{out if out else '<empty>'}\n\n"
                f"stderr:\n{err if err else '执行超时(30s)'}"
            )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        parts = [
            f"exit_code={completed.returncode}",
            f"stdout:\n{stdout}" if stdout else "stdout:\n<empty>",
            f"stderr:\n{stderr}" if stderr else "stderr:\n<empty>",
        ]
        return "\n\n".join(parts)

    def _needs_evidence_retry(self, prompt: str, answer: str, used_tools: bool) -> bool:
        if used_tools or not answer.strip():
            return False
        check_prompt = (
            "判断下面这次回答是否需要基于本地文件/目录证据重试。\n"
            "若用户问题需要分析本地项目，而回答包含具体文件/目录断言但未取证，应返回 RETRY。\n"
            "否则返回 OK。\n"
            f"用户问题: {prompt}\n"
            f"助手回答: {answer}\n"
            f"本轮是否调用过工具: {used_tools}"
        )
        verdict = self.llm_service.generate_response([{"role": "user", "content": check_prompt}]).strip().upper()
        return "RETRY" in verdict

    def register_tool(self, tool: ITool):
        self.tools[tool.metadata["function"]["name"]] = tool

    def run(self, prompt: str, stream: bool = True) -> str:
        # 1. 自动注入基础环境上下文（Gemini CLI 风格：感知当前状态）
        if self.quiet:
            try:
                # 尝试获取 git 状态和当前目录摘要作为背景
                cwd_summary = self.tools["list_current_dir"].execute() if "list_current_dir" in self.tools else "Unknown"
                git_summary = self.tools["git_status"].execute() if "git_status" in self.tools else "Not a git repo"

                context_msg = (
                    f"【当前环境状态】\n"
                    f"工作目录摘要:\n{cwd_summary[:500]}...\n"
                    f"Git 状态: {git_summary[:200]}\n"
                    "如果你需要更详细的文件内容或进行深入分析，请调用相关工具（如 rag_search 或 file_analysis）。"
                )
                # 这种系统提示不存入持久化数据库，仅作为当前轮次的上下文
                self.history.append({"role": "system", "content": context_msg})
            except:
                pass
        else:
            with self.console.status("[bold blue][System] 正在同步环境状态...", spinner="earth"):
                try:
                    # 尝试获取 git 状态和当前目录摘要作为背景
                    cwd_summary = self.tools["list_current_dir"].execute() if "list_current_dir" in self.tools else "Unknown"
                    git_summary = self.tools["git_status"].execute() if "git_status" in self.tools else "Not a git repo"
                    
                    context_msg = (
                        f"【当前环境状态】\n"
                        f"工作目录摘要:\n{cwd_summary[:500]}...\n"
                        f"Git 状态: {git_summary[:200]}\n"
                        "如果你需要更详细的文件内容或进行深入分析，请调用相关工具（如 rag_search 或 file_analysis）。"
                    )
                    # 这种系统提示不存入持久化数据库，仅作为当前轮次的上下文
                    self.history.append({"role": "system", "content": context_msg})
                except:
                    pass

        # 2. 窗口管理（保持最近上下文）
        if len(self.history) > 12:
            self.history = [self.history[0]] + self.history[-8:]
            if not self.quiet:
                self.console.print("[dim][System] 上下文窗口已自动裁减...[/]")

        # 3. 记录用户输入
        self.history.append({"role": "user", "content": prompt})
        self.storage.save_message("user", prompt)
        
        # 4. 执行核心推理（统一 ReAct 循环）
        # 模型现在根据 prompt 和环境上下文，自主决定是否调用 rag_search, file_analysis 等
        _, last_answer = self._process_iteration(stream, allow_tools=True)
        return last_answer

    def _process_iteration(self, stream: bool, allow_tools: bool, max_iterations: int = 8):
        executed_calls = set()
        tool_specs = [t.metadata for t in self.tools.values()] if allow_tools else None
        used_tools = False
        last_answer = ""
        for i in range(max_iterations):
            if not self.quiet:
                self.console.print(f"\n[bold cyan]Assistant (Step {i+1}): [/]")
            full_response = ""
            
            # 使用上下文管理防止显存碎片（可选）
            if stream and i == 0 and not self.quiet:
                with Live(vertical_overflow="visible", console=self.console) as live:
                    for chunk in self.llm_service.generate_stream(self.history, tools=tool_specs):
                        full_response += chunk
                        live.update(Markdown(full_response))
            else:
                full_response = self.llm_service.generate_response(self.history, tools=tool_specs)
                if not self.quiet:
                    self.console.print(Markdown(full_response))

            if not full_response.strip(): break
            
            # 统计 Token
            prompt_tokens = self.llm_service.get_token_count(str(self.history))
            resp_tokens = self.llm_service.get_token_count(full_response)
            turn_tokens = prompt_tokens + resp_tokens
            self.total_tokens += turn_tokens
            if not self.quiet:
                self.console.print(self._get_usage_bar(turn_tokens))

            is_tool = ("<tool_call>" in full_response or "```json" in full_response)
            python_blocks = self._extract_python_blocks(full_response)
            
            if is_tool:
                if not allow_tools:
                    self.history.append({"role": "user", "content": "当前任务不允许工具调用。请直接用自然语言回答。"})
                    continue
                if full_response in executed_calls:
                    # 自动阻断重复调用
                    self.history.append({"role": "user", "content": "检测到重复，请推进到下一个步骤。"})
                    continue
                executed_calls.add(full_response)
                
                status = self._handle_tool_call(full_response)
                used_tools = True
                if status == "STOP": break
            elif python_blocks:
                for idx, code in enumerate(python_blocks):
                    if not self.quiet:
                        self.console.print(
                            Panel(
                                f"[Python] 检测到代码块 #{idx+1}\n[dim]{code[:1200]}[/]",
                                border_style="yellow",
                            )
                        )
                    if self.auto_approve:
                        confirm = True
                    else:
                        confirm = questionary.confirm("是否执行上述 Python 代码?", default=False).ask()
                    if not confirm:
                        if not self.quiet:
                            self.console.print("[System] Python 执行已由用户拒绝。")
                        continue
                    result = self._execute_python(code)
                    if not self.quiet:
                        self.console.print(f"[System] 执行结果: [italic cyan]{result[:800]}...[/]")
                    self.history.append({"role": "tool", "name": "python_exec", "content": result})
                    self.storage.save_message("tool", f"[python_exec] {result}")
                    used_tools = True
            else:
                self.history.append({"role": "assistant", "content": full_response})
                self.storage.save_message("assistant", full_response)
                last_answer = full_response
                break 
        return used_tools, last_answer

    def _handle_tool_call(self, text: str) -> str:
        self.history.append({"role": "assistant", "content": text})
        self.storage.save_message("assistant", text)
        
        matches = re.findall(r'```json\n?(.*?)\n?```', text, re.DOTALL) or re.findall(r'<tool_call>\n?(.*?)\n?</tool_call>', text, re.DOTALL)
        if not matches: return "FAIL"

        for m in matches:
            try:
                call = json.loads(m.strip())
                name, args = call["name"], call.get("arguments", {})
                is_plan = (name == "manage_plan" and args.get("action") == "set")
                
                if not self.quiet:
                    self.console.print(Panel(f"[Call] {name}\n[Args] {json.dumps(args, ensure_ascii=False)}", border_style="yellow"))
                
                if self.auto_approve:
                    confirm = True
                else:
                    msg = "授权执行该计划吗?" if is_plan else "允许执行该工具吗?"
                    confirm = questionary.confirm(msg, default=True).ask()
                
                if confirm:
                    res = self.tools[name].execute(**args) if name in self.tools else "找不到该工具模块"
                    if not self.quiet:
                        self.console.print(f"[System] 执行成功，返回内容已注入上下文。")
                    self.history.append({"role": "tool", "name": name, "content": str(res)})
                    self.storage.save_message("tool", f"[{name}] {res}")
                else:
                    if not self.quiet:
                        self.console.print("[System] 任务已由用户手动终止。")
                    return "STOP"
            except Exception as e:
                if not self.quiet:
                    self.console.print(f"[System] 工具调用解析异常: {e}")
        return "OK"

    def clear_history(self):
        self.storage.clear_all()
        self.history = [self.history[0]]
        self.console.print("[bold green]🧹 对话历史与数据库记忆已重置。[/]")

    def _add_few_shot_examples(self):
        """添加 Few-shot 示例，帮助小模型理解工具调用格式。"""
        examples = [
            {"role": "user", "content": "现在几点了？"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "name": "get_current_datetime", "content": "2026-03-02 14:30:00"},
            {"role": "assistant", "content": "当前时间是 2026 年 3 月 2 日 14:30:00。"},
            {"role": "user", "content": "帮我找一下项目里所有的 Python 文件"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "name": "file_search", "arguments": {"pattern": "*.py"}, "content": "main.py\nutils.py"},
            {"role": "assistant", "content": "找到 Python 文件：main.py, utils.py"},
            {"role": "user", "content": "安装 requests 库"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "name": "shell_command", "arguments": {"command": "pip install requests"}, "content": "Successfully installed requests"},
            {"role": "assistant", "content": "requests 库已安装。"},
            {"role": "user", "content": "读取变更文件内容，总结下功能变更"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "name": "shell_command", "arguments": {"command": "git diff --stat"}, "content": " infrastructure/tools.py | 222 +\n main.py | 18 +-"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "name": "shell_command", "arguments": {"command": "git diff infrastructure/tools.py"}, "content": "[文件内容...]"},
            {"role": "assistant", "content": "本次变更总结了 3 个新工具..."},
        ]
        self.history.extend(examples)
