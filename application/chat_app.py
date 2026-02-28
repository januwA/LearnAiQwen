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
        collect_feedback: bool = True,
    ):
        self.llm_service = llm_service
        self.storage = storage
        self.rag_engine = rag_engine
        self.auto_approve = auto_approve
        self.collect_feedback = collect_feedback
        self.tools: Dict[str, ITool] = {}
        self.console = Console()
        
        self.default_prompt = (
            "你是一个本地 Agent，具备工具调用能力。\n"
            "工作原则：\n"
            "1) 需要实时/外部信息时，优先调用工具，不要臆测。\n"
            "2) 简单事实或推理问题可直接回答。\n"
            "3) 当你无法直接获得信息时，不要说“做不到”，而是发起工具调用。\n"
            "4) 如果需要执行代码，可返回 ```python 代码块，系统会执行并回传结果。"
        )
        self.history = [{"role": "system", "content": self.default_prompt}]
        self.history.extend(self.storage.get_all_messages())

    def _decide_strategy(self, prompt: str) -> Dict[str, bool]:
        decide_prompt = (
            "你是路由器。判断当前请求是否需要检索与工具。\n"
            "你知道系统可调用工具（目录、文件、时间、联网、计划、python执行）。\n"
            "规则：若回答依赖实时/本地文件/联网信息，use_tools 应为 true。\n"
            "仅返回 JSON（不要 markdown），格式：\n"
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

    def run(self, prompt: str, stream: bool = True):
        strategy = {"use_rag": False, "use_tools": False}
        with self.console.status("[bold magenta]🧠 正在进行意图识别...", spinner="arc"):
            strategy = self._decide_strategy(prompt)
        
        self.console.print(
            f"[bold dim]📍 策略: use_rag={strategy['use_rag']}, use_tools={strategy['use_tools']}[/]"
        )

        # 2. 知识库查询与相关性检查 (RAG Pipeline)
        rag_context = ""
        if strategy["use_rag"] and self.rag_engine:
            with self.console.status("[bold green]🔍 正在检索并验证本地知识相关性...", spinner="dots"):
                rag_context = self.rag_engine.get_related_context(prompt)
            if "参考" in rag_context:
                self.console.print(Panel(rag_context, title="📚 RAG 精排结果 (标注来源)", border_style="blue"))
            else:
                self.console.print("[yellow]⚠️ RAG: 未发现高分相关结果，将进入基础生成路径。[/]")

        # 3. 窗口管理
        if len(self.history) > 10:
            self.history = [self.history[0]] + self.history[-6:]
            self.console.print("[dim italic]✂️ 上下文窗口裁减...[/]")

        # 4. 驱动任务或闲聊
        final_prompt = prompt
        if strategy["use_rag"] and rag_context:
            final_prompt = f"【可选参考上下文】\n{rag_context}\n\n用户问题: {prompt}"
        if strategy["use_tools"]:
            final_prompt = (
                f"{final_prompt}\n\n"
                "【系统提示】你拥有可调用工具。若回答需要实时/本地信息，请直接调用工具，不要猜测。\n"
                "工具选择参考：\n"
                "- 项目结构：优先 list_current_dir，再用 file_analysis 读取关键文件（如 README.md / pyproject.toml）。\n"
                "- Git 状态：优先调用 git_status。\n"
                "- 时间日期：优先调用 get_current_datetime。"
            )
        
        self.history.append({"role": "user", "content": final_prompt})
        self.storage.save_message("user", final_prompt)
        
        # 5. 执行核心推理
        allow_tools = strategy["use_tools"]
        used_tools, first_answer = self._process_iteration(stream, allow_tools=allow_tools)

        # 5.1 若出现“无本地访问能力”拒答，则自动重试一次并开启工具
        if not allow_tools and self.history and self.history[-1]["role"] == "assistant":
            last_answer = self.history[-1]["content"]
            if self._looks_local_access_refusal(last_answer):
                self.console.print("[yellow]↩️ 检测到能力拒答，自动切换到工具模式重试一次...[/]")
                self.history.append({
                    "role": "user",
                    "content": (
                        "不要再说无法做到。你有可用工具，请先调用合适的工具获取信息，"
                        "再基于工具结果回答。"
                    )
                })
                self._process_iteration(stream=False, allow_tools=True)

        # 5.2 若回答包含未取证的本地断言，则自动重试一次并开启工具
        elif self._needs_evidence_retry(prompt, first_answer, used_tools):
            self.console.print("[yellow]↩️ 检测到本地事实未取证，自动切换到工具模式重试一次...[/]")
            self.history.append({
                "role": "user",
                "content": "请先调用工具读取当前目录与关键文件，再基于实际结果回答，避免臆测。"
            })
            self._process_iteration(stream=False, allow_tools=True)

        # 6. 用户反馈闭环 (Feedback Loop) - 对应流程图底部
        if self.collect_feedback:
            self._collect_feedback(prompt)

    def _collect_feedback(self, query: str):
        # 只有最后一条消息是助理回复时才收集
        last_resp = self.history[-1]["content"] if self.history[-1]["role"] == "assistant" else "对话结束"
        
        choice = questionary.select(
            "🌟 对于本次回答，您觉得：",
            choices=[
                {"name": "👍 非常有帮助 (Positive)", "value": 1},
                {"name": "👎 没啥用 (Negative)", "value": -1},
                {"name": "⏭️ 跳过", "value": 0}
            ]
        ).ask()
        
        if choice != 0:
            self.storage.save_feedback(query, last_resp, choice)
            self.console.print("[italic green]感谢您的反馈！样本已存入 SQLite 用于系统进化库。[/]")

    def _process_iteration(self, stream: bool, allow_tools: bool, max_iterations: int = 8):
        executed_calls = set()
        tool_specs = [t.metadata for t in self.tools.values()] if allow_tools else None
        used_tools = False
        last_answer = ""
        for i in range(max_iterations):
            self.console.print(f"\n[bold cyan]🤖 Assistant (Step {i+1}): [/]")
            full_response = ""
            
            # 使用上下文管理防止显存碎片（可选）
            if stream and i == 0:
                with Live(vertical_overflow="visible", console=self.console) as live:
                    for chunk in self.llm_service.generate_stream(self.history, tools=tool_specs):
                        full_response += chunk
                        live.update(Markdown(full_response))
            else:
                full_response = self.llm_service.generate_response(self.history, tools=tool_specs)
                self.console.print(Markdown(full_response))

            if not full_response.strip(): break
            
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
                    self.console.print(
                        Panel(
                            f"[bold yellow]🐍 检测到 Python 代码块 #{idx+1}[/]\n[dim]{code[:1200]}[/]",
                            border_style="yellow",
                        )
                    )
                    if self.auto_approve:
                        confirm = True
                    else:
                        confirm = questionary.confirm("是否执行上述 Python 代码?", default=False).ask()
                    if not confirm:
                        self.console.print("[bold red]🚫 Python 执行已由用户拒绝。[/]")
                        continue
                    result = self._execute_python(code)
                    self.console.print(f"✅ Python 执行结果: [italic cyan]{result[:800]}...[/]")
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
                
                # 使用 Rich 美化展示
                self.console.print(Panel(f"[bold yellow]🔧 调用:[/][bold white] {name}[/]\n[dim]参数: {json.dumps(args, ensure_ascii=False)}[/]", border_style="yellow"))
                
                # 使用 Questionary 替代 input 实现高级交互
                if self.auto_approve:
                    confirm = True
                else:
                    msg = "满意这个任务计划并授权执行吗?" if is_plan else "是否允许执行上述工具操作?"
                    confirm = questionary.confirm(msg, default=True).ask()
                
                if confirm:
                    res = self.tools[name].execute(**args) if name in self.tools else "找不到该工具模块"
                    self.console.print(f"✅ 执行结果: [italic cyan]{str(res)[:500]}...[/]")
                    self.history.append({"role": "tool", "name": name, "content": str(res)})
                    self.storage.save_message("tool", f"[{name}] {res}")
                else:
                    self.console.print("[bold red]🚫 任务已由用户手动终止。[/]")
                    return "STOP"
            except Exception as e:
                self.console.print(f"[bold red]❌ 解析异常: {e}[/]")
        return "OK"

    def clear_history(self):
        self.storage.clear_all()
        self.history = [self.history[0]]
        self.console.print("[bold green]🧹 对话历史与数据库记忆已重置。[/]")
