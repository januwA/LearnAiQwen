import os
import json
import re
import questionary
from typing import List, Dict, Any
from core.interfaces import ILLMService, ITool
from infrastructure.storage_service import StorageService
from application.rag_engine import RagEngine

from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text
from rich.status import Status

class ChatApp:
    def __init__(self, llm_service: ILLMService, storage: StorageService, rag_engine: RagEngine = None, auto_approve: bool = False):
        self.llm_service = llm_service
        self.storage = storage
        self.rag_engine = rag_engine
        self.auto_approve = auto_approve
        self.tools: Dict[str, ITool] = {}
        self.console = Console()
        
        self.default_prompt = (
            "你是一个受控 Agent。你的思维必须保持在核心逻辑内。\n"
            "【行为铁律】：\n"
            "1. **禁止模仿系统输出**：严禁在回复中输出 'Loading weights', 'Done', 'Exit' 等假扮系统提示的信息。\n"
            "2. **工具驱动**：只要计划中还有 pending 的任务，你必须输出 JSON 工具调用。禁止单纯的文字描述。\n"
            "3. **先规划后行动**：必须先 set。每步结束必须 update 并把 status 改为 done。"
        )
        self.history = [{"role": "system", "content": self.default_prompt}]
        self.history.extend(self.storage.get_all_messages())

    def register_tool(self, tool: ITool):
        self.tools[tool.metadata["function"]["name"]] = tool

    def run(self, prompt: str, stream: bool = True):
        # 1. 意图识别 (Intent Recognition) - 对应流程图首个菱形
        intent = "UNKNOWN"
        with self.console.status("[bold magenta]🧠 正在进行意图识别...", spinner="arc"):
             # 极速识别：询问模型意图（SEARCH/TOOL/CHAT）
             intent_prompt = f"分析以下用户输入，仅返回一个单词(SEARCH/TOOL/CHAT): '{prompt}'"
             intent = self.llm_service.generate_response([{"role": "user", "content": intent_prompt}]).strip().upper()
        
        self.console.print(f"[bold dim]📍 意图分流: {intent}[/]")

        # 2. 知识库查询与相关性检查 (RAG Pipeline)
        rag_context = ""
        if intent in ["SEARCH", "TOOL"] and self.rag_engine:
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
        if intent == "TOOL":
            tasks = self.storage.get_tasks()
            if not tasks or all(t['status'] == 'done' for t in tasks):
                final_prompt = f"【意图:工具执行】\n{rag_context}\n【必选：调用 manage_plan(action='set')】\n任务: {prompt}"
        elif intent == "SEARCH":
            final_prompt = f"【意图:知识查询】\n{rag_context}\n分析并回答: {prompt}"
        
        self.history.append({"role": "user", "content": final_prompt})
        self.storage.save_message("user", final_prompt)
        
        # 5. 执行核心推理
        self._process_iteration(stream)

        # 6. 用户反馈闭环 (Feedback Loop) - 对应流程图底部
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

    def _process_iteration(self, stream: bool, max_iterations: int = 8):
        executed_calls = set()
        for i in range(max_iterations):
            self.console.print(f"\n[bold cyan]🤖 Assistant (Step {i+1}): [/]")
            full_response = ""
            
            # 使用上下文管理防止显存碎片（可选）
            if stream and i == 0:
                with Live(vertical_overflow="visible", console=self.console) as live:
                    for chunk in self.llm_service.generate_stream(self.history, tools=[t.metadata for t in self.tools.values()]):
                        full_response += chunk
                        live.update(Markdown(full_response))
            else:
                full_response = self.llm_service.generate_response(self.history, tools=[t.metadata for t in self.tools.values()])
                self.console.print(Markdown(full_response))

            if not full_response.strip(): break
            
            is_tool = ("<tool_call>" in full_response or "```json" in full_response)
            
            if is_tool:
                if full_response in executed_calls:
                    # 自动阻断重复调用
                    self.history.append({"role": "user", "content": "检测到重复，请推进到下一个步骤。"})
                    continue
                executed_calls.add(full_response)
                
                status = self._handle_tool_call(full_response)
                if status == "STOP": break
            else:
                # 检查计划是否全部完成 (done)
                tasks = self.storage.get_tasks()
                has_pending = any(t['status'] != 'done' for t in tasks)
                
                if has_pending and i < max_iterations - 1:
                    # 强硬手段：如果任务没完模型就只说话，自动追加一条用户提示逼迫它继续
                    msg = "⚠️ 警告：当前任务计划尚未完成。严禁只用文字回复！请立刻调用工具执行下一个计划步骤。"
                    self.history.append({"role": "assistant", "content": full_response})
                    self.history.append({"role": "user", "content": msg})
                    self.console.print(f"\n[bold red]🔄 [系统驱动][/] 检测到任务未结束，正在强制驱动模型进入 Step {i+2}...")
                    continue 
                else:
                    self.history.append({"role": "assistant", "content": full_response})
                    self.storage.save_message("assistant", full_response)
                    break 

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
