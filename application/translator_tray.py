import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import pyperclip
import requests
import win32api
import win32con
import win32gui
import win32com.client
import pythoncom
import re
import uuid

# 配置 (适配 ai_qwen)
API_BASE = "http://localhost:8000/v1"
CHAT_ENDPOINT = f"{API_BASE}/chat/completions"
MODELS_ENDPOINT = f"{API_BASE}/models"
DEFAULT_MODEL = "qwen-local"
REQUEST_TIMEOUT = 30
RETRY_COUNT = 2
RETRY_DELAY = 1.0

WINDOW = None
STATUS_VAR = None
MODEL_VAR = None
SOURCE_LANG_VAR = None
TARGET_LANG_VAR = None
INPUT_TEXT = None
OUTPUT_TEXT = None
TRANSLATE_BUTTON = None
COPY_BUTTON = None
MODEL_BOX = None
EVENT_QUEUE = queue.Queue()
TRAY = None

# 语言列表
LANG_OPTIONS = ["自动检测", "自动切换(中英)", "中文", "English", "日本語", "Français", "Deutsch", "Español", "한국어", "Русский"]

def translate_text(text):
    text = text.strip()
    if not text:
        return ""

    model_name = MODEL_VAR.get() if MODEL_VAR is not None else DEFAULT_MODEL
    source_lang_ui = SOURCE_LANG_VAR.get() if SOURCE_LANG_VAR is not None else "自动检测"
    target_lang_ui = TARGET_LANG_VAR.get() if TARGET_LANG_VAR is not None else "自动切换(中英)"

    # --- 核心逻辑判定 ---
    target_lang_actual = target_lang_ui

    # 情况 A: 目标语言选的是“自动切换(中英)”
    if target_lang_actual == "自动切换(中英)":
        zh_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_count = len(re.findall(r'[a-zA-Z]', text))
        target_lang_actual = "English" if zh_count > en_count else "中文"
    
    # 构建指令
    if source_lang_ui == "自动检测":
        prompt = f"把下面这段话翻译成{target_lang_actual}，只输出译文，不要解释：\n\n{text}"
    else:
        prompt = f"请把下面这段话从{source_lang_ui}翻译成{target_lang_actual}，只返回翻译结果，不要带解释：\n\n{text}"

    # OpenAI 兼容消息格式
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": 0.3,
        "max_tokens": 1024,
    }
    
    # 调试日志
    print(payload)

    last_error = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            response = requests.post(CHAT_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            result = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return result
        except Exception as exc:
            last_error = exc
            if attempt < RETRY_COUNT:
                time.sleep(RETRY_DELAY)
    raise last_error


def set_status(message):
    if STATUS_VAR is not None:
        STATUS_VAR.set(message)


def set_output(text):
    if OUTPUT_TEXT is None:
        return
    OUTPUT_TEXT.config(state="normal")
    OUTPUT_TEXT.delete("1.0", tk.END)
    OUTPUT_TEXT.insert(tk.END, text)
    OUTPUT_TEXT.config(state="disabled")


def fetch_models():
    models = []
    try:
        response = requests.get(MODELS_ENDPOINT, timeout=3)
        response.raise_for_status()
        data = response.json()
        models = [item.get("id", "") for item in data.get("data", []) if item.get("id")]
    except Exception:
        models = [DEFAULT_MODEL]
    return models


def refresh_models():
    if MODEL_VAR is None:
        return
    set_status("正在获取 API 模型...")
    def worker():
        options = fetch_models()
        if not options:
            options = [DEFAULT_MODEL]
        def update_ui():
            if MODEL_BOX is None:
                return
            MODEL_BOX["values"] = options
            current = MODEL_VAR.get()
            if current not in options:
                MODEL_VAR.set(options[0])
            set_status("模型已同步。")
        WINDOW.after(0, update_ui)
    threading.Thread(target=worker, daemon=True).start()


def run_translation():
    if TRANSLATE_BUTTON is None:
        return
    text = INPUT_TEXT.get("1.0", tk.END)
    if not text.strip():
        set_status("请输入要翻译的文本。")
        return

    TRANSLATE_BUTTON.config(state="disabled")
    COPY_BUTTON.config(state="disabled")
    set_status("翻译中 (调用 AI Qwen API)...")

    def worker():
        start_time = time.time()
        try:
            result = translate_text(text)
            if not result:
                message = "接口返回内容为空。"
            else:
                duration = time.time() - start_time
                message = f"翻译完成，耗时 {duration:.2f}s"
        except Exception as exc:
            result = f"错误: {exc}"
            message = "翻译失败"

        def update_ui():
            set_output(result)
            set_status(message)
            TRANSLATE_BUTTON.config(state="normal")
            COPY_BUTTON.config(state="normal")

        WINDOW.after(0, update_ui)

    threading.Thread(target=worker, daemon=True).start()


def copy_output():
    text = OUTPUT_TEXT.get("1.0", tk.END).strip()
    if text:
        pyperclip.copy(text)
        set_status("已复制译文到剪贴板。")


def speak_text(text):
    if not text.strip():
        set_status("没有可朗读的内容。")
        return
    def worker():
        try:
            pythoncom.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
            WINDOW.after(0, lambda: set_status("朗读完成。"))
        except Exception as exc:
            WINDOW.after(0, lambda: set_status(f"朗读失败: {exc}"))
        finally:
            pythoncom.CoUninitialize()
    threading.Thread(target=worker, daemon=True).start()


def speak_input():
    text = INPUT_TEXT.get("1.0", tk.END)
    speak_text(text)


def speak_output():
    text = OUTPUT_TEXT.get("1.0", tk.END)
    speak_text(text)


def hide_window():
    if WINDOW is not None:
        WINDOW.withdraw()


def show_window():
    if WINDOW is None:
        return
    WINDOW.deiconify()
    WINDOW.lift()
    WINDOW.focus_force()


def build_window():
    global WINDOW, STATUS_VAR, MODEL_VAR, SOURCE_LANG_VAR, TARGET_LANG_VAR, INPUT_TEXT, OUTPUT_TEXT, TRANSLATE_BUTTON, COPY_BUTTON, MODEL_BOX
    WINDOW = tk.Tk()
    WINDOW.title("AI Qwen 翻译助手")
    WINDOW.geometry("720x520")
    WINDOW.protocol("WM_DELETE_WINDOW", hide_window)

    STATUS_VAR = tk.StringVar(value="就绪")
    MODEL_VAR = tk.StringVar(value=DEFAULT_MODEL)
    SOURCE_LANG_VAR = tk.StringVar(value="自动检测")
    TARGET_LANG_VAR = tk.StringVar(value="自动切换(中英)")

    frame = tk.Frame(WINDOW, padx=10, pady=10)
    frame.pack(fill="both", expand=True)

    top_row = tk.Frame(frame)
    top_row.pack(fill="x")
    tk.Label(top_row, text="后端模型:").pack(side="left")
    MODEL_BOX = ttk.Combobox(top_row, textvariable=MODEL_VAR, width=28, state="readonly")
    MODEL_BOX.pack(side="left", padx=6)
    tk.Button(top_row, text="同步模型", command=refresh_models).pack(side="left")

    lang_row = tk.Frame(frame)
    lang_row.pack(fill="x", pady=(10, 0))
    tk.Label(lang_row, text="从").pack(side="left")
    source_box = ttk.Combobox(lang_row, textvariable=SOURCE_LANG_VAR, values=LANG_OPTIONS[:1]+LANG_OPTIONS[2:], width=12, state="readonly")
    source_box.pack(side="left", padx=6)
    tk.Label(lang_row, text="翻译成").pack(side="left")
    target_box = ttk.Combobox(lang_row, textvariable=TARGET_LANG_VAR, values=LANG_OPTIONS[1:], width=15, state="readonly")
    target_box.pack(side="left", padx=6)
    
    def swap_languages():
        s = SOURCE_LANG_VAR.get()
        t = TARGET_LANG_VAR.get()
        if s != "自动检测" and t != "自动切换(中英)":
            SOURCE_LANG_VAR.set(t)
            TARGET_LANG_VAR.set(s)

    tk.Button(lang_row, text="⇅ 互换", command=swap_languages, padx=5).pack(side="left", padx=10)

    tk.Label(frame, text="输入 (Ctrl+Enter 翻译)").pack(anchor="w", pady=(8, 0))
    INPUT_TEXT = scrolledtext.ScrolledText(frame, height=10, wrap=tk.WORD)
    INPUT_TEXT.pack(fill="both", expand=True)
    INPUT_TEXT.bind("<Control-Return>", lambda event: run_translation())

    button_row = tk.Frame(frame)
    button_row.pack(fill="x", pady=6)
    TRANSLATE_BUTTON = tk.Button(button_row, text="立即翻译", command=run_translation, bg="#0078d4", fg="white", padx=10)
    TRANSLATE_BUTTON.pack(side="left")
    COPY_BUTTON = tk.Button(button_row, text="复制结果", command=copy_output)
    COPY_BUTTON.pack(side="left", padx=8)
    tk.Button(button_row, text="朗读原文", command=speak_input).pack(side="left", padx=8)
    tk.Button(button_row, text="朗读译文", command=speak_output).pack(side="left", padx=8)
    
    tk.Label(frame, text="输出").pack(anchor="w")
    OUTPUT_TEXT = scrolledtext.ScrolledText(frame, height=12, wrap=tk.WORD)
    OUTPUT_TEXT.config(state="disabled")
    OUTPUT_TEXT.pack(fill="both", expand=True)

    status_bar = tk.Label(frame, textvariable=STATUS_VAR, anchor="w", fg="#666")
    status_bar.pack(fill="x", pady=(6, 0))

    WINDOW.after(0, WINDOW.focus_force)
    refresh_models()


class TrayThread(threading.Thread):
    def __init__(self, event_queue):
        super().__init__(daemon=True)
        self.event_queue = event_queue
        self.hwnd = None

    def run(self):
        message_map = {
            win32con.WM_COMMAND: self.on_command,
            win32con.WM_DESTROY: self.on_destroy,
            win32con.WM_USER + 20: self.on_notify,
        }
        wc = win32gui.WNDCLASS()
        wc.hInstance = win32api.GetModuleHandle(None)
        wc.lpszClassName = "AIQwenTranslatorTray"
        wc.lpfnWndProc = message_map
        try:
            win32gui.RegisterClass(wc)
        except win32gui.error:
            pass

        self.hwnd = win32gui.CreateWindow(wc.lpszClassName, "AIQwenTranslator", 0, 0, 0, 0, 0, 0, 0, wc.hInstance, None)
        icon = win32gui.LoadIcon(0, win32con.IDI_APPLICATION)
        nid = (self.hwnd, 0, win32gui.NIF_ICON | win32gui.NIF_MESSAGE | win32gui.NIF_TIP, win32con.WM_USER + 20, icon, "AI Qwen 翻译助手")
        win32gui.Shell_NotifyIcon(win32gui.NIM_ADD, nid)
        win32gui.PumpMessages()

    def stop(self):
        if self.hwnd is not None:
            win32gui.PostMessage(self.hwnd, win32con.WM_DESTROY, 0, 0)

    def on_notify(self, hwnd, msg, wparam, lparam):
        if lparam == win32con.WM_LBUTTONUP:
            self.event_queue.put("show")
        elif lparam == win32con.WM_RBUTTONUP:
            self.show_menu()
        return True

    def show_menu(self):
        menu = win32gui.CreatePopupMenu()
        win32gui.AppendMenu(menu, win32con.MF_STRING, 1, "显示面板")
        win32gui.AppendMenu(menu, win32con.MF_STRING, 2, "完全退出")
        x, y = win32gui.GetCursorPos()
        win32gui.SetForegroundWindow(self.hwnd)
        cmd = win32gui.TrackPopupMenu(menu, win32con.TPM_LEFTALIGN | win32con.TPM_RIGHTBUTTON | win32con.TPM_RETURNCMD, x, y, 0, self.hwnd, None)
        if cmd == 1:
            self.event_queue.put("show")
        elif cmd == 2:
            self.event_queue.put("quit")
        win32gui.PostMessage(self.hwnd, win32con.WM_NULL, 0, 0)

    def on_command(self, hwnd, msg, wparam, lparam):
        return True

    def on_destroy(self, hwnd, msg, wparam, lparam):
        win32gui.Shell_NotifyIcon(win32gui.NIM_DELETE, (self.hwnd, 0))
        win32gui.PostQuitMessage(0)
        return True


def process_events():
    try:
        while True:
            event = EVENT_QUEUE.get_nowait()
            if event == "show":
                show_window()
            elif event == "quit":
                if TRAY is not None:
                    TRAY.stop()
                if WINDOW is not None:
                    WINDOW.destroy()
    except queue.Empty:
        pass
    if WINDOW is not None:
        WINDOW.after(100, process_events)


def start_app():
    global TRAY
    build_window()
    show_window()
    TRAY = TrayThread(EVENT_QUEUE)
    TRAY.start()
    process_events()
    WINDOW.mainloop()


if __name__ == "__main__":
    start_app()
