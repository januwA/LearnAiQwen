from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
import torch
from typing import List, Dict, Any, Generator
from abc import ABC, abstractmethod
from threading import Thread
import os

# --- Interfaces ---

class ILLMService(ABC):
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> str:
        pass
    
    @abstractmethod
    def generate_stream(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> Generator[str, None, None]:
        """流式生成接口"""
        pass

# --- Implementations ---

class QwenService(ILLMService):
    def __init__(self, model_path: str, use_4bit: bool = True):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        
        print(f"🔄 正在加载模型: {model_path}")
        print(f"   4-bit 量化: {'启用' if use_4bit else '禁用'}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            padding_side="left"
        )
        
        # 如果没 pad_token，用 eos_token 代替
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 量化配置（节省显存）
        if use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # 减少 CPU 内存占用
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        
        self.model.eval()
        print(f"✅ 模型加载完成，设备: {next(self.model.parameters()).device}")

    def generate_response(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer(
            [text], 
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():  # 禁用梯度计算，节省显存
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # 去除输入部分
        input_length = model_inputs.input_ids.shape[1]
        generated_ids = generated_ids[:, input_length:]
        
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def generate_stream(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> Generator[str, None, None]:
        """流式生成"""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer(
            [text], 
            return_tensors="pt"
        ).to(self.model.device)
        
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        generation_kwargs = dict(
            input_ids=model_inputs.input_ids,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            yield text
        
        thread.join()

# --- Application Layer ---

class ChatApp:
    def __init__(self, llm_service: ILLMService):
        self.llm_service = llm_service

    def run(self, prompt: str, stream: bool = True):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        print(f"\n👤 User: {prompt}")
        print("🤖 Assistant: ", end="", flush=True)
        
        if stream and hasattr(self.llm_service, 'generate_stream'):
            for chunk in self.llm_service.generate_stream(messages):
                print(chunk, end="", flush=True)
        else:
            response = self.llm_service.generate_response(messages)
            print(response, end="")
        
        print("\n")

# --- Main ---

def main():
    # 模型路径
    local_model_path = r"d:\ajanuw\ai_qwen\qwen2.5-1.5b"
    
    # 检查显存
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 显存: {gpu_memory:.1f} GB")
        use_4bit = gpu_memory < 8  # 小于8GB自动启用4-bit
    else:
        print("⚠️  未检测到 GPU，将使用 CPU（极慢）")
        use_4bit = False
    
    try:
        # 依赖注入
        llm_service = QwenService(local_model_path, use_4bit=use_4bit)
        app = ChatApp(llm_service)
        
        # 交互式对话
        print("\n💡 输入 'exit' 退出\n")
        while True:
            user_input = input("👤 You: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            if user_input:
                app.run(user_input, stream=True)
                
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n可能的解决方案:")
        print("1. 检查模型路径是否正确")
        print("2. 安装依赖: uv add transformers torch accelerate bitsandbytes")
        print("3. 降低 max_new_tokens 或启用 4-bit 量化")

if __name__ == "__main__":
    main()
