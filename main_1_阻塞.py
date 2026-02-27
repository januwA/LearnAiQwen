from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Any
from abc import ABC, abstractmethod

# --- Interfaces ---

class ILLMService(ABC):
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> str:
        pass

# --- Implementations ---

class QwenCoderService(ILLMService):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )

    def generate_response(self, messages: List[Dict[str, str]], max_new_tokens: int = 512) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        
        # Trim the input tokens from the output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# --- Application Layer ---

class ChatApp:
    def __init__(self, llm_service: ILLMService):
        self.llm_service = llm_service

    def run(self, prompt: str):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        print(f"User: {prompt}")
        print("Assistant: ", end="", flush=True)
        response = self.llm_service.generate_response(messages)
        print(response)

# --- Dependency Injection and Main ---

def main():
    # 指向你本地下载的模型文件夹路径
    local_model_path = r"d:\ajanuw\ai_qwen\qwen2.5-1.5b"
    
    # 依赖注入
    llm_service = QwenCoderService(local_model_path)
    app = ChatApp(llm_service)
    
    # 运行
    app.run("请用 Python 写一个快速排序算法。")

if __name__ == "__main__":
    main()
