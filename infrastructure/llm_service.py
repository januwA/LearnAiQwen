import os
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from typing import List, Dict, Any, Generator, Optional
from threading import Thread
from core.interfaces import ILLMService

class QwenService(ILLMService):
    """
    通用 Qwen 服务优化版
    针对消费级显卡（如 RTX 3060 6GB/8GB）进行了深度性能调优
    """
    def __init__(
        self, 
        model_path: str, 
        use_4bit: bool = True,
        max_gpu_memory: str = "4.8GB"  # 针对 6GB 显卡预留系统空间
    ):
        # 1. 硬件探测与最优精度选择
        self.is_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda else "cpu")
        
        # RTX 30/40 系推荐 bf16，但 3060 6GB 移动端用 fp16 往往显存更稳且速度持平
        self.compute_dtype = torch.float16 if self.is_cuda else torch.float32
        
        if self.is_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 [QwenService] 检测到 GPU: {gpu_name} | 显存上限: {max_gpu_memory}")
        
        # 2. 分词器加载
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 3. 核心加载配置
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": self.compute_dtype if not use_4bit else None,
            "max_memory": {0: max_gpu_memory} if self.is_cuda else None,
            "offload_folder": "./offload" if not self.is_cuda else None,
        }
        
        # 4. 自动选择最佳注意力机制 (Flash Attention 2 -> SDPA -> Default)
        if self.is_cuda:
            import importlib.util
            # 探测 flash_attn 包是否存在
            has_flash_attn = importlib.util.find_spec("flash_attn") is not None
            
            if has_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("⚡ [加速中] 环境已就绪，启用 Flash Attention 2")
            else:
                # 如果没装 Flash Attention，强制使用 PyTorch 原生内核 (SDPA)
                # 这在 PyTorch 2.x 中是“版本答案”，无需任何额外安装
                model_kwargs["attn_implementation"] = "sdpa"
                print("✨ [加速中] 采用 PyTorch 原生 SDPA 加速 (无需安装多余依赖)")
        
        # 5. 4bit 量化细节优化
        if use_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage=self.compute_dtype, # 减少量化转换损耗
            )
            model_kwargs["quantization_config"] = quant_config
            print(f"💎 开启 4bit 量化 (存储类型={self.compute_dtype})")
        
        # 6. 正式加载模型
        print(f"⏳ 正在加载模型: {os.path.basename(model_path)}...")
        start_t = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        print(f"✅ 加载成功! 耗时: {time.time() - start_t:.1f}s")
        
        # 7. 推理状态初始化
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def get_token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict[str, Any]]] = None, 
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7
    ) -> str:
        prompt_text = self.tokenizer.apply_chat_template(
            messages, 
            tools=tools, 
            add_generation_prompt=True, 
            tokenize=False
        )
        inputs = self.tokenizer(
            prompt_text, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).to(self.model.device)
        
        input_ids_len = inputs.input_ids.shape[1]
        
        start_t = time.time()
        with torch.no_grad():
            output = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=do_sample, 
                temperature=temperature if do_sample else 1.0,
                repetition_penalty=1.1,
                use_cache=True, # 确保开启 KV Cache
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        duration = time.time() - start_t
        new_tokens = output[0][input_ids_len:].shape[0]
        speed = new_tokens / duration if duration > 0 else 0
        
        print(f"📊 [推理完成] 耗时: {duration:.2f}s | 生成: {new_tokens} tokens | 速度: {speed:.1f} tok/s")
        
        return self.tokenizer.decode(output[0][input_ids_len:], skip_special_tokens=True)

    def generate_stream(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict[str, Any]]] = None, 
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        prompt_text = self.tokenizer.apply_chat_template(
            messages, 
            tools=tools, 
            add_generation_prompt=True, 
            tokenize=False
        )
        inputs = self.tokenizer(
            prompt_text, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).to(self.model.device)
        
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True,
            timeout=30.0 # 防止模型卡死的超时保护
        )
        
        generation_kwargs = dict(
            **inputs, 
            streamer=streamer, 
            max_new_tokens=max_new_tokens, 
            do_sample=do_sample, 
            temperature=temperature if do_sample else 1.0,
            repetition_penalty=1.1,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            yield text
            
        thread.join()

    def get_gpu_memory_info(self) -> Dict[str, float]:
        """监控实时显存开销"""
        if not self.is_cuda:
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "usage_percent": 0}
        
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        used = torch.cuda.memory_allocated(0) / 1024**3
        return {
            "total_gb": round(total, 2),
            "used_gb": round(used, 2),
            "free_gb": round(total - used, 2),
            "usage_percent": round(used / total * 100, 1)
        }
