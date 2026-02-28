import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from typing import List, Dict, Any, Generator
from threading import Thread
from core.interfaces import ILLMService

class QwenService(ILLMService):
    def __init__(self, model_path: str, use_4bit: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if use_4bit:
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
            self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quant_config, device_map="auto", trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None, max_new_tokens: int = 512) -> str:
        prompt_text = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
        return self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def generate_stream(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None, max_new_tokens: int = 512) -> Generator[str, None, None]:
        prompt_text = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(target=self.model.generate, kwargs=dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7))
        thread.start()
        for text in streamer: yield text
        thread.join()
