from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator
from pydantic import BaseModel

class ITool(ABC):
    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """返回 OpenAI/Qwen 格式的函数声明"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        pass

class ILLMService(ABC):
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None, max_new_tokens: int = 512) -> str:
        pass
    
    @abstractmethod
    def generate_stream(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None, max_new_tokens: int = 512) -> Generator[str, None, None]:
        pass

class IVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[str]):
        """索引文档"""
        pass

    @abstractmethod
    def query(self, text: str, top_k: int = 3) -> List[str]:
        """检索相关片段"""
        pass
