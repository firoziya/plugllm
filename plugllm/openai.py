"""OpenAI provider implementation"""

import json
from typing import Union, List, Optional, Dict, Any, AsyncGenerator
import httpx
from .base import BaseLLM, Message, ChatResponse


class ChatOpenAI(BaseLLM):
    """OpenAI ChatGPT provider"""
    
    def __init__(
        self,
        model: str = "gpt-5.4",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def generate(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> ChatResponse:
        """Generate chat completion"""
        messages = self._format_messages(prompt)
        
        payload = {
            "model": self.model,
            "messages": messages,
            **self.config,
            **kwargs
        }
        
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self._get_headers(),
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        return ChatResponse(
            content=data["choices"][0]["message"]["content"],
            model=data["model"],
            usage=data.get("usage"),
            raw_response=data,
            finish_reason=data["choices"][0].get("finish_reason")
        )
    
    async def agenerate(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> ChatResponse:
        """Async generate chat completion"""
        messages = self._format_messages(prompt)
        
        payload = {
            "model": self.model,
            "messages": messages,
            **self.config,
            **kwargs
        }
        
        response = await self.async_client.post(
            f"{self.base_url}/chat/completions",
            headers=self._get_headers(),
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        return ChatResponse(
            content=data["choices"][0]["message"]["content"],
            model=data["model"],
            usage=data.get("usage"),
            raw_response=data,
            finish_reason=data["choices"][0].get("finish_reason")
        )
    
    def stream(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ):
        """Stream response synchronously"""
        messages = self._format_messages(prompt)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self.config,
            **kwargs
        }
        
        with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._get_headers(),
            json=payload
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    if line == "data: [DONE]":
                        break
                    try:
                        data = json.loads(line[6:])
                        if data["choices"][0].get("delta", {}).get("content"):
                            yield data["choices"][0]["delta"]["content"]
                    except:
                        continue
    
    async def astream(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response asynchronously"""
        messages = self._format_messages(prompt)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self.config,
            **kwargs
        }
        
        async with self.async_client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._get_headers(),
            json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    if line == "data: [DONE]":
                        break
                    try:
                        data = json.loads(line[6:])
                        if data["choices"][0].get("delta", {}).get("content"):
                            yield data["choices"][0]["delta"]["content"]
                    except:
                        continue
    
    def close(self):
        """Close client connections"""
        self.client.close()
    
    def _get_env_var_name(self) -> str:
        return "OPENAI_API_KEY"