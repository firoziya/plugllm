"""Cohere provider"""

from typing import Union, List, AsyncGenerator, Dict, Any
import json
import httpx
from .base import BaseLLM, Message, ChatResponse, Optional


class ChatCohere(BaseLLM):
    """Cohere provider"""
    
    def __init__(
        self,
        model: str = "command-a-03-2025",
        api_key: Optional[str] = None,
        timeout: int = 60,
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.base_url = "https://api.cohere.com/v2"
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
        """Generate response from Cohere"""
        messages = self._format_messages(prompt)
        
        payload = {
            "model": self.model,
            "messages": messages,
            **self.config,
            **kwargs
        }
        
        response = self.client.post(
            f"{self.base_url}/chat",
            headers=self._get_headers(),
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        return ChatResponse(
            content=data["message"]["content"][0]["text"],
            model=self.model,
            usage=data.get("usage"),
            raw_response=data,
            finish_reason=data.get("finish_reason")
        )
    
    async def agenerate(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> ChatResponse:
        """Async generate response from Cohere"""
        messages = self._format_messages(prompt)
        
        payload = {
            "model": self.model,
            "messages": messages,
            **self.config,
            **kwargs
        }
        
        response = await self.async_client.post(
            f"{self.base_url}/chat",
            headers=self._get_headers(),
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        return ChatResponse(
            content=data["message"]["content"][0]["text"],
            model=self.model,
            usage=data.get("usage"),
            raw_response=data,
            finish_reason=data.get("finish_reason")
        )
    
    def stream(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ):
        """Stream response from Cohere"""
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
            f"{self.base_url}/chat",
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
                        if data.get("type") == "text-generation":
                            yield data["text"]
                    except:
                        continue
    
    async def astream(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Async stream from Cohere"""
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
            f"{self.base_url}/chat",
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
                        if data.get("type") == "text-generation":
                            yield data["text"]
                    except:
                        continue
    
    def close(self):
        """Close client connections"""
        self.client.close()
    
    def _get_env_var_name(self) -> str:
        return "CO_API_KEY"