"""Anthropic Claude provider"""

from typing import Union, List, AsyncGenerator, Dict, Any
import json
import httpx
from .base import BaseLLM, Message, ChatResponse, Optional


class ChatClaude(BaseLLM):
    """Anthropic Claude provider"""
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        timeout: int = 60,
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.base_url = "https://api.anthropic.com/v1"
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
    
    def _format_claude_messages(self, messages: List[dict]) -> List[dict]:
        """Format messages for Claude API"""
        formatted = []
        system_message = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                formatted.append(msg)
        
        return formatted, system_message
    
    def generate(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> ChatResponse:
        """Generate response from Claude"""
        messages = self._format_messages(prompt)
        formatted_messages, system_message = self._format_claude_messages(messages)
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            **self.config,
            **kwargs
        }
        
        if system_message:
            payload["system"] = system_message
        
        response = self.client.post(
            f"{self.base_url}/messages",
            headers=self._get_headers(),
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        return ChatResponse(
            content=data["content"][0]["text"],
            model=data["model"],
            usage=data.get("usage"),
            raw_response=data,
            finish_reason=data.get("stop_reason")
        )
    
    async def agenerate(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> ChatResponse:
        """Async generate response from Claude"""
        messages = self._format_messages(prompt)
        formatted_messages, system_message = self._format_claude_messages(messages)
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            **self.config,
            **kwargs
        }
        
        if system_message:
            payload["system"] = system_message
        
        response = await self.async_client.post(
            f"{self.base_url}/messages",
            headers=self._get_headers(),
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        return ChatResponse(
            content=data["content"][0]["text"],
            model=data["model"],
            usage=data.get("usage"),
            raw_response=data,
            finish_reason=data.get("stop_reason")
        )
    
    def stream(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ):
        """Stream response from Claude"""
        messages = self._format_messages(prompt)
        formatted_messages, system_message = self._format_claude_messages(messages)
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            "stream": True,
            **self.config,
            **kwargs
        }
        
        if system_message:
            payload["system"] = system_message
        
        with self.client.stream(
            "POST",
            f"{self.base_url}/messages",
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
                        if data.get("type") == "content_block_delta":
                            yield data["delta"]["text"]
                    except:
                        continue
    
    async def astream(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Async stream from Claude"""
        messages = self._format_messages(prompt)
        formatted_messages, system_message = self._format_claude_messages(messages)
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": kwargs.pop("max_tokens", 1024),
            "stream": True,
            **self.config,
            **kwargs
        }
        
        if system_message:
            payload["system"] = system_message
        
        async with self.async_client.stream(
            "POST",
            f"{self.base_url}/messages",
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
                        if data.get("type") == "content_block_delta":
                            yield data["delta"]["text"]
                    except:
                        continue
    
    def close(self):
        """Close client connections"""
        self.client.close()
    
    def _get_env_var_name(self) -> str:
        return "ANTHROPIC_API_KEY"