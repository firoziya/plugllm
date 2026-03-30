"""Ollama local provider"""

from typing import Union, List, AsyncGenerator, Dict, Any
import json
import httpx
from .base import BaseLLM, Message, ChatResponse, Optional


class ChatOllama(BaseLLM):
    """Ollama local provider (no API key required)"""
    
    def __init__(
        self,
        model: str = "gemma3",
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)
    
    def _allow_no_api_key(self) -> bool:
        """Ollama doesn't require API key"""
        return True
    
    def _format_ollama_payload(self, prompt: Union[str, List[Message]], **kwargs) -> Dict[str, Any]:
        """Format payload for Ollama API"""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = self._format_messages(prompt)
        
        # Convert to Ollama format
        if len(messages) == 1 and messages[0]["role"] == "user":
            # Simple prompt format for generate endpoint
            return {
                "model": self.model,
                "prompt": messages[0]["content"],
                "stream": kwargs.get("stream", False),
                **self.config,
                **{k: v for k, v in kwargs.items() if k != "stream"}
            }
        else:
            # Chat format
            return {
                "model": self.model,
                "messages": messages,
                "stream": kwargs.get("stream", False),
                **self.config,
                **{k: v for k, v in kwargs.items() if k != "stream"}
            }
    
    def generate(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> ChatResponse:
        """Generate response from Ollama"""
        payload = self._format_ollama_payload(prompt, stream=False, **kwargs)
        
        # Determine which endpoint to use
        if "messages" in payload:
            endpoint = f"{self.base_url}/api/chat"
        else:
            endpoint = f"{self.base_url}/api/generate"
        
        response = self.client.post(
            endpoint,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        content = data.get("message", {}).get("content", "") if "message" in data else data.get("response", "")
        
        return ChatResponse(
            content=content,
            model=self.model,
            raw_response=data
        )
    
    async def agenerate(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> ChatResponse:
        """Async generate response from Ollama"""
        payload = self._format_ollama_payload(prompt, stream=False, **kwargs)
        
        # Determine which endpoint to use
        if "messages" in payload:
            endpoint = f"{self.base_url}/api/chat"
        else:
            endpoint = f"{self.base_url}/api/generate"
        
        response = await self.async_client.post(
            endpoint,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        content = data.get("message", {}).get("content", "") if "message" in data else data.get("response", "")
        
        return ChatResponse(
            content=content,
            model=self.model,
            raw_response=data
        )
    
    def stream(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ):
        """Stream response from Ollama"""
        payload = self._format_ollama_payload(prompt, stream=True, **kwargs)
        
        # Determine which endpoint to use
        if "messages" in payload:
            endpoint = f"{self.base_url}/api/chat"
        else:
            endpoint = f"{self.base_url}/api/generate"
        
        with self.client.stream(
            "POST",
            endpoint,
            json=payload
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                        elif "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except:
                        continue
    
    async def astream(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Async stream from Ollama"""
        payload = self._format_ollama_payload(prompt, stream=True, **kwargs)
        
        # Determine which endpoint to use
        if "messages" in payload:
            endpoint = f"{self.base_url}/api/chat"
        else:
            endpoint = f"{self.base_url}/api/generate"
        
        async with self.async_client.stream(
            "POST",
            endpoint,
            json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                        elif "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except:
                        continue
    
    def close(self):
        """Close client connections"""
        self.client.close()