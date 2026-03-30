"""Google Gemini provider"""

import json
from typing import Union, List, AsyncGenerator, Dict, Any
import httpx
from .base import BaseLLM, Message, ChatResponse, Optional


class ChatGemini(BaseLLM):
    """Google Gemini provider"""
    
    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        api_key: Optional[str] = None,
        timeout: int = 60,
        **kwargs
    ):
        super().__init__(model, api_key, **kwargs)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)
    
    def _format_gemini_payload(self, messages: List[dict]) -> Dict[str, Any]:
        """Format messages for Gemini API"""
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
            else:
                gemini_role = "user" if role == "user" else "model"
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}]
                })
        
        payload = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        
        return payload
    
    def generate(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> ChatResponse:
        """Generate response from Gemini"""
        messages = self._format_messages(prompt)
        payload = self._format_gemini_payload(messages)
        payload.update(kwargs)
        
        response = self.client.post(
            f"{self.base_url}/models/{self.model}:generateContent",
            params={"key": self.api_key},
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        
        return ChatResponse(
            content=content,
            model=self.model,
            raw_response=data,
            finish_reason=data["candidates"][0].get("finishReason")
        )
    
    async def agenerate(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> ChatResponse:
        """Async generate response from Gemini"""
        messages = self._format_messages(prompt)
        payload = self._format_gemini_payload(messages)
        payload.update(kwargs)
        
        response = await self.async_client.post(
            f"{self.base_url}/models/{self.model}:generateContent",
            params={"key": self.api_key},
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        
        return ChatResponse(
            content=content,
            model=self.model,
            raw_response=data,
            finish_reason=data["candidates"][0].get("finishReason")
        )
    
    def stream(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ):
        """Stream response from Gemini"""
        messages = self._format_messages(prompt)
        payload = self._format_gemini_payload(messages)
        payload["stream"] = True
        payload.update(kwargs)
        
        with self.client.stream(
            "POST",
            f"{self.base_url}/models/{self.model}:streamGenerateContent",
            params={"key": self.api_key},
            json=payload
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if data.get("candidates"):
                            content = data["candidates"][0]["content"]["parts"][0]["text"]
                            yield content
                    except:
                        continue
    
    async def astream(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Async stream from Gemini"""
        messages = self._format_messages(prompt)
        payload = self._format_gemini_payload(messages)
        payload["stream"] = True
        payload.update(kwargs)
        
        async with self.async_client.stream(
            "POST",
            f"{self.base_url}/models/{self.model}:streamGenerateContent",
            params={"key": self.api_key},
            json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if data.get("candidates"):
                            content = data["candidates"][0]["content"]["parts"][0]["text"]
                            yield content
                    except:
                        continue
    
    def close(self):
        """Close client connections"""
        self.client.close()
    
    def _get_env_var_name(self) -> str:
        return "GEMINI_API_KEY"