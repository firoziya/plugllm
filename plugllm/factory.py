"""Factory pattern for creating LLM instances"""

from typing import Optional, Dict, Any, Type
from .base import BaseLLM
from .openai import ChatOpenAI
from .gemini import ChatGemini
from .groq import ChatGroq
from .claude import ChatClaude
from .grok import ChatGrok
from .sarvamai import ChatSarvamAI
from .mistral import ChatMistral
from .llama import ChatLlama
from .deepseek import ChatDeepSeek
from .qwen import ChatQwen
from .kimi import ChatKimi
from .cohere import ChatCohere
from .ollama import ChatOllama


class LLMFactory:
    """Factory to create LLM instances"""
    
    _providers: Dict[str, Type[BaseLLM]] = {
        "openai": ChatOpenAI,
        "gemini": ChatGemini,
        "groq": ChatGroq,
        "claude": ChatClaude,
        "grok": ChatGrok,
        "sarvamai": ChatSarvamAI,
        "mistral": ChatMistral,
        "llama": ChatLlama,
        "deepseek": ChatDeepSeek,
        "qwen": ChatQwen,
        "kimi": ChatKimi,
        "cohere": ChatCohere,
        "ollama": ChatOllama,
    }
    
    @classmethod
    def create(
        cls,
        provider: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """
        Create an LLM instance
        
        Args:
            provider: Provider name (openai, gemini, groq, etc.)
            model: Model name (uses provider default if None)
            api_key: API key (uses environment variable if None)
            **kwargs: Additional provider-specific arguments
        
        Returns:
            BaseLLM instance
        """
        provider = provider.lower()
        
        if provider not in cls._providers:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available providers: {list(cls._providers.keys())}"
            )
        
        llm_class = cls._providers[provider]
        
        # Get default model if not specified
        if model is None:
            model = cls._get_default_model(provider)
        
        return llm_class(model=model, api_key=api_key, **kwargs)
    
    @classmethod
    def _get_default_model(cls, provider: str) -> str:
        """Get default model for provider"""
        defaults = {
            "openai": "gpt-5.4",
            "gemini": "gemini-3-flash-preview",
            "groq": "llama-3.3-70b-versatile",
            "claude": "claude-sonnet-4-6",
            "grok": "grok-4.20-reasoning",
            "sarvamai": "sarvam-105b",
            "mistral": "mistral-large-latest",
            "llama": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "deepseek": "deepseek-chat",
            "qwen": "qwen3.5-plus",
            "kimi": "kimi-k2.5",
            "cohere": "command-a-03-2025",
            "ollama": "gemma3",
        }
        return defaults.get(provider, "unknown")
    
    @classmethod
    def register_provider(cls, name: str, llm_class: Type[BaseLLM]):
        """Register a custom provider"""
        cls._providers[name.lower()] = llm_class
    
    @classmethod
    def list_providers(cls) -> list:
        """List all available providers"""
        return list(cls._providers.keys())