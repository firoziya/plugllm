"""PlugLLM - Unified LLM API Interface"""

# v1 imports
from plugllm.v1 import config, generate, chat, reset_chat

# v2 imports
from plugllm.base import BaseLLM, Message, ChatResponse
from plugllm.openai import ChatOpenAI
from plugllm.gemini import ChatGemini
from plugllm.groq import ChatGroq
from plugllm.claude import ChatClaude
from plugllm.grok import ChatGrok
from plugllm.sarvamai import ChatSarvamAI
from plugllm.mistral import ChatMistral
from plugllm.llama import ChatLlama
from plugllm.deepseek import ChatDeepSeek
from plugllm.qwen import ChatQwen
from plugllm.kimi import ChatKimi
from plugllm.cohere import ChatCohere
from plugllm.ollama import ChatOllama
from plugllm.factory import LLMFactory


__version__ = "2.0.0"
__all__ = [
    "config", 
    "generate", 
    "chat", 
    "reset_chat",
    "BaseLLM",
    "Message",
    "ChatResponse",
    "ChatOpenAI",
    "ChatGemini",
    "ChatGroq",
    "ChatClaude",
    "ChatGrok",
    "ChatSarvamAI",
    "ChatMistral",
    "ChatLlama",
    "ChatDeepSeek",
    "ChatQwen",
    "ChatKimi",
    "ChatCohere",
    "ChatOllama",
    "LLMFactory",
]