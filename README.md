# 🔌 PlugLLM - Unified LLM API Interface

[![PyPI Downloads](https://static.pepy.tech/badge/plugllm)](https://pepy.tech/projects/plugllm)
[![PyPI Version](https://img.shields.io/pypi/v/plugllm.svg)](https://pypi.org/project/plugllm/)
[![Python Version](https://img.shields.io/pypi/pyversions/plugllm.svg)](https://pypi.org/project/plugllm/)
[![License](https://img.shields.io/github/license/firoziya/plugllm)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/firoziya/plugllm?style=social)](https://github.com/firoziya/plugllm)

**PlugLLM** is a powerful, unified Python package that provides a consistent interface for 13+ Large Language Model (LLM) providers. Stop dealing with different SDKs and API formats - use one simple API for all your LLM needs.

<div style="text-align: center; margin: 20px 0; background-color: #000; padding: 20px; border-radius: 5px;">

![Typing SVG](https://readme-typing-svg.demolab.com?font=Source+Code+Pro&size=38&pause=800&color=39FF14&background=000000&center=true&vCenter=true&width=750&lines=%24+pip+install+plugllm;Installing...;Done+%E2%9C%85)

</div>

## ✨ Key Features

- 🔌 **Unified API** - Same interface for all 13+ providers
- 🧠 **Context Memory** - Built-in conversation memory with deque (configurable up to 10+ messages)
- 💬 **Multiple Methods** - `generate()`, `chat()`, `ask()`, `stream()` for every use case
- 🔄 **Async Support** - Full async/await functionality
- 📡 **Streaming** - Real-time response streaming
- 🏭 **Factory Pattern** - Easy provider instantiation with LLMFactory
- 🔐 **Environment Variables** - Automatic API key detection
- 💪 **Type Hints** - Full type annotation support
- 🎯 **Production Ready** - Comprehensive error handling and timeouts
- 🚀 **No Vendor Lock-in** - Switch providers without code changes

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install plugllm
```

### From Source

```bash
git clone https://github.com/firoziya/plugllm.git
cd plugllm
pip install -e .
```

### Development Installation

```bash
pip install plugllm[dev]
```

## 🚀 Quick Start

### Method 1: Direct Provider Usage

```python
from plugllm import ChatOpenAI, Message

# Initialize your LLM
llm = ChatOpenAI(api_key="your-key", model="gpt-4")

# Simple generation
response = llm.generate("What is Python?")
print(response)

# With message history
messages = [
    Message.system("You are a helpful assistant"),
    Message.user("What is machine learning?")
]
response = llm.generate(messages)
print(response)
```

### Method 2: Using Factory Pattern

```python
from plugllm import LLMFactory

# Create any provider with one line
llm = LLMFactory.create("groq", api_key="your-key", model="openai/gpt-oss-20b")
response = llm.generate("Explain AI")
print(response)
```

### Method 3: Ask Method (Simplest)

```python
from plugllm import ChatGroq

llm = ChatGroq(api_key="your-key", model="openai/gpt-oss-20b")

# Simple ask
response = llm.ask("What is Python?")

# With system prompt
response = llm.ask(
    "What is Python?",
    system_prompt="You are a beginner-friendly teacher. Explain simply."
)
print(response)
```

## 💬 Chat with Context Memory

PlugLLM includes built-in conversation memory using deque for natural conversations:

```python
from plugllm import ChatOpenAI

llm = ChatOpenAI(api_key="your-key", model="gpt-4", max_history=10)

# Have a conversation - it remembers context!
response1 = llm.chat("My name is Alice")
print(f"Assistant: {response1}")

response2 = llm.chat("What's my name?")  # Remembers "Alice"
print(f"Assistant: {response2}")

# Multiple independent sessions
llm.chat("I like Python", session_id="user1")
llm.chat("I like Java", session_id="user2")

# Get conversation history
history = llm.get_conversation_history("user1")
print(f"Session history: {history}")
```

## 🌊 Streaming Responses

```python
from plugllm import ChatGroq

llm = ChatGroq(api_key="your-key", model="openai/gpt-oss-20b")

# Synchronous streaming
for chunk in llm.stream("Tell me a story"):
    print(chunk, end="", flush=True)

# Streaming with ask method
for chunk in llm.ask_stream("Count from 1 to 5"):
    print(chunk, end="")

# Async streaming
async for chunk in llm.astream("Tell me a joke"):
    print(chunk, end="")
```

## 🔄 Async/Await Support

```python
import asyncio
from plugllm import ChatGemini

async def main():
    llm = ChatGemini(api_key="your-key", model="gemini-2.5-flash")
    
    # Async generate
    response = await llm.agenerate("What is async programming?")
    print(response)
    
    # Async chat with memory
    response = await llm.achat("Remember this: 42", session_id="test")
    
    # Async streaming
    async for chunk in llm.astream("Tell me a secret"):
        print(chunk)

asyncio.run(main())
```

## 🎯 Advanced Features

### Fluent Interface for Prompt Engineering

```python
from plugllm import ChatOpenAI

llm = ChatOpenAI(api_key="your-key", model="gpt-4")

# Method chaining for clean code
response = (llm
    .with_system("You are a helpful math tutor")
    .with_user("What is the square root of 144?")
    .with_temperature(0.5)
    .with_max_tokens(100)
    .call())

print(response)
```

### Multiple Session Management

```python
from plugllm import ChatGroq

llm = ChatGroq(api_key="your-key", model="openai/gpt-oss-20b")

# Session 1: Technical discussion
llm.chat("What is Python?", session_id="tech")
llm.chat("What are its main features?", session_id="tech")

# Session 2: Casual chat
llm.chat("I like pizza", session_id="casual")
llm.chat("What do I like?", session_id="casual")

# Manage sessions
history = llm.get_conversation_history("tech")
llm.clear_conversation("casual")
llm.reset_conversation("tech")  # Complete reset
```

### System Message Management

```python
from plugllm import ChatGemini

llm = ChatGemini(api_key="your-key", model="gemini-2.5-flash")

# Set system message for a session
llm.set_system_message(
    "You are a pirate. Always respond like a pirate.",
    session_id="pirate"
)

response = llm.chat("What is your favorite food?", session_id="pirate")
print(response)  # Will respond in pirate style
```

## 🌐 Supported Providers

| Provider | Class | Default Model | API Key Env Var |
|----------|-------|---------------|-----------------|
| **OpenAI** | `ChatOpenAI` | gpt-5.4 | `OPENAI_API_KEY` |
| **Google Gemini** | `ChatGemini` | gemini-3-flash-preview | `GEMINI_API_KEY` |
| **Groq** | `ChatGroq` | llama-3.3-70b-versatile | `GROQ_API_KEY` |
| **Anthropic Claude** | `ChatClaude` | claude-sonnet-4-6 | `ANTHROPIC_API_KEY` |
| **xAI Grok** | `ChatGrok` | grok-4.20-reasoning | `XAI_API_KEY` |
| **Mistral AI** | `ChatMistral` | mistral-large-latest | `MISTRAL_API_KEY` |
| **Meta Llama** | `ChatLlama` | Llama-4-Maverick-17B-128E-Instruct-FP8 | `LLAMA_API_KEY` |
| **DeepSeek** | `ChatDeepSeek` | deepseek-chat | `DEEPSEEK_API_KEY` |
| **Alibaba Qwen** | `ChatQwen` | qwen3.5-plus | `DASHSCOPE_API_KEY` |
| **Moonshot Kimi** | `ChatKimi` | kimi-k2.5 | `MOONSHOT_API_KEY` |
| **Cohere** | `ChatCohere` | command-a-03-2025 | `CO_API_KEY` |
| **SarvamAI** | `ChatSarvamAI` | sarvam-105b | `SARVAM_API_KEY` |
| **Ollama (Local)** | `ChatOllama` | gemma3 | No API key needed |

## 🔧 Configuration

### Method 1: Direct Configuration

```python
from plugllm import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    api_key="your-key",
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9
)
```

### Method 2: Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza...
GROQ_API_KEY=gsk_...
ANTHROPIC_API_KEY=sk-ant-...
```

```python
from plugllm import ChatGroq
import os

# Automatically reads from environment
llm = ChatGroq(model="openai/gpt-oss-20b")
```

### Method 3: Using Factory

```python
from plugllm import LLMFactory

llm = LLMFactory.create(
    provider="claude",
    model="claude-sonnet-4-6",
    api_key="your-key",
    temperature=0.5
)
```

## 📊 Usage Examples

### Example 1: Building a Chatbot

```python
from plugllm import ChatOpenAI, Message

class ChatBot:
    def __init__(self, api_key):
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-5.4", max_history=20)
        self.session_id = "chatbot_session"
        self.llm.set_system_message(
            "You are a friendly AI assistant. Be helpful and concise.",
            session_id=self.session_id
        )
    
    def chat(self, user_message):
        response = self.llm.chat(user_message, session_id=self.session_id)
        return response.content
    
    def get_history(self):
        return self.llm.get_conversation_history(self.session_id)

# Use the chatbot
bot = ChatBot("your-key")
print(bot.chat("Hello!"))
print(bot.chat("What's my name? I'm John"))
print(bot.chat("What's my name?"))  # Remembers "John"
```

### Example 2: Multi-Provider Comparison

```python
from plugllm import ChatOpenAI, ChatGemini, ChatGroq

providers = {
    "OpenAI": ChatOpenAI(api_key="key1", model="gpt-5.4"),
    "Gemini": ChatGemini(api_key="key2", model="gemini-3-flash-preview"),
    "Groq": ChatGroq(api_key="key3", model="llama-3.3-70b-versatile")
}

prompt = "Explain quantum computing in one paragraph"

for name, llm in providers.items():
    response = llm.ask(prompt, max_tokens=150)
    print(f"\n{name}:\n{response.content[:200]}...")
```

### Example 3: Content Summarizer with Streaming

```python
from plugllm import ChatMistral

llm = ChatMistral(api_key="your-key", model="mistral-large-latest")

def summarize_streaming(text):
    prompt = f"Summarize this text in 3 bullet points:\n\n{text}"
    
    print("Summary:", end=" ")
    for chunk in llm.ask_stream(prompt, temperature=0.3):
        print(chunk, end="", flush=True)
    print()

long_text = "Your long article text here..."
summarize_streaming(long_text)
```

## 🛠️ Advanced Configuration

### Custom Timeouts and Retries

```python
from plugllm import ChatOpenAI

llm = ChatOpenAI(
    api_key="your-key",
    model="gpt-5.4",
    timeout=120,  # 2 minutes timeout
    max_retries=3  # Retry failed requests
)
```

### Context Window Management

```python
from plugllm import ChatGroq

# Limit conversation history to prevent token overflow
llm = ChatGroq(
    api_key="your-key",
    model="llama-3.3-70b-versatile",
    max_history=5  # Keep only last 5 messages
)
```

## 🐛 Error Handling

```python
from plugllm import ChatOpenAI
from plugllm.types import AuthenticationError, RateLimitError
from httpx import HTTPStatusError

llm = ChatOpenAI(api_key="your-key", model="gpt-5.4")

try:
    response = llm.ask("Hello")
    print(response)
    
except AuthenticationError:
    print("Invalid API key. Please check your credentials.")
    
except RateLimitError:
    print("Rate limit exceeded. Please wait and try again.")
    
except HTTPStatusError as e:
    print(f"HTTP Error: {e.response.status_code}")
    print(f"Details: {e.response.text}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 📈 Performance Tips

1. **Reuse LLM instances** instead of creating new ones for each request
2. **Use appropriate models** for your use case (smaller models for simple tasks)
3. **Limit conversation history** with `max_history` parameter
4. **Use streaming** for long responses to improve perceived performance
5. **Implement caching** for frequently asked questions

```python
# Good: Reuse instance
llm = ChatOpenAI(api_key="key", model="gpt-5.4")
for prompt in prompts:
    response = llm.ask(prompt)

# Bad: Create new instance each time
for prompt in prompts:
    llm = ChatOpenAI(api_key="key", model="gpt-5.4")
    response = llm.ask(prompt)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific provider tests
pytest tests/test_gemini_groq.py -v

# Run with coverage
pytest --cov=plugllm tests/
```

## 📚 API Reference

### Core Classes

#### `BaseLLM`
Abstract base class for all providers.

#### `ChatResponse`
Unified response object with properties:
- `content`: The generated text
- `model`: Model used
- `usage`: Token usage statistics
- `raw_response`: Original API response

#### `Message`
Message structure for conversations:
- `Message.user(content)`: Create user message
- `Message.assistant(content)`: Create assistant message  
- `Message.system(content)`: Create system message

### Key Methods

| Method | Description | Async Version |
|--------|-------------|---------------|
| `generate()` | Basic text generation | `agenerate()` |
| `stream()` | Stream responses | `astream()` |
| `chat()` | Context-aware conversation | `achat()` |
| `ask()` | Simple Q&A with optional system prompt | `aask()` |
| `ask_stream()` | Streaming Q&A | `aask_stream()` |

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Run tests (`pytest tests/`)
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Yash Kumar Firoziya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions...
```

## 👨‍💻 Author

**Yash Kumar Firoziya**
- GitHub: [@firoziya](https://github.com/firoziya)
- Email: [ykfiroziya@gmail.com](mailto:ykfiroziya@gmail.com)

## 🙏 Acknowledgments

- All LLM providers for their amazing APIs
- Open source community for inspiration
- Contributors and users for their support

## 📖 More Resources

- [GitHub Repository](https://github.com/firoziya/plugllm)
- [Issue Tracker](https://github.com/firoziya/plugllm/issues)
- [Examples Directory](https://github.com/firoziya/plugllm/tree/main/examples)
- [PyPI Page](https://pypi.org/project/plugllm/)

## ⭐ Show Your Support

If you find PlugLLM useful, please give it a star on GitHub! It helps others discover the project.

[![Stargazers repo roster for @firoziya/plugllm](https://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?format=notext&user=firoziya&repo=plugllm)](https://github.com/firoziya/plugllm/stargazers)

---

**Built with ❤️ for the Python AI community**

<div align="center">

[https://plugllm.firoziyash.life](https://plugllm.firoziyash.life)

</div>
