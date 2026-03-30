"""Base classes for LLM providers with enhanced prompt methods"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, AsyncGenerator, Deque
from dataclasses import dataclass, field
from collections import deque
import os


@dataclass
class Message:
    """Message structure for chat completion"""
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}
    
    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)
    
    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role="assistant", content=content)
    
    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)


@dataclass
class ChatResponse:
    """Unified response structure"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None
    finish_reason: Optional[str] = None
    
    def __str__(self) -> str:
        return self.content
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
        }


@dataclass
class ConversationContext:
    """Manages conversation context and memory"""
    messages: Deque[Message] = field(default_factory=lambda: deque(maxlen=10))
    system_message: Optional[Message] = None
    max_history: int = 10
    
    def add_message(self, message: Message):
        """Add a message to conversation history"""
        self.messages.append(message)
    
    def add_user_message(self, content: str):
        """Add user message"""
        self.add_message(Message.user(content))
    
    def add_assistant_message(self, content: str):
        """Add assistant message"""
        self.add_message(Message.assistant(content))
    
    def get_conversation(self) -> List[Message]:
        """Get full conversation including system message"""
        conversation = []
        if self.system_message:
            conversation.append(self.system_message)
        conversation.extend(list(self.messages))
        return conversation
    
    def clear(self):
        """Clear conversation history (keeps system message)"""
        self.messages.clear()
    
    def reset(self):
        """Reset entire context including system message"""
        self.messages.clear()
        self.system_message = None
    
    def set_system_message(self, content: str):
        """Set system message"""
        self.system_message = Message.system(content)
    
    def get_history_length(self) -> int:
        """Get number of messages in history"""
        return len(self.messages)
    
    def truncate(self, max_tokens: int, token_count_func: callable):
        """Truncate conversation based on token count"""
        while self.messages and token_count_func(self.get_conversation()) > max_tokens:
            if self.messages:
                self.messages.popleft()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "system_message": self.system_message.to_dict() if self.system_message else None,
            "max_history": self.max_history
        }


class BaseLLM(ABC):
    """Abstract base class for all LLM providers with enhanced prompt methods"""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        max_history: int = 10,
        **kwargs
    ):
        self.model = model
        self.api_key = api_key or os.getenv(self._get_env_var_name())
        self.config = kwargs
        self.max_history = max_history
        self._contexts: Dict[str, ConversationContext] = {}
        self._validate_config()
    
    def _get_env_var_name(self) -> str:
        """Get environment variable name for API key"""
        return f"{self.__class__.__name__.upper()}_API_KEY"
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if not self.model:
            raise ValueError("Model name is required")
        if not self.api_key and not self._allow_no_api_key():
            raise ValueError(f"API key is required for {self.__class__.__name__}")
    
    def _allow_no_api_key(self) -> bool:
        """Override for providers that don't require API key (e.g., Ollama)"""
        return False
    
    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> ChatResponse:
        """Generate response synchronously"""
        pass
    
    @abstractmethod
    async def agenerate(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> ChatResponse:
        """Generate response asynchronously"""
        pass
    
    @abstractmethod
    def stream(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> Any:
        """Stream response synchronously"""
        pass
    
    @abstractmethod
    async def astream(
        self,
        prompt: Union[str, List[Message]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream response asynchronously"""
        pass
    
    # ============================================
    # NEW METHODS FOR SYSTEM AND USER PROMPTS
    # ============================================
    
    def with_system(self, system_prompt: str) -> 'BaseLLM':
        """
        Set system prompt for the current call (fluent interface)
        
        Args:
            system_prompt: System prompt to use
        
        Returns:
            Self for method chaining
        """
        self._temp_system_prompt = system_prompt
        return self
    
    def with_user(self, user_prompt: str) -> 'BaseLLM':
        """
        Set user prompt for the current call (fluent interface)
        
        Args:
            user_prompt: User prompt to use
        
        Returns:
            Self for method chaining
        """
        self._temp_user_prompt = user_prompt
        return self
    
    def with_assistant(self, assistant_prompt: str) -> 'BaseLLM':
        """
        Set assistant prompt for context (fluent interface)
        
        Args:
            assistant_prompt: Assistant prompt to use as context
        
        Returns:
            Self for method chaining
        """
        self._temp_assistant_prompt = assistant_prompt
        return self
    
    def with_temperature(self, temperature: float) -> 'BaseLLM':
        """
        Set temperature for the current call
        
        Args:
            temperature: Temperature value (0-2)
        
        Returns:
            Self for method chaining
        """
        self._temp_temperature = temperature
        return self
    
    def with_max_tokens(self, max_tokens: int) -> 'BaseLLM':
        """
        Set max tokens for the current call
        
        Args:
            max_tokens: Maximum tokens to generate
        
        Returns:
            Self for method chaining
        """
        self._temp_max_tokens = max_tokens
        return self
    
    def call(self, **kwargs) -> ChatResponse:
        """
        Execute the call with accumulated prompts
        
        Args:
            **kwargs: Additional generation parameters
        
        Returns:
            ChatResponse
        """
        messages = []
        
        # Add system prompt if set
        if hasattr(self, '_temp_system_prompt') and self._temp_system_prompt:
            messages.append(Message.system(self._temp_system_prompt))
        
        # Add assistant prompt if set (for context)
        if hasattr(self, '_temp_assistant_prompt') and self._temp_assistant_prompt:
            messages.append(Message.assistant(self._temp_assistant_prompt))
        
        # Add user prompt if set
        if hasattr(self, '_temp_user_prompt') and self._temp_user_prompt:
            messages.append(Message.user(self._temp_user_prompt))
        else:
            raise ValueError("User prompt is required. Use .with_user() or .call(user=...)")
        
        # Prepare parameters
        params = {}
        if hasattr(self, '_temp_temperature'):
            params['temperature'] = self._temp_temperature
        if hasattr(self, '_temp_max_tokens'):
            params['max_tokens'] = self._temp_max_tokens
        
        params.update(kwargs)
        
        # Clear temporary attributes
        self._clear_temp_attrs()
        
        # Generate response
        return self.generate(messages, **params)
    
    async def acall(self, **kwargs) -> ChatResponse:
        """
        Async execute the call with accumulated prompts
        
        Args:
            **kwargs: Additional generation parameters
        
        Returns:
            ChatResponse
        """
        messages = []
        
        # Add system prompt if set
        if hasattr(self, '_temp_system_prompt') and self._temp_system_prompt:
            messages.append(Message.system(self._temp_system_prompt))
        
        # Add assistant prompt if set (for context)
        if hasattr(self, '_temp_assistant_prompt') and self._temp_assistant_prompt:
            messages.append(Message.assistant(self._temp_assistant_prompt))
        
        # Add user prompt if set
        if hasattr(self, '_temp_user_prompt') and self._temp_user_prompt:
            messages.append(Message.user(self._temp_user_prompt))
        else:
            raise ValueError("User prompt is required. Use .with_user() or .acall(user=...)")
        
        # Prepare parameters
        params = {}
        if hasattr(self, '_temp_temperature'):
            params['temperature'] = self._temp_temperature
        if hasattr(self, '_temp_max_tokens'):
            params['max_tokens'] = self._temp_max_tokens
        
        params.update(kwargs)
        
        # Clear temporary attributes
        self._clear_temp_attrs()
        
        # Generate response
        return await self.agenerate(messages, **params)
    
    def call_stream(self, **kwargs):
        """
        Stream the call with accumulated prompts
        
        Args:
            **kwargs: Additional generation parameters
        
        Yields:
            Chunks of the response
        """
        messages = []
        
        # Add system prompt if set
        if hasattr(self, '_temp_system_prompt') and self._temp_system_prompt:
            messages.append(Message.system(self._temp_system_prompt))
        
        # Add assistant prompt if set (for context)
        if hasattr(self, '_temp_assistant_prompt') and self._temp_assistant_prompt:
            messages.append(Message.assistant(self._temp_assistant_prompt))
        
        # Add user prompt if set
        if hasattr(self, '_temp_user_prompt') and self._temp_user_prompt:
            messages.append(Message.user(self._temp_user_prompt))
        else:
            raise ValueError("User prompt is required. Use .with_user() or .call_stream(user=...)")
        
        # Prepare parameters
        params = {}
        if hasattr(self, '_temp_temperature'):
            params['temperature'] = self._temp_temperature
        if hasattr(self, '_temp_max_tokens'):
            params['max_tokens'] = self._temp_max_tokens
        
        params.update(kwargs)
        
        # Clear temporary attributes
        self._clear_temp_attrs()
        
        # Stream response
        yield from self.stream(messages, **params)
    
    async def acall_stream(self, **kwargs):
        """
        Async stream the call with accumulated prompts
        
        Args:
            **kwargs: Additional generation parameters
        
        Yields:
            Chunks of the response
        """
        messages = []
        
        # Add system prompt if set
        if hasattr(self, '_temp_system_prompt') and self._temp_system_prompt:
            messages.append(Message.system(self._temp_system_prompt))
        
        # Add assistant prompt if set (for context)
        if hasattr(self, '_temp_assistant_prompt') and self._temp_assistant_prompt:
            messages.append(Message.assistant(self._temp_assistant_prompt))
        
        # Add user prompt if set
        if hasattr(self, '_temp_user_prompt') and self._temp_user_prompt:
            messages.append(Message.user(self._temp_user_prompt))
        else:
            raise ValueError("User prompt is required. Use .with_user() or .acall_stream(user=...)")
        
        # Prepare parameters
        params = {}
        if hasattr(self, '_temp_temperature'):
            params['temperature'] = self._temp_temperature
        if hasattr(self, '_temp_max_tokens'):
            params['max_tokens'] = self._temp_max_tokens
        
        params.update(kwargs)
        
        # Clear temporary attributes
        self._clear_temp_attrs()
        
        # Stream response
        async for chunk in self.astream(messages, **params):
            yield chunk
    
    def _clear_temp_attrs(self):
        """Clear temporary attributes after call"""
        temp_attrs = ['_temp_system_prompt', '_temp_user_prompt', '_temp_assistant_prompt', 
                      '_temp_temperature', '_temp_max_tokens']
        for attr in temp_attrs:
            if hasattr(self, attr):
                delattr(self, attr)
    
    # ============================================
    # CONVENIENCE METHODS FOR PROMPTS
    # ============================================
    
    def ask(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        assistant_context: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Simple ask method with optional system and assistant context
        
        Args:
            user_prompt: User's question/prompt
            system_prompt: Optional system instruction
            assistant_context: Optional assistant message for context
            **kwargs: Additional generation parameters
        
        Returns:
            ChatResponse
        """
        messages = []
        
        if system_prompt:
            messages.append(Message.system(system_prompt))
        
        if assistant_context:
            messages.append(Message.assistant(assistant_context))
        
        messages.append(Message.user(user_prompt))
        
        return self.generate(messages, **kwargs)
    
    async def aask(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        assistant_context: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Async ask method with optional system and assistant context
        
        Args:
            user_prompt: User's question/prompt
            system_prompt: Optional system instruction
            assistant_context: Optional assistant message for context
            **kwargs: Additional generation parameters
        
        Returns:
            ChatResponse
        """
        messages = []
        
        if system_prompt:
            messages.append(Message.system(system_prompt))
        
        if assistant_context:
            messages.append(Message.assistant(assistant_context))
        
        messages.append(Message.user(user_prompt))
        
        return await self.agenerate(messages, **kwargs)
    
    def ask_stream(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        assistant_context: Optional[str] = None,
        **kwargs
    ):
        """
        Streaming ask method with optional system and assistant context
        
        Args:
            user_prompt: User's question/prompt
            system_prompt: Optional system instruction
            assistant_context: Optional assistant message for context
            **kwargs: Additional generation parameters
        
        Yields:
            Chunks of the response
        """
        messages = []
        
        if system_prompt:
            messages.append(Message.system(system_prompt))
        
        if assistant_context:
            messages.append(Message.assistant(assistant_context))
        
        messages.append(Message.user(user_prompt))
        
        yield from self.stream(messages, **kwargs)
    
    # ============================================
    # CHAT METHODS WITH CONTEXT MEMORY
    # ============================================
    
    def chat(
        self,
        message: str,
        session_id: str = "default",
        system_message: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Chat with context memory
        
        Args:
            message: User message
            session_id: Session ID for maintaining separate conversations
            system_message: Optional system message to set for this session
            **kwargs: Additional generation parameters
        
        Returns:
            ChatResponse with assistant's reply
        """
        # Get or create context for this session
        if session_id not in self._contexts:
            self._contexts[session_id] = ConversationContext(max_history=self.max_history)
        
        context = self._contexts[session_id]
        
        # Set system message if provided
        if system_message:
            context.set_system_message(system_message)
        
        # Add user message to context
        context.add_user_message(message)
        
        # Get full conversation
        conversation = context.get_conversation()
        
        # Generate response
        response = self.generate(conversation, **kwargs)
        
        # Add assistant response to context
        context.add_assistant_message(response.content)
        
        return response
    
    async def achat(
        self,
        message: str,
        session_id: str = "default",
        system_message: Optional[str] = None,
        **kwargs
    ) -> ChatResponse:
        """Async chat with context memory"""
        if session_id not in self._contexts:
            self._contexts[session_id] = ConversationContext(max_history=self.max_history)
        
        context = self._contexts[session_id]
        
        if system_message:
            context.set_system_message(system_message)
        
        context.add_user_message(message)
        conversation = context.get_conversation()
        response = await self.agenerate(conversation, **kwargs)
        context.add_assistant_message(response.content)
        
        return response
    
    def chat_stream(
        self,
        message: str,
        session_id: str = "default",
        system_message: Optional[str] = None,
        **kwargs
    ):
        """Stream chat with context memory"""
        if session_id not in self._contexts:
            self._contexts[session_id] = ConversationContext(max_history=self.max_history)
        
        context = self._contexts[session_id]
        
        if system_message:
            context.set_system_message(system_message)
        
        context.add_user_message(message)
        conversation = context.get_conversation()
        
        full_response = []
        for chunk in self.stream(conversation, **kwargs):
            full_response.append(chunk)
            yield chunk
        
        context.add_assistant_message("".join(full_response))
    
    async def achat_stream(
        self,
        message: str,
        session_id: str = "default",
        system_message: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Async stream chat with context memory"""
        if session_id not in self._contexts:
            self._contexts[session_id] = ConversationContext(max_history=self.max_history)
        
        context = self._contexts[session_id]
        
        if system_message:
            context.set_system_message(system_message)
        
        context.add_user_message(message)
        conversation = context.get_conversation()
        
        full_response = []
        async for chunk in self.astream(conversation, **kwargs):
            full_response.append(chunk)
            yield chunk
        
        context.add_assistant_message("".join(full_response))
    
    # ============================================
    # HELPER METHODS
    # ============================================
    
    def get_conversation_history(self, session_id: str = "default") -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        if session_id not in self._contexts:
            return []
        
        context = self._contexts[session_id]
        return [msg.to_dict() for msg in context.get_conversation()]
    
    def clear_conversation(self, session_id: str = "default"):
        """Clear conversation history for a session"""
        if session_id in self._contexts:
            self._contexts[session_id].clear()
    
    def reset_conversation(self, session_id: str = "default"):
        """Reset conversation completely for a session"""
        if session_id in self._contexts:
            self._contexts[session_id].reset()
        elif session_id in self._contexts:
            del self._contexts[session_id]
    
    def set_system_message(self, system_message: str, session_id: str = "default"):
        """Set system message for a session"""
        if session_id not in self._contexts:
            self._contexts[session_id] = ConversationContext(max_history=self.max_history)
        
        self._contexts[session_id].set_system_message(system_message)
    
    def _format_messages(
        self,
        prompt: Union[str, List[Message]]
    ) -> List[Dict[str, str]]:
        """Convert prompt to list of message dicts"""
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            return [msg.to_dict() for msg in prompt]
        elif isinstance(prompt, Message):
            return [prompt.to_dict()]
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")
    
    def close(self):
        """Close client connections"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()