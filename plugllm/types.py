"""Type definitions for PlugLLM"""

from typing import TypedDict, Optional, List, Dict, Any, Union, Literal
from enum import Enum


class Role(str, Enum):
    """Message role types"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class MessageDict(TypedDict):
    """Dictionary representation of a message"""
    role: str
    content: str
    name: Optional[str]


class UsageDict(TypedDict, total=False):
    """Token usage information"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionParams(TypedDict, total=False):
    """Parameters for chat completion"""
    model: str
    messages: List[MessageDict]
    temperature: float
    top_p: float
    n: int
    stream: bool
    stop: Union[str, List[str]]
    max_tokens: int
    presence_penalty: float
    frequency_penalty: float
    logit_bias: Dict[str, float]
    user: str


class ProviderConfig(TypedDict, total=False):
    """Provider-specific configuration"""
    api_key: str
    base_url: str
    timeout: int
    max_retries: int
    organization: str  # OpenAI specific
    project_id: str    # OpenAI specific
    region: str        # Some providers
    version: str       # API version


class ModelInfo(TypedDict):
    """Information about a model"""
    id: str
    name: str
    provider: str
    context_length: int
    max_tokens: int
    supports_streaming: bool
    supports_function_calling: bool
    supports_vision: bool
    pricing: Optional[Dict[str, float]]


class StreamingChunk(TypedDict):
    """Streaming chunk structure"""
    content: str
    finish_reason: Optional[str]
    index: int


class ToolCall(TypedDict):
    """Function/tool call structure"""
    id: str
    type: Literal["function"]
    function: Dict[str, Any]


class ResponseFormat(TypedDict, total=False):
    """Response format specification"""
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[Dict[str, Any]]


# Type aliases for better readability
PromptType = Union[str, List[Union[MessageDict, Any]]]
TemperatureType = Union[float, int]
StopSequenceType = Union[str, List[str]]
TimeoutType = Union[int, float]

# Provider name type
ProviderName = Literal[
    "openai",
    "gemini",
    "groq",
    "claude",
    "grok",
    "mistral",
    "llama",
    "deepseek",
    "qwen",
    "kimi",
    "cohere",
    "sarvamai",
    "ollama"
]


class LLMConfig(TypedDict, total=False):
    """Complete LLM configuration"""
    provider: ProviderName
    model: str
    api_key: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    timeout: int
    max_retries: int
    stream: bool


class BatchRequest(TypedDict):
    """Batch request structure"""
    id: str
    prompt: PromptType
    params: Optional[Dict[str, Any]]


class BatchResponse(TypedDict):
    """Batch response structure"""
    id: str
    response: Any
    error: Optional[str]
    status: Literal["success", "error"]


# Provider-specific response types
class OpenAIResponse(TypedDict):
    """OpenAI API response structure"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: UsageDict


class GeminiResponse(TypedDict):
    """Gemini API response structure"""
    candidates: List[Dict[str, Any]]
    promptFeedback: Optional[Dict[str, Any]]


class ClaudeResponse(TypedDict):
    """Claude API response structure"""
    id: str
    type: str
    role: str
    content: List[Dict[str, Any]]
    model: str
    stop_reason: Optional[str]
    usage: UsageDict


class CohereResponse(TypedDict):
    """Cohere API response structure"""
    id: str
    message: Dict[str, Any]
    usage: UsageDict


# Error types
class LLMError(Exception):
    """Base LLM exception"""
    pass


class AuthenticationError(LLMError):
    """API key authentication failed"""
    pass


class RateLimitError(LLMError):
    """Rate limit exceeded"""
    pass


class InvalidRequestError(LLMError):
    """Invalid request parameters"""
    pass


class ModelNotFoundError(LLMError):
    """Requested model not found"""
    pass


class TimeoutError(LLMError):
    """Request timeout"""
    pass


class StreamingError(LLMError):
    """Error during streaming"""
    pass


# Completion status
class CompletionStatus(str, Enum):
    """Status of completion"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Cache configuration
class CacheConfig(TypedDict, total=False):
    """Cache configuration"""
    enabled: bool
    ttl: int  # Time to live in seconds
    max_size: int  # Maximum cache size
    backend: Literal["memory", "redis", "disk"]


# Retry configuration
class RetryConfig(TypedDict, total=False):
    """Retry configuration"""
    max_retries: int
    backoff_factor: float
    retry_on_status: List[int]
    retry_on_exceptions: List[str]