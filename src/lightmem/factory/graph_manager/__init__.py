from .base import BaseGraphManager
from .openai import OpenAIGraphManager

GraphManager = OpenAIGraphManager

__all__ = ["BaseGraphManager", "OpenAIGraphManager", "GraphManager"]
