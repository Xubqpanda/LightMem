from pydantic import BaseModel, Field
from typing import Optional, Literal


class GraphManagerConfig(BaseModel):
    """Configuration for the offline GraphManager."""

    enabled: Optional[bool] = Field(default=True, description="Enable the GraphManager")
    implementation: str = Field(
        default="openai",
        description="Graph manager implementation key used by the factory",
    )
    storage_path: Optional[str] = Field(default="graph.pkl", description="Path to persist the graph file")
    merge_threshold: Optional[float] = Field(default=0.85, description="Embedding cosine similarity threshold for merging")
    temporal_window_days: Optional[int] = Field(default=90, description="Temporal window (days) for stronger linking of events")
    summarizer_provider: Optional[Literal["openai"]] = Field(
        default="openai",
        description="LLM provider for event summarization",
    )
    summarizer_model: Optional[str] = Field(
        default="gpt-4o-mini",
        description="Model name for the summarizer",
    )
    summarizer_temperature: float = Field(
        default=0.0,
        description="Sampling temperature for the summarizer",
    )
    summarizer_max_tokens: int = Field(
        default=512,
        description="Maximum tokens for summarization outputs",
    )
    summarizer_api_key: Optional[str] = Field(
        default=None,
        description="API key for the summarizer (overrides environment)",
    )
    summarizer_base_url: Optional[str] = Field(
        default=None,
        description="Override base URL for the summarizer provider",
    )
    max_cluster_size: Optional[int] = Field(
        default=None,
        description="Maximum entries allowed per cluster before automatic splitting (0 disables)",
    )
    embedder_provider: Optional[Literal["openai"]] = Field(
        default="openai",
        description="Embedding provider for clustering",
    )
    embedder_model: Optional[str] = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    embedder_api_key: Optional[str] = Field(
        default=None,
        description="API key for the embedding provider (overrides environment)",
    )
    embedder_base_url: Optional[str] = Field(
        default=None,
        description="Override base URL for the embedding provider",
    )
    embedder_embedding_dims: Optional[int] = Field(
        default=None,
        description="Embedding dimension (if provider requires it)",
    )
