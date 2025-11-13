import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from lightmem.configs.graph_manager.base import GraphManagerConfig
from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig
from lightmem.memory.utils import MemoryEntry
from .base import BaseGraphManager

logger = logging.getLogger(__name__)


class _OpenAISummarizer:
    """Lightweight OpenAI-based summarizer used internally by OpenAIGraphManager."""

    def __init__(self, config: GraphManagerConfig):
        try:
            from openai import OpenAI  # lazy import
        except ImportError as exc:
            raise ImportError("openai package is required for OpenAI summarizer") from exc

        api_key = config.summarizer_api_key or os.getenv("OPENAI_API_KEY")
        base_url = (
            config.summarizer_base_url
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
        )
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = config.summarizer_model or "gpt-4o-mini"
        self.temperature = config.summarizer_temperature

    def __call__(self, entries: List[MemoryEntry]) -> Dict[str, Any]:
        if not entries:
            return {"description": ""}

        selected_entries = _select_representative_entries(entries)

        snippets = []
        for item in selected_entries:
            snippets.append(
                f"ID: {item.id}\nMentioned: {item.time_stamp}\nContent: {item.memory}"
            )
        user_prompt = "Entries (ordered chronologically, may include short contextual snippets):\n" + "\n---\n".join(snippets)
        user_prompt += "\n\nFollow the system instructions exactly and respond in JSON only."
        system_prompt = (
            "You are an expert analyst who turns clustered diary entries into a single high-level event narrative. "
            "Blend concrete facts with inferred persona insights (motives, attitudes, emotional shifts) in a concise way. "
            "Output JSON with a single key 'description' containing one to three sentences that cover the shared storyline and any relevant profile insights. "
            "Write in paragraph form with flowing sentences, avoiding bullet points, numbered lists, or separators such as '|'. "
            "Avoid quoting entries verbatim and do not invent details beyond the provided notes. "
            "Example response: {\n  \"description\": \"Caroline channels her grief into volunteering, leaning on friends for stability while recommitting to community causes.\"\n} "
            "Always return valid JSON with only the 'description' key."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.warning("Summarizer JSON enforcement failed (%s); retrying without response_format", exc)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
        text = response.choices[0].message.content

        payload: Dict[str, Any]
        try:
            payload = _safe_json_parse(text)
        except Exception:
            logger.warning("Failed to parse summarizer JSON response; falling back to raw text")
            payload = {"description": text.strip()}
        description = payload.get("description", "") if isinstance(payload, dict) else str(payload)
        cleaned = " ".join(part.strip() for part in description.replace("\n", " ").split("|") if part.strip())
        normalized = " ".join(cleaned.split())
        return {"description": normalized}

def _safe_json_parse(text: str) -> Dict[str, Any]:
    import json
    import re

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        raise


def _select_representative_entries(entries: List[MemoryEntry], limit: int = 20) -> List[MemoryEntry]:
    if not entries:
        return []

    seen = set()
    deduped: List[MemoryEntry] = []
    for item in entries:
        text = (item.memory or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    def sort_key(entry: MemoryEntry):
        stamp = entry.time_stamp or ""
        try:
            return datetime.fromisoformat(stamp)
        except Exception:
            return datetime.min

    deduped.sort(key=sort_key)
    if len(deduped) <= limit or limit <= 0:
        return deduped[:limit] if limit > 0 else deduped

    if limit == 1:
        return [deduped[len(deduped) // 2]]

    selected: List[MemoryEntry] = []
    span = len(deduped) - 1
    for idx in range(limit):
        pos = round(idx * span / (limit - 1))
        selected.append(deduped[pos])

    # Ensure uniqueness in case rounding collides
    unique_entries: List[MemoryEntry] = []
    seen_ids = set()
    for entry in selected:
        if entry.id in seen_ids:
            continue
        seen_ids.add(entry.id)
        unique_entries.append(entry)

    if len(unique_entries) < limit:
        for entry in deduped:
            if entry.id in seen_ids:
                continue
            seen_ids.add(entry.id)
            unique_entries.append(entry)
            if len(unique_entries) == limit:
                break

    return unique_entries


class OpenAIGraphManager(BaseGraphManager):
    def __init__(self, config: GraphManagerConfig, embedder: Optional[Any] = None):
        if config.embedder_provider and config.embedder_provider != "openai":
            raise ValueError("OpenAIGraphManager requires embedder_provider='openai'")
        if config.summarizer_provider and config.summarizer_provider != "openai":
            raise ValueError("OpenAIGraphManager requires summarizer_provider='openai'")
        super().__init__(config=config, embedder=embedder)

    def _init_embedder(self, cfg: GraphManagerConfig):
        if not cfg.embedder_provider:
            return None
        try:
            from lightmem.factory.text_embedder.openai import TextEmbedderOpenAI
        except ImportError as exc:
            raise ImportError("OpenAI embedder requested but openai package not installed") from exc

        config = BaseTextEmbedderConfig(
            model=cfg.embedder_model,
            api_key=cfg.embedder_api_key or os.getenv("OPENAI_API_KEY"),
            embedding_dims=cfg.embedder_embedding_dims,
            openai_base_url=
                cfg.embedder_base_url or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL"),
        )
        return TextEmbedderOpenAI(config)

    def _init_summarizer(self, cfg: GraphManagerConfig):
        if not cfg.summarizer_provider:
            return None
        return _OpenAISummarizer(cfg)
