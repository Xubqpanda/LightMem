import sys
import types
from pathlib import Path


def test_factory_resolves_openai():
    # Ensure a dummy `openai` module exists so importing the embedder module won't fail
    dummy = types.ModuleType("openai")

    class DummyOpenAIClient:
        def __init__(self, *a, **k):
            pass

    dummy.OpenAI = DummyOpenAIClient
    sys.modules["openai"] = dummy

    # Import the factory and the config model
    from lightmem.configs.text_embedder.base import TextEmbedderConfig
    from lightmem.factory.text_embedder.factory import TextEmbedderFactory

    # Build a minimal config that selects `openai` and provides empty configs
    cfg = TextEmbedderConfig(model_name="openai", configs={})

    embedder = TextEmbedderFactory.from_config(cfg)

    # Basic sanity checks: we got an object and it has an `embed` or `from_config` attribute
    assert embedder is not None
    assert hasattr(embedder, "embed") or hasattr(embedder, "from_config")


if __name__ == "__main__":
    test_factory_resolves_openai()
    print("ok")
