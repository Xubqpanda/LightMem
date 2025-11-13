from typing import Dict, Any
from importlib import import_module
from lightmem.configs.graph_manager.base import GraphManagerConfig

class GraphManagerFactory:
    _MODEL_MAPPING: Dict[str, str] = {
        "default": "lightmem.factory.graph_manager.openai.OpenAIGraphManager",
        "openai": "lightmem.factory.graph_manager.openai.OpenAIGraphManager",
    }

    @classmethod
    def from_config(cls, config: GraphManagerConfig, *, default_embedder: Any = None):
        """Instantiate GraphManager from config."""
        key = getattr(config, "implementation", "default")
        class_path = cls._MODEL_MAPPING.get(key, cls._MODEL_MAPPING["default"])
        module_path, class_name = class_path.rsplit('.', 1)
        module = import_module(module_path)
        manager_class = getattr(module, class_name)
        return manager_class(config=config, embedder=default_embedder)
