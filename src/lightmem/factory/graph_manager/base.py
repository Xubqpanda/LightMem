import os
import uuid
import pickle
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
from importlib import import_module

from lightmem.memory.utils import MemoryEntry
from lightmem.configs.graph_manager.base import GraphManagerConfig

logger = logging.getLogger(__name__)


class BaseGraphManager:
    def __init__(
        self,
        config: GraphManagerConfig,
        embedder: Optional[Any] = None,
    ):
        self.config = config
        self.storage_path = config.storage_path
        self.merge_threshold = config.merge_threshold
        self.temporal_window_days = config.temporal_window_days

        try:
            self._nx = import_module("networkx")
            self._np = import_module("numpy")
            sklearn_cluster = import_module("sklearn.cluster")
            self._DBSCAN = getattr(sklearn_cluster, "DBSCAN")
        except ImportError as exc:
            raise ImportError(
                "GraphManager requires networkx, numpy, and scikit-learn to be installed"
            ) from exc

        self.graph = self._nx.DiGraph()

        self.embedder = embedder or self._init_embedder(config)
        if self.embedder is None:
            logger.warning(
                "GraphManager embedder not configured; falling back to simple length-based embedding"
            )
            self.embedder = self._fallback_embedder

        self.summarizer = self._init_summarizer(config)

    def _init_embedder(self, cfg: GraphManagerConfig):
        return None

    def _init_summarizer(self, cfg: GraphManagerConfig):
        return None

    @staticmethod
    def _fallback_embedder(text: str) -> List[float]:
        return [float(len(text or ""))]

    def _embed_texts(self, texts: List[str]) -> Any:
        if hasattr(self.embedder, "embed"):
            vectors = self.embedder.embed(texts)
        else:
            vectors = [self.embedder(t) for t in texts]
        return self._np.array(vectors, dtype=float)

    def save(self, path: Optional[str] = None):
        path = path or self.storage_path
        with open(path, "wb") as f:
            pickle.dump(self._nx.node_link_data(self.graph), f)
        logger.info("Graph saved to %s", path)

    def load(self, path: Optional[str] = None):
        path = path or self.storage_path
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.graph = self._nx.node_link_graph(data)
        logger.info("Graph loaded from %s", path)

    def build_graph(
        self,
        entries: List[MemoryEntry],
        *,
        sample_size_per_cluster: int = 5,
        dbscan_eps: float = 0.15,
        dbscan_min_samples: int = 2,
        summarizer_override: Optional[Callable[[List[MemoryEntry]], Dict[str, Any]]] = None,
    ):
        if not entries:
            logger.info("GraphManager.build_graph called with no entries")
            return

        for entry in entries:
            self._ensure_entry_node(entry)

        text_list = [entry.memory or "" for entry in entries]
        embeddings = self._embed_texts(text_list)

        clustering = self._DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="cosine")
        labels = clustering.fit_predict(embeddings)
        logger.info(
            "DBSCAN clusters=%s noise=%s",
            len(set(labels)) - (1 if -1 in labels else 0),
            int(self._np.sum(labels == -1)),
        )

        clusters: Dict[int, List[Any]] = {}
        for label, entry, emb in zip(labels, entries, embeddings):
            clusters.setdefault(label, []).append((entry, emb))

        for label, members in clusters.items():
            if label == -1:
                for entry, emb in members:
                    self._create_event([(entry, emb)], sample_size_per_cluster, summarizer_override)
                continue

            for subgroup in self._split_large_cluster(members):
                self._create_event(subgroup, sample_size_per_cluster, summarizer_override)

    def _ensure_entry_node(self, entry: MemoryEntry):
        if not self.graph.has_node(entry.id):
            self.graph.add_node(
                entry.id,
                type="entry",
                memory=entry.memory,
                mentioned_time=entry.time_stamp,
                metadata={
                    "topic_id": entry.topic_id,
                    "speaker_id": entry.speaker_id,
                    "speaker_name": entry.speaker_name,
                },
            )

    def _create_event(
        self,
        members: List[Any],
        sample_size: int,
        summarizer_override: Optional[Callable[[List[MemoryEntry]], Dict[str, Any]]],
    ):
        entries = [entry for entry, _ in members]
        embeddings = self._np.vstack([vec for _, vec in members])
        centroid = list(self._np.mean(embeddings, axis=0))
        representative_entries = entries[:sample_size]

        if len(entries) == 1:
            summary_info = {"description": "", "entry_event_times": {}}
        else:
            summary_info = self._summarize(representative_entries, summarizer_override)
        description = summary_info.get("description", "")
        inferred_times = summary_info.get("entry_event_times", {})

        times: List[datetime] = []
        for entry in entries:
            if inferred_times and inferred_times.get(entry.id):
                try:
                    times.append(datetime.fromisoformat(inferred_times[entry.id]))
                    continue
                except Exception:
                    pass
            try:
                times.append(datetime.fromisoformat(entry.time_stamp))
            except Exception:
                continue

        start = min(times).isoformat() if times else None
        end = max(times).isoformat() if times else None

        event_id = str(uuid.uuid4())
        self.graph.add_node(
            event_id,
            type="event",
            description=description,
            refs=[entry.id for entry in entries],
            centroid=centroid,
            start_date=start,
            end_date=end,
            support_count=len(entries),
        )

        for entry in entries:
            self.graph.add_edge(
                event_id,
                entry.id,
                relation="mentions",
                mentioned_time=entry.time_stamp,
                event_time=inferred_times.get(entry.id) if inferred_times else None,
            )

        logger.debug("Created event node %s with %s refs", event_id, len(entries))

    def _summarize(
        self,
        entries: List[MemoryEntry],
        summarizer_override: Optional[Callable[[List[MemoryEntry]], Dict[str, Any]]],
    ) -> Dict[str, Any]:
        if summarizer_override:
            try:
                return summarizer_override(entries) or {"description": "", "entry_event_times": {}}
            except Exception:
                logger.exception("Override summarizer failed; falling back to default")
        if self.summarizer:
            try:
                return self.summarizer(entries)
            except Exception:
                logger.exception(
                    "GraphManager summarizer failed; falling back to concatenation"
                )
        description = " ; ".join([entry.memory for entry in entries[:5]])
        return {"description": description, "entry_event_times": {}}

    def _split_large_cluster(self, members: List[Any]) -> List[List[Any]]:
        max_size = self.config.max_cluster_size or 0
        if max_size <= 0 or len(members) <= max_size:
            return [members]

        def _sort_key(item: Any):
            entry: MemoryEntry = item[0]
            stamp = entry.time_stamp or ""
            try:
                return (datetime.fromisoformat(stamp), entry.id)
            except Exception:
                return (datetime.min, entry.id)

        ordered = sorted(members, key=_sort_key)
        return [ordered[i : i + max_size] for i in range(0, len(ordered), max_size)]
