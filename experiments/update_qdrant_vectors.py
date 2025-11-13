"""Update stored Qdrant vectors in on-disk collections using OpenAI embeddings.

This script walks a qdrant data directory (the on-disk layout created by
Qdrant) and replaces or fills the `vector` field inside the pickled `point`
blobs in `storage.sqlite` for each collection.

Usage notes:
- Dry-run by default (no writes). Use --apply to persist changes.
- Creates a `.bak` copy of each sqlite file before modifying.
- Batches embedding requests for efficiency.
- Text to embed is extracted with heuristics: payload['memory'] or
  payload['original_memory'] or payload['compressed_memory'] or payload.get('memory')
  or top-level 'memory' fields.

Be careful: this mutates on-disk data. Keep backups and test on a subset first.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import pickle
import shutil
import logging
from typing import List, Tuple, Optional, Dict, Any
API_KEY='sk-mYmdqXKCUL9FqNfI27855c29E94d419c995bA6D54c20Af21'
API_BASE_URL='https://api.gpts.vin/v1'

logger = logging.getLogger("update_qdrant_vectors")


def find_collections(root: str) -> List[str]:
    # Collections are stored under <root>/<collection_name>/collection/<collection_name>/storage.sqlite
    cols = []
    if not os.path.isdir(root):
        return cols
    for name in os.listdir(root):
        full = os.path.join(root, name)
        # skip files like meta.json
        if not os.path.isdir(full):
            continue
        # possible nested 'collection' dir
        candidate = os.path.join(full, 'collection', name, 'storage.sqlite')
        if os.path.exists(candidate):
            cols.append(candidate)
        else:
            # sometimes root itself may contain collection folders directly
            candidate2 = os.path.join(root, name, 'collection', name, 'storage.sqlite')
            if os.path.exists(candidate2):
                cols.append(candidate2)
    return cols


def extract_text_from_obj(obj: Any) -> Optional[str]:
    # Heuristic extraction of text to embed
    item = {}
    if isinstance(obj, dict):
        item = obj
    else:
        try:
            item = getattr(obj, '__dict__', {}) or {}
        except Exception:
            item = {}

    # point wrapper
    point = item.get('point') if isinstance(item.get('point'), dict) else None
    payload = None
    if point:
        payload = point.get('payload', {})
    else:
        payload = item.get('payload') if isinstance(item.get('payload'), dict) else None

    candidates = []
    if payload:
        for k in ('memory', 'original_memory', 'compressed_memory'):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                candidates.append(v.strip())
    # top-level memory
    for k in ('memory', 'original_memory', 'compressed_memory'):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(v.strip())

    # fallback: if point exists and has payload with other text fields
    if not candidates and point and isinstance(point.get('payload'), dict):
        for v in point.get('payload').values():
            if isinstance(v, str) and len(v) > 20:
                candidates.append(v)

    return candidates[0] if candidates else None


def update_sqlite_vectors(sqlite_path: str, embedder, batch_size: int = 16, apply: bool = False) -> Tuple[int, int]:
    """Return (processed, updated) counts."""
    logger.info(f"Processing sqlite: {sqlite_path}")
    conn = sqlite3.connect(sqlite_path)
    cur = conn.execute("SELECT id, point FROM points")

    rows: List[Tuple[int, bytes, Any]] = []
    for row in cur:
        pid, blob = row
        rows.append((pid, blob, None))
    logger.info(f"Found {len(rows)} rows")

    processed = 0
    updated = 0

    # Prepare batches of texts to embed
    i = 0
    while i < len(rows):
        batch = rows[i:i+batch_size]
        texts = []
        parsed_objs = []
        ids = []
        for pid, blob, _ in batch:
            try:
                obj = pickle.loads(blob)
            except Exception:
                try:
                    obj = pickle.loads(blob, fix_imports=True)
                except Exception as e:
                    logger.warning(f"Failed to unpickle id={pid}: {e}")
                    obj = None
            parsed_objs.append(obj)
            txt = extract_text_from_obj(obj) if obj is not None else None
            texts.append(txt if txt is not None else "")
            ids.append(pid)

        # Compute embeddings only for non-empty texts
        emb_inputs = [t for t in texts]
        try:
            embeddings = embedder.embed(emb_inputs)
        except Exception as e:
            logger.error(f"Embedding batch failed at offset {i}: {e}")
            break

        # Ensure embeddings align (single vs list)
        if len(embeddings) == 0:
            logger.warning("Received empty embeddings for batch; skipping")
            i += batch_size
            continue
        if not isinstance(embeddings[0], (list, tuple)):
            # single vector returned for single input
            embeddings = [embeddings]

        # Update objects and write back if apply
        for idx, pid in enumerate(ids):
            obj = parsed_objs[idx]
            emb = embeddings[idx]
            if obj is None:
                continue
            # set vector on object in a few possible places
            wrote = False
            if isinstance(obj, dict):
                if 'point' in obj and isinstance(obj['point'], dict):
                    obj['point']['vector'] = emb
                    wrote = True
                else:
                    obj['vector'] = emb
                    wrote = True
            else:
                # try setting attributes
                try:
                    if hasattr(obj, 'point'):
                        pt = getattr(obj, 'point')
                        if isinstance(pt, dict):
                            pt['vector'] = emb
                        else:
                            setattr(obj, 'vector', emb)
                        wrote = True
                    else:
                        setattr(obj, 'vector', emb)
                        wrote = True
                except Exception as e:
                    logger.warning(f"Failed to set vector on object id={pid}: {e}")

            if wrote:
                processed += 1
                if apply:
                    new_blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
                    try:
                        conn.execute("UPDATE points SET point = ? WHERE id = ?", (new_blob, pid))
                        updated += 1
                    except Exception as e:
                        logger.error(f"Failed to update id={pid}: {e}")

        if apply:
            conn.commit()

        i += batch_size

    conn.close()
    logger.info(f"Done {sqlite_path}: processed={processed} updated={updated}")
    return processed, updated


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qdrant-root', default='qdrant_data_locomo', help='Path to qdrant data dir')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--apply', action='store_true', help='If set, write changes to sqlite (default: dry-run)')
    p.add_argument('--collections', nargs='*', help='Optional list of collection names to restrict')
    p.add_argument('--embed-model', default=None, help='Optional embedder model override (passed to TextEmbedderOpenAI via config.model)')
    args = p.parse_args()

    root = os.path.abspath(args.qdrant_root)
    if not os.path.exists(root):
        logger.error(f"qdrant root not found: {root}")
        return

    from lightmem.factory.text_embedder.openai import TextEmbedderOpenAI
    from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig
    config = BaseTextEmbedderConfig()
    if args.embed_model:
        setattr(config, 'model', args.embed_model)
    embedder = TextEmbedderOpenAI(config)

    sqlite_files = find_collections(root)
    if args.collections:
        sqlite_files = [s for s in sqlite_files if any(os.path.basename(os.path.dirname(os.path.dirname(s))) == c or os.path.basename(os.path.dirname(s)) == c for c in args.collections)]

    if not sqlite_files:
        logger.error(f"No sqlite collections found under {root}")
        return

    for sqlite_path in sqlite_files:
        # backup
        bak = sqlite_path + '.bak'
        if args.apply:
            if not os.path.exists(bak):
                shutil.copy2(sqlite_path, bak)
                logger.info(f"Backup created: {bak}")
        else:
            logger.info(f"Dry-run: would backup {sqlite_path} to {bak} when --apply specified")

        update_sqlite_vectors(sqlite_path, embedder, batch_size=args.batch_size, apply=args.apply)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
