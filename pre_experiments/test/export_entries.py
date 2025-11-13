import os
import json
import sqlite3
import argparse
from typing import Any, Dict, List, Set

from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant
from lightmem.configs.retriever.embeddingretriever.qdrant import QdrantConfig


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_spacy(model_name: str = "en_core_web_sm"):
    try:
        import spacy
    except Exception as e:
        raise RuntimeError("spaCy is required for term extraction. Install with: pip install spacy") from e
    try:
        return spacy.load(model_name)
    except Exception as e:
        raise RuntimeError(f"spaCy model '{model_name}' not installed. Run: python -m spacy download {model_name}") from e


# common stopwords to filter from extracted terms (case-insensitive)
STOPWORDS = set(x.lower() for x in [
    'user','have','they','it','can','that','make','try','use','be','do','i','you',
    # common question words and function words to exclude
    'what', 'when', 'how', 'which', 'who', 'whom', 'whose', 'why',
    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'there', 'here'
])


def extract_terms_from_text(nlp, text: str, include_verbs: bool = True) -> List[str]:
    """Extract lemmas from NER entities, noun tokens/chunks, and verb tokens.

    Returns a deduplicated list of lemmas (strings).
    """
    if not text:
        return []
    doc = nlp(text)
    terms: List[str] = []

    # NER
    for ent in doc.ents:
        lemma = " ".join([tok.lemma_ for tok in ent])
        terms.append(lemma.strip())

    # Noun tokens
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            terms.append(token.lemma_.strip())

    # noun chunks: include lemma-based chunk (captures modifiers but normalized)
    for chunk in doc.noun_chunks:
        # build lemma form of the chunk to normalize plurals/inflections
        lemmas = [tok.lemma_.strip() for tok in chunk if tok.lemma_.strip()]
        if lemmas:
            chunk_lemma = " ".join(lemmas)
            terms.append(chunk_lemma)

    # adjectival modifiers with noun lemma (e.g., 'vintage film')
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            mods = [m.lemma_.strip() for m in token.lefts if m.dep_ == "amod" and m.pos_ == "ADJ"]
            if mods:
                phrase = " ".join(mods + [token.lemma_.strip()])
                terms.append(phrase)

    # verbs (optional)
    if include_verbs:
        for token in doc:
            if token.pos_ in ("VERB", "AUX"):
                if token.lemma_.lower() in ("be", "do"):
                    continue
                terms.append(token.lemma_.strip())

    # Normalize and deduplicate while preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for t in terms:
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def export_collection(collection_name: str, qdrant_path: str, out_dir: str, include_vectors: bool = False,
                      extract_terms: bool = False, spacy_model: str = "en_core_web_sm",
                      include_verbs: bool = True):
    ensure_dir(out_dir)

    cfg = QdrantConfig(collection_name=collection_name, path=qdrant_path, embedding_model_dims=384, on_disk=True)
    q = Qdrant(cfg)

    print(f"Reading all points from collection '{collection_name}' (this may take a while)...")
    points = q.get_all(with_vectors=include_vectors, with_payload=True)
    print(f"Retrieved {len(points)} points")

    # If the Qdrant wrapper returned zero points but storage.sqlite exists, try to read it directly.
    if len(points) == 0:
        storage_sqlite = os.path.join(qdrant_path, collection_name, 'collection', collection_name, 'storage.sqlite')
        if os.path.exists(storage_sqlite):
            print(f"No points returned by client; attempting direct sqlite fallback: {storage_sqlite}")
            try:
                import pickle

                conn_local = sqlite3.connect(storage_sqlite)
                cur_local = conn_local.execute("SELECT id, point FROM points")
                raw_points = []
                for row in cur_local:
                    pid, blob = row
                    try:
                        obj = pickle.loads(blob)
                    except Exception:
                        # qdrant may have used protocol with latin1 - try decoding then unpickle
                        try:
                            obj = pickle.loads(blob, fix_imports=True)
                        except Exception:
                            # skip unparseable rows
                            continue

                    # obj may be a PointStruct or dict-like
                    if isinstance(obj, dict):
                        raw_points.append(obj)
                    else:
                        # try to map attributes
                        rec = {}
                        if hasattr(obj, '__dict__'):
                            rec.update(getattr(obj, '__dict__', {}))
                        # also try common attrs
                        for a in ('id', 'payload', 'vector', 'score'):
                            if hasattr(obj, a) and a not in rec:
                                rec[a] = getattr(obj, a)
                        raw_points.append(rec)

                conn_local.close()
                points = []
                # Normalize to expected dict shape: {'id':..., 'payload':..., 'vector':...}
                for rp in raw_points:
                    item = {}
                    # id
                    if 'id' in rp:
                        item['id'] = rp.get('id')
                    elif 'point' in rp and isinstance(rp.get('point'), dict) and 'id' in rp.get('point'):
                        item['id'] = rp['point']['id']
                    # payload
                    if 'payload' in rp:
                        item['payload'] = rp.get('payload') or {}
                    elif 'point' in rp and isinstance(rp.get('point'), dict) and 'payload' in rp['point']:
                        item['payload'] = rp['point']['payload'] or {}
                    else:
                        # some serialized objects store payload fields at top-level
                        payload_guess = {k: v for k, v in rp.items() if k not in ('id', 'vector', 'point')}
                        item['payload'] = payload_guess
                    # vector
                    if 'vector' in rp and rp.get('vector') is not None:
                        item['vector'] = rp.get('vector')
                    elif 'point' in rp and isinstance(rp.get('point'), dict) and 'vector' in rp['point']:
                        item['vector'] = rp['point']['vector']

                    points.append(item)

                print(f"Fallback read found {len(points)} points from sqlite")
            except Exception as e:
                print("Fallback sqlite read failed:", e)

    # Optionally load spaCy model
    nlp = None
    if extract_terms:
        print(f"Loading spaCy model '{spacy_model}' for term extraction...")
        nlp = load_spacy(spacy_model)

    # Write JSON with inserted 'terms' field between id and payload
    json_path = os.path.join(out_dir, f"{collection_name}_entries.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        serializable = []
        for p in points:
            item = dict(p)
            # prepare combined terms if requested
            if extract_terms:
                payload = item.get('payload', {}) or {}
                # prefer memory-like fields
                memory_text = payload.get('memory') or payload.get('original_memory') or payload.get('compressed_memory') or payload.get('text') or payload.get('content')
                # fallback to top-level memory
                if not memory_text:
                    memory_text = item.get('memory')
                raw_terms = extract_terms_from_text(nlp, memory_text or "", include_verbs=include_verbs)
                # filter stopwords (case-insensitive) and deduplicate preserving order
                seen = set()
                terms = []
                for t in raw_terms:
                    if not t:
                        continue
                    key = t.lower()
                    if key in STOPWORDS:
                        continue
                    if key in seen:
                        continue
                    seen.add(key)
                    terms.append(t)
            else:
                terms = []

            # include vectors if requested
            if include_vectors and "vector" in item and item["vector"] is not None:
                item["vector"] = list(item["vector"])

            # Build new order: put 'id', then 'terms', then 'payload'
            new_item = {}
            if 'id' in item:
                new_item['id'] = item['id']
            if extract_terms:
                new_item['terms'] = terms
            if 'payload' in item:
                new_item['payload'] = item['payload']
            # copy remaining keys (vector, etc.)
            for k, v in item.items():
                if k in ('id', 'payload'):
                    continue
                new_item[k] = v

            serializable.append(new_item)
        json.dump(serializable, jf, ensure_ascii=False, indent=2)
    print(f"Wrote JSON to {json_path}")

    # Write SQLite (unchanged behavior, does not include 'terms' column)
    db_path = os.path.join(out_dir, f"{collection_name}_entries.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS entries (
            id TEXT PRIMARY KEY,
            time_stamp TEXT,
            float_time_stamp REAL,
            weekday TEXT,
            category TEXT,
            subcategory TEXT,
            memory_class TEXT,
            memory TEXT,
            original_memory TEXT,
            compressed_memory TEXT,
            hit_time INTEGER,
            topic_id INTEGER,
            topic_summary TEXT,
            update_queue TEXT,
            payload_json TEXT,
            vector BLOB
        )
        """
    )

    # Ensure newer columns (added in later schema versions) exist so INSERT statements
    # that refer to them won't fail on older DB files. Use ALTER TABLE ADD COLUMN
    # which is safe in SQLite (adds NULLable columns).
    def ensure_columns(cursor, table: str = 'entries', required: List[tuple] = None):
        if required is None:
            required = [('topic_id', 'INTEGER'), ('topic_summary', 'TEXT')]
        cursor.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}  # name is at index 1
        for col_name, col_type in required:
            if col_name not in existing:
                try:
                    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                    print(f"Added missing column '{col_name}' to table '{table}'")
                except Exception as e:
                    print(f"Warning: failed to add column {col_name} to {table}: {e}")

    ensure_columns(cur, 'entries', [('topic_id', 'INTEGER'), ('topic_summary', 'TEXT')])

    insert_sql = """
    INSERT OR REPLACE INTO entries (
        id, time_stamp, float_time_stamp, weekday, category, subcategory,
        memory_class, memory, original_memory, compressed_memory, hit_time, 
        topic_id, topic_summary, update_queue, payload_json, vector
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    for p in points:
        pid = str(p.get("id"))
        payload = p.get("payload", {}) or {}
        time_stamp = payload.get("time_stamp")
        float_time_stamp = payload.get("float_time_stamp")
        weekday = payload.get("weekday")
        category = payload.get("category")
        subcategory = payload.get("subcategory")
        memory_class = payload.get("memory_class")
        memory = payload.get("memory")
        original_memory = payload.get("original_memory")
        compressed_memory = payload.get("compressed_memory")
        hit_time = payload.get("hit_time")
        topic_id = payload.get("topic_id")
        topic_summary = payload.get("topic_summary")
        update_queue = json.dumps(payload.get("update_queue", []), ensure_ascii=False)
        payload_json = json.dumps(payload, ensure_ascii=False)

        vector_blob = None
        if include_vectors:
            vec = p.get("vector")
            if vec is not None:
                # store as JSON bytes for simplicity
                vector_blob = json.dumps(list(vec)).encode("utf-8")

        cur.execute(insert_sql, (
            pid, time_stamp, float_time_stamp, weekday, category, subcategory,
            memory_class, memory, original_memory, compressed_memory, hit_time,
            topic_id, topic_summary, update_queue, payload_json, vector_blob
        ))

    conn.commit()
    conn.close()
    print(f"Wrote SQLite DB to {db_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, help="Qdrant collection name (e.g., question_id)")
    parser.add_argument("--qdrant-path", default="./qdrant_data", help="Path to qdrant data directory")
    parser.add_argument("--out-dir", default="./exports", help="Output directory for exports")
    parser.add_argument("--include-vectors", action="store_true", help="Include vectors in exports (JSON + SQLite)")
    parser.add_argument("--extract-terms", action="store_true", help="Extract combined terms (NER+nouns+verbs) into 'terms' field")
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model to use for term extraction")
    parser.add_argument("--exclude-verbs", action="store_true", help="Exclude verbs from extracted terms")
    args = parser.parse_args()

    include_verbs = not args.exclude_verbs
    export_collection(args.collection, args.qdrant_path, args.out_dir, include_vectors=args.include_vectors,
                      extract_terms=args.extract_terms, spacy_model=args.spacy_model,
                      include_verbs=include_verbs)


if __name__ == '__main__':
    main()