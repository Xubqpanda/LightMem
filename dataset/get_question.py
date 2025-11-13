#!/usr/bin/env python3
"""
get_question.py

Extract the `question` field from each item in a possibly-large JSON file.

Usage:
    python get_question.py /path/to/longmemeval_s_cleaned.json /path/to/longmemeval_questions.json

This script supports two input formats:
- A JSON array of objects (large file is streamed and parsed incrementally).
- JSONL (one JSON object per line).

The output is a JSON array of strings written atomically to the target file.
"""
import sys
import json
import os
import tempfile
from typing import Iterator


def iter_questions_from_json_array_file(file_path: str) -> Iterator[str]:
    decoder = json.JSONDecoder()
    with open(file_path, "r", encoding="utf-8") as fp:
        # Seek to the first '['
        while True:
            ch = fp.read(1)
            if not ch:
                return
            if ch.isspace():
                continue
            if ch == "[":
                break
            # unexpected char, keep it
            fp.seek(fp.tell() - 1)
            break

        buffer = ""
        while True:
            chunk = fp.read(65536)
            if not chunk:
                break
            buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == ",":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    return
                try:
                    obj, end = decoder.raw_decode(buffer)
                except ValueError:
                    # need more data
                    break
                else:
                    buffer = buffer[end:]
                    q = obj.get("question")
                    if q is not None:
                        yield q

        # final buffer
        buffer = buffer.strip()
        if buffer and buffer[0] not in (']', ','):
            try:
                obj, end = decoder.raw_decode(buffer)
                q = obj.get("question")
                if q is not None:
                    yield q
            except Exception:
                return


def iter_questions_from_jsonl_file(file_path: str) -> Iterator[str]:
    with open(file_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            q = item.get("question")
            if q is not None:
                yield q


def detect_and_iter(file_path: str) -> Iterator[str]:
    with open(file_path, "r", encoding="utf-8") as fp:
        head = fp.read(1024)
    if not head:
        return iter(())
    s = head.lstrip()
    if s.startswith("["):
        return iter_questions_from_json_array_file(file_path)
    return iter_questions_from_jsonl_file(file_path)


def main():
    if len(sys.argv) < 3:
        print("Usage: get_question.py <input.json|jsonl> <output.json>")
        sys.exit(2)
    inp = sys.argv[1]
    outp = sys.argv[2]

    it = detect_and_iter(inp)

    out_dir = os.path.dirname(outp) or "."
    tmpf = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=out_dir) as tf:
            tmpf = tf.name
            tf.write("[")
            first = True
            count = 0
            for q in it:
                if not first:
                    tf.write(",\n")
                else:
                    first = False
                tf.write(json.dumps(q, ensure_ascii=False))
                count += 1
            tf.write("]")
        os.replace(tmpf, outp)
        tmpf = None
        print(f"Wrote {count} questions to {outp}")
    finally:
        if tmpf and os.path.exists(tmpf):
            try:
                os.remove(tmpf)
            except Exception:
                pass


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Extract the `question` field from each item in a possibly-large JSON file.

Usage:
    python scripts/extract_questions.py \
        /path/to/longmemeval_s_cleaned.json \
        /path/to/longmemeval_questions.json

The script detects JSON array vs JSONL and streams the input to avoid
loading the entire file into memory.
"""
import sys
import json
from typing import Iterable


def iter_questions_from_json_array(fp) -> Iterable[str]:
    # Load iteratively: use json.load once but the file is large; try to
    # fallback to streaming by incremental parsing via a simple heuristic.
    data = json.load(fp)
    for item in data:
        q = item.get("question")
        if q is not None:
            yield q


def iter_questions_from_jsonl(fp) -> Iterable[str]:
    for line in fp:
        line = line.strip()
        if not line:
            continue
        #!/usr/bin/env python3
        """
        Extract the `question` field from each item in a possibly-large JSON file.

        Usage:
            python get_question.py /path/to/longmemeval_s_cleaned.json /path/to/longmemeval_questions.json

        The script detects JSON array vs JSONL and returns an iterator that opens
        the input file when iterated, so the file remains open for the duration
        of the iteration and avoids the "I/O operation on closed file" error.
        """
        import sys
        import json
        import os
        import tempfile
        from typing import Iterable, Iterator


        def iter_questions_from_json_array_file(file_path: str) -> Iterator[str]:
            # Stream-parse the JSON array using JSONDecoder.raw_decode so we don't
            # need to load the entire file into memory. This reads chunks and
            # decodes one object at a time.
            decoder = json.JSONDecoder()
            with open(file_path, "r", encoding="utf-8") as fp:
                # Skip whitespace until '['
                while True:
                    ch = fp.read(1)
                    if not ch:
                        return
                    if ch.isspace():
                        continue
                    if ch == "[":
                        break
                    # unexpected; rewind one char and proceed
                    fp.seek(fp.tell() - 1)
                    break

                buffer = ""
                while True:
                    chunk = fp.read(65536)
                    if not chunk:
                        break
                    buffer += chunk
                    while True:
                        buffer = buffer.lstrip()
                        if not buffer:
                            break
                        # consume leading commas or closing bracket
                        if buffer[0] == ",":
                            buffer = buffer[1:]
                            continue
                        if buffer[0] == "]":
                            return
                        try:
                            obj, end = decoder.raw_decode(buffer)
                        except ValueError:
                            # need more data
                            break
                        else:
                            buffer = buffer[end:]
                            q = obj.get("question")
                            if q is not None:
                                yield q

                # Final attempt to decode remaining buffer
                buffer = buffer.strip()
                if buffer and buffer[0] not in (']', ','):
                    try:
                        obj, end = decoder.raw_decode(buffer)
                        q = obj.get("question")
                        if q is not None:
                            yield q
                    except Exception:
                        return


        def iter_questions_from_jsonl_file(file_path: str) -> Iterator[str]:
            with open(file_path, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except Exception:
                        # skip malformed line
                        continue
                    q = item.get("question")
                    if q is not None:
                        yield q


        def detect_and_iter(file_path: str) -> Iterator[str]:
            # Detect JSON array vs JSONL by peeking the first non-whitespace char
            with open(file_path, "r", encoding="utf-8") as fp:
                first = fp.read(1024)
                if not first:
                    return iter(())
                s = first.lstrip()
                if s.startswith("["):
                    # JSON array
                    return iter_questions_from_json_array_file(file_path)
                else:
                    return iter_questions_from_jsonl_file(file_path)


        def main():
            if len(sys.argv) < 3:
                print("Usage: get_question.py <input.json|jsonl> <output.json>")
                sys.exit(2)
            inp = sys.argv[1]
            outp = sys.argv[2]

            it = detect_and_iter(inp)

            # Write out as JSON array streaming to a temp file then atomically
            # replace the target file so we don't leave a partial file on failure.
            out_dir = os.path.dirname(outp) or "."
            tmpf = None
            try:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=out_dir) as tf:
                    tmpf = tf.name
                    tf.write("[")
                    first = True
                    for q in it:
                        if not first:
                            tf.write(",\n")
                        else:
                            first = False
                        tf.write(json.dumps(q, ensure_ascii=False))
                    tf.write("]")
                os.replace(tmpf, outp)
                tmpf = None
                print(f"Wrote questions to {outp}")
            finally:
                if tmpf and os.path.exists(tmpf):
                    try:
                        os.remove(tmpf)
                    except Exception:
                        pass


        if __name__ == "__main__":
            main()
