"""Normalize examples, protect held-out documents, and export chat JSONL.

Only exact normalized-text overlap is detected. Semantic paraphrases and attack
families require additional auditing; this code does not claim to detect them.
"""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruction provided. Note that the answer can only be "yes" or "no".'
)
SPLIT_RANK = {"train": 0, "validation": 1, "test": 2}
TASKS = {"prompt_moderation", "response_moderation", "refusal", "prompt_injection"}


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def digest(text: str) -> str:
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def example_key(row: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            [
                row["task"],
                normalized(row["instruction"]),
                normalized(row["query"]),
                normalized(row["document"]),
            ],
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def validate(row: dict) -> None:
    for key in (
        "source",
        "source_id",
        "source_split",
        "task",
        "instruction",
        "query",
        "document",
        "answer",
    ):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Normalized record has missing/empty {key}")
    if row["source_split"] not in SPLIT_RANK:
        raise ValueError("Unsupported source split")
    if row["task"] not in TASKS or row["answer"] not in ("yes", "no"):
        raise ValueError("Unsupported task or answer")
    if not isinstance(row.get("group_texts"), list) or not row["group_texts"]:
        raise ValueError("Each record requires a document/conversation group anchor")
    if any(not isinstance(x, str) or not x.strip() for x in row["group_texts"]):
        raise ValueError("Invalid group anchor")
    if row["source"] == "notinject" and row["source_split"] != "test":
        raise ValueError("NotInject must remain evaluation-only")


def combine(records: list[dict], seed: int = 42, validation_fraction: float = 0.05):
    """Return output splits plus audit counts, without changing input records.

    Connected document groups cannot span output splits. Where original splits
    overlap, lower-priority records are discarded (test > validation > train).
    Original training examples are never promoted into an existing official test
    set. Train-only groups receive a deterministic validation assignment.
    """
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    rows = sorted(
        records,
        key=lambda r: (r["source"], r["source_id"], r["task"], r["source_split"], r["document"]),
    )
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    first = {}
    anchors_by_row = []
    for i, row in enumerate(rows):
        validate(row)
        anchors = sorted({digest(t) for t in row["group_texts"] + [row["document"]]})
        anchors_by_row.append(anchors)
        for anchor in anchors:
            if anchor in first:
                union(i, first[anchor])
            else:
                first[anchor] = i

    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    buckets = {s: [] for s in SPLIT_RANK}
    counters = Counter(input_records=len(rows), document_groups=len(groups))
    overlap_by_source = Counter()

    for indices in groups.values():
        group_id = min(anchor for i in indices for anchor in anchors_by_row[i])
        highest = max(SPLIT_RANK[rows[i]["source_split"]] for i in indices)
        original_split = next(s for s, rank in SPLIT_RANK.items() if rank == highest)
        target = original_split
        if target == "train":
            draw = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:16], 16) / 2**64
            if draw < validation_fraction:
                target = "validation"
        conflicts = defaultdict(set)
        for i in indices:
            if SPLIT_RANK[rows[i]["source_split"]] == highest:
                conflicts[example_key(rows[i])].add(rows[i]["answer"])
        conflict_keys = {k for k, labels in conflicts.items() if len(labels) > 1}
        counters["conflicting_example_keys"] += len(conflict_keys)
        unique = {}
        for i in indices:
            row = rows[i]
            if SPLIT_RANK[row["source_split"]] < highest:
                counters["dropped_split_overlap"] += 1
                overlap_by_source[row["source"]] += 1
                continue
            key = example_key(row)
            if key in conflict_keys:
                counters["dropped_conflicting_labels"] += 1
                continue
            origin = {k: row[k] for k in ("source", "source_id", "source_split")}
            if row.get("provenance"):
                origin["provenance"] = row["provenance"]
            if key in unique:
                counters["dropped_duplicates"] += 1
                if origin not in unique[key]["metadata"]["origins"]:
                    unique[key]["metadata"]["origins"].append(origin)
                continue
            unique[key] = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"<Instruct>: {row['instruction']}\n\n<Query>: {row['query']}"
                            f"\n\n<Document>: {row['document']}"
                        ),
                    },
                    {"role": "assistant", "content": row["answer"]},
                ],
                "metadata": {
                    "task": row["task"],
                    "answer": row["answer"],
                    "source": row["source"],
                    "source_id": row["source_id"],
                    "source_split": row["source_split"],
                    "split": target,
                    "document_sha256": digest(row["document"]),
                    "group_id": group_id,
                    "example_id": key,
                    "origins": [origin],
                    "provenance": row.get("provenance", {}),
                },
            }
        buckets[target].extend(unique.values())

    counts = {}
    for split, entries in buckets.items():
        entries.sort(key=lambda r: r["metadata"]["example_id"])
        random.Random(seed).shuffle(entries)
        counts[split] = {
            "records": len(entries),
            "by_source": dict(sorted(Counter(r["metadata"]["source"] for r in entries).items())),
            "by_task": dict(sorted(Counter(r["metadata"]["task"] for r in entries).items())),
            "by_task_answer": dict(
                sorted(
                    Counter(
                        r["metadata"]["task"] + ":" + r["metadata"]["answer"] for r in entries
                    ).items()
                )
            ),
        }
    return buckets, {
        "counts": counts,
        "deduplication": dict(counters),
        "split_overlap_drops_by_source": dict(overlap_by_source),
        "dedup_scope": "NFKC, casefold, whitespace-normalized exact text; no semantic deduplication",
    }


def export(buckets: dict, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in buckets.items():
        path = directory / f"{split}.jsonl"
        sha = hashlib.sha256()
        with path.open("wb") as stream:
            for row in rows:
                raw = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
                stream.write(raw)
                sha.update(raw)
        files[path.name] = {
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": sha.hexdigest(),
        }
    return files
