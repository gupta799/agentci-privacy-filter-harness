"""Download approved-access datasets and build Shieldstral chat JSONL."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from shieldstral_finetuning.datasets.combine import combine, export

SOURCES = ("aegis", "wildguard", "deepset", "bipia", "notinject")
DEFAULT_SOURCES = ("aegis", "deepset", "bipia", "notinject")


def capture_notices(statuses, cache_dir, stage):
    """Carry original source notices alongside the derived dataset."""
    for name, metadata in statuses.items():
        if metadata["status"] != "available":
            continue
        if name == "bipia":
            source_paths = [Path(p) for p in metadata.pop("cached_license_files", [])]
        else:
            base = cache_dir / metadata["repo_id"].replace("/", "--") / metadata["revision"]
            source_paths = [base / item["path"] for item in metadata.get("upstream_notices", [])]
        metadata["included_notices"] = []
        for source in source_paths:
            relative = Path("upstream-notices") / name / source.name
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            metadata["included_notices"].append(str(relative))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=SOURCES)
    parser.add_argument("--output", type=Path, default=Path("data/combined"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/datasets"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Explicitly allow a build with unavailable sources; manifest stays incomplete.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace generated output files.")
    parser.add_argument(
        "--source-lock",
        "--lock-manifest",
        dest="lock_manifest",
        type=Path,
        default=Path("configs/data/sources.lock.json"),
        help="Pin source revisions and hashes using a source lock or previous build manifest.",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.validation_fraction < 1:
        parser.error("--validation-fraction must be in [0, 1)")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        parser.error("Output directory is not empty; choose a new path or use --overwrite.")
    from shieldstral_finetuning.datasets.sources.bipia import fetch_bipia
    from shieldstral_finetuning.datasets.sources.huggingface import fetch_hf_source

    old = json.loads(args.lock_manifest.read_text())
    locked_metadata = {s["source"]: s for s in old["sources"]}
    locked = {name: s.get("revision") for name, s in locked_metadata.items()}
    requested = list(dict.fromkeys(args.sources or old.get("default_sources", DEFAULT_SOURCES)))
    if not requested or any(name not in SOURCES for name in requested):
        parser.error("The source lock must select at least one supported source.")
    if any(not locked.get(name) for name in requested):
        parser.error(
            "Every requested source must have a revision in --source-lock; use --sources to select pinned sources."
        )
    records, statuses = [], {}
    token = os.environ.get("HF_TOKEN")

    def fetch(name):
        revision = locked.get(name)
        if name == "bipia":
            result = fetch_bipia(args.cache_dir, revision=revision)
        else:
            result = fetch_hf_source(name, args.cache_dir, token=token, revision=revision)
        if name in locked_metadata:
            previous_files = locked_metadata[name].get("files", [])
            current_files = {f["path"]: f["sha256"] for f in result[1].get("files", [])}
            if previous_files and {f["path"] for f in previous_files} != set(current_files):
                raise ValueError(f"{name}: source file list differs from source lock")
            for entry in previous_files:
                if current_files.get(entry["path"]) != entry["sha256"]:
                    raise ValueError(
                        f"{name}: source file differs from locked manifest: {entry['path']}"
                    )
        return result

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch, name): name for name in requested}
        for future in as_completed(futures):
            name = futures[future]
            try:
                rows, metadata = future.result()
                records.extend(rows)
                statuses[name] = {
                    **metadata,
                    "source": name,
                    "status": "available",
                    "normalized_records": len(rows),
                }
                print(f"{name}: {len(rows):,} normalized examples", flush=True)
            except Exception as exc:
                # Do not print response bodies, token values, or headers.
                message = str(exc)
                if token:
                    message = message.replace(token, "[REDACTED]")
                statuses[name] = {
                    **getattr(exc, "metadata", {}),
                    "source": name,
                    "status": "unavailable",
                    "error_type": type(exc).__name__,
                    "reason": message,
                    "revision": getattr(exc, "metadata", {}).get("revision") or locked.get(name),
                }
                print(f"{name}: unavailable ({type(exc).__name__})", flush=True)
    unavailable = [name for name in requested if statuses[name]["status"] != "available"]
    manifest = {
        "schema_version": 1,
        "built_at_utc": datetime.now(UTC).isoformat(),
        "complete": not unavailable,
        "requested_sources": requested,
        "unavailable_sources": unavailable,
        "sources": [statuses[name] for name in requested],
        "seed": args.seed,
        "validation_fraction_for_train_only_groups": args.validation_fraction,
        "label_semantics": "yes means the explicitly stated policy question is true; tasks are distinct",
        "training_run_performed": False,
        "sampling": "All available normalized rows before deduplication; no oversampling or task balancing.",
    }
    if unavailable and not args.allow_partial:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        status_path = args.output.parent / (args.output.name + "-source-status.json")
        status_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"Build stopped: {', '.join(unavailable)} unavailable. Details: {status_path}",
            file=sys.stderr,
        )
        print(
            "For WildGuard, accept the publisher terms and set HF_TOKEN. "
            "Use --allow-partial only if you want an explicitly incomplete dataset.",
            file=sys.stderr,
        )
        return 2
    if not records:
        print("No source records are available.", file=sys.stderr)
        return 2
    buckets, audit = combine(records, args.seed, args.validation_fraction)
    if not buckets["train"]:
        print("No training records remain; no dataset exported.", file=sys.stderr)
        return 2
    manifest.update(audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".shieldstral-build-", dir=args.output.parent) as temp:
        stage = Path(temp)
        manifest["files"] = export(buckets, stage)
        capture_notices(statuses, args.cache_dir, stage)
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        args.output.mkdir(parents=True, exist_ok=True)
        for file in sorted(stage.rglob("*"), key=lambda p: p.name == "manifest.json"):
            if file.is_file():
                destination = args.output / file.relative_to(stage)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(file, destination)
    for split, counts in manifest["counts"].items():
        print(f"{split}: {counts['records']:,} examples")
    if unavailable:
        print(f"INCOMPLETE: omitted {', '.join(unavailable)}; see manifest.json.")
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
