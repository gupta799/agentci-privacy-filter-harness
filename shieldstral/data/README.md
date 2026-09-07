# Local training data

Run `uv run shieldstral-build-data` from `shieldstral/` to download source datasets,
normalize labels, deduplicate records, protect evaluation splits, and write the
training-ready chat JSONL to `data/combined/`.

Downloaded and generated data, manifests, and copied upstream notices stay local.
They are ignored by Git. Source revisions and input hashes live in
`configs/data/sources.lock.json`; the build records output checksums and task/source
counts in `data/combined/manifest.json`.

`examples/sample.jsonl` contains four synthetic format examples only.
