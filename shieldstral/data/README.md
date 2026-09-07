# Data

`combined/` contains the actual compressed multi-source snapshot, source notices, checksums, and a dataset card. Run `uv run shieldstral-unpack-data` from the `shieldstral/` project directory to restore the JSONL files used by Axolotl.

`examples/sample.jsonl` contains only four synthetic format examples, kept for reference.

The combined dataset preserves task-specific policy questions and original source attribution. NotInject is evaluation-only. See `combined/DATASET_CARD.md` for the exact included source coverage and access limitations.
