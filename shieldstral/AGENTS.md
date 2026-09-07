# Repository conventions

- GitHub owner: `gupta799` (Jaiydev Gupta).
- Every commit must use `Jaiydev Gupta <137670660+gupta799@users.noreply.github.com>` for both author and committer. Do not add an assistant/bot co-author.
- Set this identity in repository-local Git configuration before committing. Do not change global Git settings.
- Use uv with `pyproject.toml` and the committed `uv.lock`. Run `uv sync --locked` to install dependencies.
- Keep Python code under `src/shieldstral_finetuning/`: CLI entry points in `cli/`, dataset logic in `datasets/`, upstream adapters in `datasets/sources/`.
- Put training configuration in `configs/training/`, source documentation in `docs/datasets/`, and tests in `tests/`.
- Run `uv run ruff check .`, `uv run ruff format --check .`, and `uv run python -m unittest discover -s tests -v` before committing code changes.
- Preserve dataset provenance, license notices, task-specific labels, and held-out split boundaries. Never commit tokens or uncompressed dataset files.
