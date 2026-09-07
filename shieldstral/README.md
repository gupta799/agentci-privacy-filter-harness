# Shieldstral fine-tuning

A reproducible data preparation pipeline and LoRA SFT configuration for
[`mistralai/Shieldstral-1.0-3B`](https://huggingface.co/mistralai/Shieldstral-1.0-3B).
The repository contains code, source version locks, and configuration. Downloaded
and generated datasets stay local and are ignored by Git.

## Prepare training data

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from
the repository root:

```bash
cd shieldstral
uv sync --locked
uv run shieldstral-build-data
```

This downloads **Aegis, deepset prompt-injections, BIPIA, and NotInject** at the
revisions in `configs/data/sources.lock.json`. Raw files are cached in
`.cache/datasets/`. The command creates:

| Local output | Purpose |
| --- | --- |
| `data/combined/train.jsonl` | Shieldstral chat examples for SFT |
| `data/combined/validation.jsonl` | Validation during training |
| `data/combined/test.jsonl` | Final held-out evaluation |
| `data/combined/manifest.json` | Source versions, hashes, counts, labels, and omissions |
| `data/combined/upstream-notices/` | Original source licenses and attribution |

The pipeline checks pinned source hashes, converts original annotations into
separate policy-conditioned tasks, skips missing/redacted annotations, removes
exact duplicates and conflicting labels, and protects original evaluation splits.
Connected document groups stay in one split. NotInject is evaluation-only.

Outputs contain messages, yes/no targets, task labels, and provenance. This is the
data engineering stage for fine-tuning; tokenization and loss masking happen in
Axolotl. It does not create tabular feature vectors or run GPU training.

## Include WildGuardMix

WildGuard is optional and requires approved access through
[the publisher](https://huggingface.co/datasets/allenai/wildguardmix). After accepting
the upstream terms, set `HF_TOKEN` locally using that authorized account, then run:

```bash
uv run shieldstral-build-data \
  --sources aegis wildguard deepset bipia notinject \
  --overwrite
```

A requested source that fails stops the build by default. `--allow-partial`
explicitly permits omissions and records them in the manifest. A complete manifest
means all **requested** sources succeeded; the default does not request WildGuard.
Tokens and datasets must never be committed.

## Reproducibility and source selection

Use `--sources` to select adapters, `--output` to choose the output directory, and
`--cache-dir` to reuse downloaded inputs. Existing output requires `--overwrite`.
The default split seed is 42; train-only document groups receive a deterministic
5% validation assignment. Official validation/test boundaries take precedence.

`--source-lock` accepts another source lock or a prior build manifest;
`--lock-manifest` remains an alias. Source revision changes must be reviewed and
hashes updated deliberately. BIPIA also verifies its pinned file hashes in code.

## Train

See the [first GPU run guide](docs/training/run.md) for environment setup,
preprocessing checks, the ten-step smoke run, and the full run.

Install Axolotl separately in a compatible CUDA environment using its
[installation guide](https://docs.axolotl.ai/docs/installation.html) and
[Shieldstral requirements](https://docs.axolotl.ai/docs/models/shieldstral.html).
Then run from `shieldstral/`:

```bash
axolotl train configs/training/shieldstral-3b-lora.yaml
```

The configuration uses assistant-only LoRA SFT. Axolotl's `test_datasets` setting
points to `validation.jsonl`; final `test.jsonl` stays unused during training.
Audit token lengths against the configured 4,096-token limit before a full run.
GPU training and token-length auditing have not been performed.

## Data preparation limits

Content moderation, response moderation, refusal, and prompt injection retain
separate questions and targets. Unsafe content is never relabeled as injection.
The default mixture is mostly content moderation; no task balancing is applied.
BIPIA uses accessible EmailQA/TableQA/CodeQA contexts; its WebQA and summarization
contexts are excluded. Exact normalized-text deduplication does not detect
semantic paraphrases or guarantee unseen attack families. Keep a fresh,
deployment-like evaluation set.

See [HF adapters](docs/datasets/huggingface.md),
[BIPIA adaptation](docs/datasets/bipia.md), and
[third-party notices](THIRD_PARTY_NOTICES.md) for source terms and label mappings.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
```

| Path | Contents |
| --- | --- |
| `src/shieldstral_finetuning/cli/` | Data preparation command |
| `src/shieldstral_finetuning/datasets/` | Normalization, grouping, and export |
| `src/shieldstral_finetuning/datasets/sources/` | Upstream download adapters |
| `configs/data/` | Pinned source revisions and checksums |
| `configs/training/` | Axolotl training configuration |
| `docs/datasets/` | Source schemas and adaptation details |
| `tests/` | Source, pipeline, and split-integrity checks |

All commits use Jaiydev Gupta's `gupta799` GitHub identity. See `AGENTS.md`.
