# Shieldstral fine-tuning

Text-only LoRA SFT for [`mistralai/Shieldstral-1.0-3B`](https://huggingface.co/mistralai/Shieldstral-1.0-3B), with a combined, policy-conditioned dataset.

**Included snapshot:** 35,675 training / 3,730 validation / 3,582 test examples. These are normalized real source records, not the four synthetic format examples. The snapshot includes Aegis, deepset, BIPIA's accessible EmailQA/TableQA/CodeQA contexts, and evaluation-only NotInject. **WildGuardMix is not included: upstream access is gated.** The builder supports adding it using an authorized account. GPU training has not been run.

## Use the included dataset

From the repository root, enter `shieldstral/` before running any commands below:

```bash
cd shieldstral
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then install the locked project dependencies and restore the compressed dataset:

```bash
uv sync --locked
uv run shieldstral-unpack-data
```

The training archive is stored in two compressed parts listed in the manifest; the unpack command joins them in order and verifies the original checksum.

This creates `data/combined/train.jsonl`, `validation.jsonl`, and `test.jsonl`. No source-dataset download or API token is needed for this included snapshot. See [dataset card](data/combined/DATASET_CARD.md) for source counts, licenses, labels, and limitations.

The uv project locks the dataset tooling and development tools. Install Axolotl in a separate CUDA-compatible GPU environment from source following its [installation guide](https://docs.axolotl.ai/docs/installation.html) and [Shieldstral requirements](https://docs.axolotl.ai/docs/models/shieldstral.html), including Flash Attention 2, Cut Cross Entropy, and the mistral-common tokenizer. Then run from this directory:

```bash
axolotl train configs/training/shieldstral-3b-lora.yaml
```

The configuration trains on the assistant answer and uses the already-grouped validation file. Axolotl calls this input `test_datasets`; it points to **validation.jsonl**, never the final test file. GPU memory depends on sequence length and batch size. Review examples that exceed the configured 4,096-token limit before a full training run; tokenization/truncation has not been measured here.

## Rebuild or add WildGuard

```bash
uv sync --locked
uv run shieldstral-build-data --output data/rebuilt
```

The default requests all five sources and stops if any are unavailable. For WildGuard, first accept the publisher's terms on [its dataset page](https://huggingface.co/datasets/allenai/wildguardmix), then set `HF_TOKEN` in your local environment using a token from that authorized account. Do not commit the token. No access terms are accepted automatically.

For an explicitly partial build using accessible sources:

```bash
uv run shieldstral-build-data --allow-partial --output data/rebuilt
```

To replay the included snapshot's source revisions and compare cached source hashes:

```bash
uv run shieldstral-build-data --lock-manifest data/combined/manifest.json --allow-partial --output data/replayed
```

Use `--sources aegis wildguard deepset bipia notinject` to choose sources. Existing output requires explicit `--overwrite`. Only `data/combined` is allowlisted for committing compressed data; other data directories stay ignored. A rebuild in another directory needs its paths applied to the training configuration.

## What is combined

Every JSONL example contains `messages` plus source/task/provenance metadata. Instructions and questions preserve separate targets for prompt moderation, response moderation, refusal detection, and prompt injection. **An unsafe-content label is never converted into a prompt-injection label.** A `yes` answer means the specific question is true, including refusal when that task is present.

The builder preserves original evaluation splits and groups exact matching documents and their derivatives. Test takes precedence over validation, which takes precedence over training; overlapping lower-priority records are dropped. Train-only document groups receive a deterministic validation assignment. Exact conflicting retained labels are excluded. NotInject remains test-only. Semantic paraphrase/translation leakage is not detected.

All retained rows are included without task balancing. This snapshot is mostly content moderation; it is not an optimized mixture for an injection-only model. Missing/redacted annotations are skipped, not reconstructed or relabeled. Public-source exposure during Shieldstral's original training is unknown, so keep a fresh deployment-like test set as well.

## Validation

```bash
uv run ruff check .
uv run ruff format --check .
uv run python -m unittest discover -s tests -v
```

Checks cover source-label mapping, missing annotations, exact duplicate handling, split isolation, conflicting labels, and provenance. The snapshot manifest includes source revisions, hashes, skipped counts, and task/label distributions. Data preparation is validated; no training-quality improvement or benchmark score is claimed.

## Sources and licensing

[Dataset card](data/combined/DATASET_CARD.md), [HF adapter details](docs/datasets/huggingface.md), [BIPIA adaptation](docs/datasets/bipia.md), and [third-party notices](THIRD_PARTY_NOTICES.md) document the source terms. Data retain their respective licenses; the combined snapshot is not covered by a single blanket code license.

## Project layout

| Path | Contents |
| --- | --- |
| `src/shieldstral_finetuning/cli/` | Dataset-building and unpacking commands |
| `src/shieldstral_finetuning/datasets/` | Normalization, grouping, split protection, export |
| `src/shieldstral_finetuning/datasets/sources/` | Hugging Face and BIPIA source adapters |
| `configs/training/` | Axolotl training configurations |
| `data/` | Compressed snapshot, source notices, and examples |
| `docs/datasets/` | Source schemas, licensing, and adaptation details |
| `tests/` | Unit tests for mappings and split integrity |
| `pyproject.toml`, `uv.lock`, `ruff.toml` | Project metadata, locked dependencies, lint/format rules |

## Git identity and repository

This project lives in [`gupta799/agentci-privacy-filter-harness/shieldstral`](https://github.com/gupta799/agentci-privacy-filter-harness/tree/main/shieldstral).

Use `Jaiydev Gupta <137670660+gupta799@users.noreply.github.com>` as both Git author and committer. See `AGENTS.md` for project conventions.
