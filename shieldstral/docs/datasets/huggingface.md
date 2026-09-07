# Hugging Face adapter sources

Verified against the upstream repository APIs, cards, and public downloads on 2026-09-07.
Every build resolves a commit SHA, downloads files at that SHA, and records SHA-256
file hashes. Pass `revision=` to `fetch_hf_source` to replay a recorded revision.
Upstream cards and any license/notice files are preserved beside raw downloads.

| Alias | Upstream | Label conversion | Split handling |
| --- | --- | --- | --- |
| `aegis` | [NVIDIA Aegis 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | `unsafe` → `yes`, `safe` → `no`, separately for prompt and response moderation | Preserve train, validation, test; include the two official refusal augmentation files |
| `wildguard` | [AI2 WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) | `harmful` → `yes`, `unharmful` → `no`; separately, `refusal` → `yes`, `compliance` → `no` | Preserve train/test; requires authorized account access |
| `deepset` | [deepset prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | `1` → injection `yes`, `0` → injection `no` | Preserve 546 train and 116 test rows before downstream deduplication |
| `notinject` | [NotInject](https://huggingface.co/datasets/leolee99/NotInject) | All 339 official benchmark examples → injection `no` | Evaluation only; never training |

The content-moderation labels are not relabeled as prompt injection. Each record
includes an explicit task, policy instruction, yes/no question, and provenance.
Response tasks contain both the prompt and response in a JSON document. Prompt
anchors keep task expansions and conversation variants grouped during splitting.
Missing annotations, empty text, and redacted text are counted and skipped; labels
are never imputed from other tasks. Unexpected schema or label values stop the build.

## Aegis

The official default configuration uses `train.json`, `refusals_train.json`,
`validation.json`, `refusals_validation.json`, and `test.json`. Upstream publishes
33,416 interactions before filtering and task expansion. Rows whose prompts are
`REDACTED` are excluded; this pipeline does not reconstruct them from other datasets.
Refusal augmentation contributes the provided moderation labels, not inferred
refusal labels. Source policy names and original annotation provenance are retained.
The upstream license is CC-BY-4.0; credit NVIDIA and the AEGIS2.0 authors.

## WildGuardMix

The public card documents independent `prompt_harm_label`, `response_harm_label`,
and `response_refusal_label` fields, with missing labels when annotations do not
agree. The source declares ODC-BY and requires acceptance of AI2's use conditions.
An unauthenticated request returned HTTP 401 during verification. This adapter
does not use mirrors or dataset-viewer endpoints to obtain gated content. Once
the user has accepted the upstream terms, an authorized `HF_TOKEN` permits the
ordinary repository download. This source was not downloaded during verification.

## deepset

The author's [model configuration](https://huggingface.co/deepset/deberta-v3-base-injection/blob/main/config.json)
confirms `0=LEGIT` and `1=INJECTION`. The dataset card's top-level license is
Apache-2.0, while its nested `dataset_info.license` says CC-BY-4.0. The manifest
retains this discrepancy; it does not silently declare a single license for a
merged corpus. Preserve attribution and consult the upstream project for resolution.

## NotInject

Only the card's three configured subsets, `NotInject_one`, `NotInject_two`, and
`NotInject_three`, are selected. Historical files named train/validation/test are
not selected again, preventing duplicate inclusion. The card identifies every
example as benign. The source declares MIT; credit Hao Li and Xiaogeng Liu and
retain the dataset card. This benchmark measures false positives, not attack recall.
