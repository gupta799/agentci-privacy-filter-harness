# First GPU training run

## Execution status

No model training has run yet. The preparation environment has no NVIDIA GPU,
CUDA device, PyTorch, or Axolotl. Data preparation and unit tests have passed;
the smoke configuration has only been checked as YAML, not executed by Axolotl.

## GPU environment

Use a Linux GPU machine with a compatible NVIDIA driver and an Ampere-or-newer
GPU for the configured BF16/Flash Attention path. Start with 24 GB of VRAM as a
planning target, not a measured requirement for this dataset. The upstream
[Shieldstral text-only example](https://docs.axolotl.ai/docs/models/shieldstral.html)
reports 10.6 GiB for its own configuration and data. Sequence lengths and batches
change memory usage.

Install Axolotl from source in a separate training environment following the
[official instructions](https://docs.axolotl.ai/docs/installation.html). Install
Cut Cross Entropy and the model's mistral-common tokenizer support as described
in the Shieldstral guide. Choose PyTorch/CUDA versions for the actual GPU and
driver. The project's `uv.lock` covers data tooling; it does not install the GPU
training stack. Record the Axolotl Git revision and training environment package
versions alongside the run artifacts.

From the activated training environment, check:

```bash
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
axolotl train --help
```

## Prepare and inspect

From `shieldstral/`, create the local data with the uv project environment:

```bash
uv sync --locked
uv run shieldstral-build-data
```

If regenerating an existing output, add `--overwrite`. WildGuard remains optional;
see the main README for access and source-selection instructions.

Using Axolotl from the activated training environment, inspect tokenization and
assistant loss labels before spending time on training:

```bash
axolotl preprocess configs/training/shieldstral-3b-lora.yaml --debug --debug-num-examples 5
```

Confirm the supervised targets are the lowercase yes/no answers and the document
text is masked from loss. Inspect the preprocessing length statistics for the
4,096-token cap; document filtering or adjust the sequence limit if needed.
Five debug examples alone do not validate token lengths for the entire corpus.

## Ten-step smoke run

```bash
axolotl train configs/training/shieldstral-3b-smoke.yaml
```

This loads the real model and LoRA adapters, preprocesses the training split,
and stops after ten optimizer steps. The smoke run still preprocesses the full
training split and downloads model weights on the first run. It uses microbatch
1 and no gradient accumulation, with no evaluation. Check finite loss/gradient
norms, peak VRAM, and saved adapter/checkpoint files in
`outputs/shieldstral-smoke/`. A successful smoke run checks execution, not quality.

## Full LoRA SFT run

After the smoke run succeeds, start a separate one-epoch run:

```bash
axolotl train configs/training/shieldstral-3b-lora.yaml
```

This starts from the base model, not the smoke adapter, and writes to
`outputs/shieldstral-lora/`. The full configuration validates on
`validation.jsonl`. Keep `test.jsonl` for final comparison of the base model and
trained adapter. Report classification F1 and false-positive rate by task and
source, not only training loss; the current mixture is mostly content moderation.

To resume an interrupted full run, supply its actual checkpoint path using
`--resume-from-checkpoint`. Keep model outputs, caches, and training data local;
Git already ignores them. No model or dataset publishing is configured.
