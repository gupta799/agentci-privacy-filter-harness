# Third-party notices

`configs/training/shieldstral-3b-lora.yaml` is adapted from the Axolotl project:

https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/shieldstral/shieldstral-3b-lora.yaml

Axolotl is licensed under Apache License 2.0. A copy is included at `third_party/Apache-2.0.txt`. This adaptation changes the training dataset to a local JSONL path and removes unused optional fields.

The fixed system prompt in the sample data follows the Mistral Shieldstral model card:

https://huggingface.co/mistralai/Shieldstral-1.0-3B

The upstream model is licensed under Apache License 2.0. Model weights are not included. The sample documents and labels were created for this starter.

## Combined dataset

The preparation pipeline produces adapted records retaining separate source licenses. Upstream README, LICENSE, and NOTICE files available with each downloaded source are copied to the local `data/combined/upstream-notices/` output. Downloaded and generated datasets are not tracked in the current repository tree. Row metadata preserve source revisions, original identifiers, label metadata, and BIPIA attribution. Changes include message-format conversion, task expansion, BIPIA attack insertion, grouping, deduplication, and validation partitioning.

| Source | Terms recorded by the source | Modification / attribution |
| --- | --- | --- |
| NVIDIA Aegis 2.0 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Prompt and response labels converted into separate questions; NVIDIA source card retained. |
| deepset prompt-injections | Top-level Apache-2.0; nested metadata says CC BY 4.0 | Inconsistent source metadata retained explicitly; deepset attribution and both license references preserved. Apache-2.0 text is in `third_party/Apache-2.0.txt`. |
| BIPIA EmailQA and attack templates | MIT | Original BIPIA LICENSE and NOTICE retained; detector examples generated from clean context plus attack insertion. |
| BIPIA TableQA and CodeQA | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) according to BIPIA's source exceptions | These adapted records retain CC BY-SA 4.0; original source/author URLs preserved where supplied. |
| NotInject | MIT | All official benchmark rows retained as benign evaluation examples; source card retained. |
| WildGuardMix | [ODC-BY](https://opendatacommons.org/licenses/by/1-0/) and publisher access terms | Optional adapter; downloading requires an authorized account. |

BIPIA attributes WikiTableQuestions to Panupong Pasupat and Percy Liang and CodeQA data to the Stack Exchange archive. Its original per-record Stack Overflow URLs and author URLs are preserved. Two original CodeQA test contexts have no author URL in the source; the manifest records this omission. No author identity was invented. [BIPIA source license](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/LICENSE).
