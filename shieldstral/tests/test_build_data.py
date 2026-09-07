import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shieldstral_finetuning.cli.build_data import main


class BuildDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "combined"
        self.lock = self.root / "sources.lock.json"
        self.lock.write_text(
            json.dumps(
                {
                    "default_sources": ["deepset", "notinject"],
                    "sources": [
                        {
                            "source": source,
                            "revision": "pinned-revision",
                            "files": [{"path": "rows.json", "sha256": "expected"}],
                        }
                        for source in ("deepset", "notinject")
                    ],
                }
            )
        )

    def fetch(self, name, cache_dir, *, token, revision):
        split = "test" if name == "notinject" else "train"
        row = {
            "source": name,
            "source_id": "1",
            "source_split": split,
            "task": "prompt_injection",
            "instruction": "Detect instruction overrides.",
            "query": "Is this an instruction override?",
            "document": f"An ordinary question from {name}",
            "answer": "no",
            "group_texts": [f"An ordinary question from {name}"],
            "provenance": {"revision": revision},
        }
        metadata = {
            "repo_id": f"publisher/{name}",
            "revision": revision,
            "files": [{"path": "rows.json", "sha256": "expected"}],
        }
        return [row], metadata

    def run_build(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(
                [
                    "--source-lock",
                    str(self.lock),
                    "--output",
                    str(self.output),
                    "--cache-dir",
                    str(self.root / "cache"),
                    "--validation-fraction",
                    "0",
                    *extra,
                ]
            )

    def test_locked_downloads_produce_local_training_files_without_archives(self):
        with patch(
            "shieldstral_finetuning.datasets.sources.huggingface.fetch_hf_source",
            side_effect=self.fetch,
        ) as fetch:
            self.assertEqual(self.run_build(), 0)
        self.assertEqual({call.args[0] for call in fetch.call_args_list}, {"deepset", "notinject"})
        self.assertTrue(
            all(call.kwargs["revision"] == "pinned-revision" for call in fetch.call_args_list)
        )
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["counts"]["train"]["by_source"], {"deepset": 1})
        self.assertEqual(manifest["counts"]["test"]["by_source"], {"notinject": 1})
        self.assertEqual(len(list(self.output.glob("*.jsonl"))), 3)
        self.assertEqual(list(self.output.glob("*.gz")), [])
        row = json.loads((self.output / "train.jsonl").read_text())
        self.assertEqual(row["messages"][-1], {"role": "assistant", "content": "no"})
        self.assertEqual(row["metadata"]["provenance"]["revision"], "pinned-revision")

    def test_hash_mismatch_stops_build_and_preserves_existing_output(self):
        self.output.mkdir()
        existing = self.output / "train.jsonl"
        existing.write_text("previous data\n")

        def corrupt(*args, **kwargs):
            rows, metadata = self.fetch(*args, **kwargs)
            metadata["files"][0]["sha256"] = "changed"
            return rows, metadata

        with patch(
            "shieldstral_finetuning.datasets.sources.huggingface.fetch_hf_source",
            side_effect=corrupt,
        ):
            self.assertEqual(self.run_build("--overwrite"), 2)
        self.assertEqual(existing.read_text(), "previous data\n")
        self.assertFalse((self.output / "manifest.json").exists())

    def test_source_omission_requires_explicit_partial_build(self):
        def unavailable(name, *args, **kwargs):
            if name == "notinject":
                raise RuntimeError("Source unavailable")
            return self.fetch(name, *args, **kwargs)

        with patch(
            "shieldstral_finetuning.datasets.sources.huggingface.fetch_hf_source",
            side_effect=unavailable,
        ):
            self.assertEqual(self.run_build(), 2)
            self.assertFalse(self.output.exists())
            self.assertEqual(self.run_build("--allow-partial"), 0)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["unavailable_sources"], ["notinject"])


if __name__ == "__main__":
    unittest.main()
