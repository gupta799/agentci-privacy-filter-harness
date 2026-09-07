import unittest

from shieldstral_finetuning.datasets.sources.huggingface import _source_files, normalize_hf_rows


class SourceMappingTests(unittest.TestCase):
    def test_aegis_response_label_is_not_prompt_label(self):
        rows, _ = normalize_hf_rows(
            "aegis",
            [
                dict(
                    prompt="A neutral question",
                    response="An unsafe response",
                    prompt_label="safe",
                    response_label="unsafe",
                )
            ],
            "train",
            "train.json",
        )
        self.assertEqual(
            {r["task"]: r["answer"] for r in rows},
            {"prompt_moderation": "no", "response_moderation": "yes"},
        )
        self.assertNotIn("prompt_injection", {r["task"] for r in rows})

    def test_refusal_is_not_harmfulness(self):
        rows, _ = normalize_hf_rows(
            "wildguard",
            [
                dict(
                    prompt="An unsafe request",
                    response="I cannot help with that.",
                    prompt_harm_label="harmful",
                    response_harm_label="unharmful",
                    response_refusal_label="refusal",
                )
            ],
            "train",
            "train.parquet",
        )
        self.assertEqual(
            {r["task"]: r["answer"] for r in rows},
            {"prompt_moderation": "yes", "response_moderation": "no", "refusal": "yes"},
        )

    def test_redacted_data_and_missing_labels_are_skipped(self):
        rows, skipped = normalize_hf_rows(
            "aegis",
            [dict(prompt="REDACTED", response=None, prompt_label="safe", response_label=None)],
            "train",
            "train.json",
        )
        self.assertEqual(rows, [])
        self.assertEqual(skipped["redacted_prompt_rows"], 1)
        rows, _ = normalize_hf_rows(
            "wildguard",
            [
                dict(
                    prompt="An ordinary prompt",
                    response=None,
                    prompt_harm_label="unharmful",
                    response_harm_label=None,
                    response_refusal_label=None,
                )
            ],
            "train",
            "train.parquet",
        )
        self.assertEqual(len(rows), 1)

    def test_unknown_label_fails_instead_of_becoming_negative(self):
        with self.assertRaises(ValueError):
            normalize_hf_rows(
                "deepset", [{"text": "Some content", "label": 7}], "train", "file.parquet"
            )

    def test_stale_notinject_training_files_are_never_selected(self):
        names = ["data/train-00000.parquet", "data/test-00000.parquet"] + [
            f"data/NotInject_{n}-00000.parquet" for n in ("one", "two", "three")
        ]
        selected = _source_files("notinject", names)
        self.assertEqual(len(selected), 3)
        self.assertTrue(
            all(split == "test" and "NotInject_" in filename for filename, split in selected)
        )


if __name__ == "__main__":
    unittest.main()
