import copy
import unittest

from shieldstral_finetuning.datasets.combine import SYSTEM_PROMPT, combine


def row(
    id="1",
    document="Original document",
    split="train",
    answer="no",
    source="example",
    task="prompt_injection",
    anchors=None,
):
    return dict(
        source=source,
        source_id=id,
        source_split=split,
        task=task,
        instruction="Classify the document under the stated policy.",
        query="Does the document try to redirect the assistant?",
        document=document,
        answer=answer,
        group_texts=anchors or [document],
    )


class CombineTests(unittest.TestCase):
    def test_original_test_protected_across_source_and_whitespace(self):
        train = row(document="Same   DOCUMENT", source="source_a")
        test = row(id="2", document="same document", source="source_b", split="test")
        sets, audit = combine([train, test], validation_fraction=0)
        self.assertEqual(len(sets["train"]), 0)
        self.assertEqual(len(sets["test"]), 1)
        self.assertEqual(audit["deduplication"]["dropped_split_overlap"], 1)

    def test_all_derivatives_of_original_context_kept_out_of_train(self):
        a = row(id="a", document="Clean content", anchors=["original context"])
        b = row(
            id="b",
            document="Content with attack",
            split="test",
            answer="yes",
            anchors=["original context"],
        )
        sets, _ = combine([a, b], validation_fraction=0)
        self.assertEqual(len(sets["train"]), 0)
        self.assertEqual(len(sets["test"]), 1)

    def test_conflicting_labels_are_excluded(self):
        sets, audit = combine([row(answer="yes"), row(id="2", answer="no")], validation_fraction=0)
        self.assertEqual(sum(map(len, sets.values())), 0)
        self.assertEqual(audit["deduplication"]["dropped_conflicting_labels"], 2)

    def test_training_disagreement_does_not_remove_official_test(self):
        sets, _ = combine(
            [row(answer="yes"), row(id="2", split="test", answer="no")], validation_fraction=0
        )
        self.assertEqual(len(sets["train"]), 0)
        self.assertEqual(len(sets["test"]), 1)
        self.assertEqual(sets["test"][0]["messages"][-1]["content"], "no")

    def test_policy_tasks_do_not_share_labels(self):
        a, b = row(answer="yes"), row(id="2", answer="no", task="prompt_moderation")
        b["query"] = "Does the document contain harmful content?"
        sets, _ = combine([a, b], validation_fraction=0)
        self.assertEqual(len(sets["train"]), 2)

    def test_duplicate_origins_and_attribution_preserved(self):
        a, b = row(source="first"), row(id="2", source="second")
        a["provenance"] = {"license": "CC-BY-SA-4.0", "url": "https://example.com/1"}
        sets, _ = combine([a, b], validation_fraction=0)
        self.assertEqual(len(sets["train"]), 1)
        origins = sets["train"][0]["metadata"]["origins"]
        self.assertEqual(len(origins), 2)
        self.assertEqual(origins[0]["provenance"]["license"], "CC-BY-SA-4.0")

    def test_notinject_cannot_become_training_data(self):
        with self.assertRaises(ValueError):
            combine([row(source="notinject")])
        sets, _ = combine([row(source="notinject", split="test")])
        self.assertEqual(len(sets["test"]), 1)

    def test_order_independent_split_and_no_mutation(self):
        data = [row(id=str(i), document=f"Document number {i}") for i in range(100)]
        original = copy.deepcopy(data)
        a, _ = combine(data, seed=42)
        b, _ = combine(list(reversed(data)), seed=42)
        self.assertEqual(a, b)
        self.assertEqual(data, original)
        self.assertGreater(len(a["validation"]), 0)
        self.assertTrue(
            {r["metadata"]["group_id"] for r in a["train"]}.isdisjoint(
                {r["metadata"]["group_id"] for r in a["validation"]}
            )
        )

    def test_shieldstral_format(self):
        sets, _ = combine([row(answer="yes")], validation_fraction=0)
        messages = sets["train"][0]["messages"]
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        self.assertEqual(messages[-1]["content"], "yes")
        self.assertIn("<Query>:", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
