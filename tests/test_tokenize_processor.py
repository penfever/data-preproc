"""Tests for the TokenizeProcessor."""

from datasets import Dataset

from data_preproc.processors.tokenize import TokenizeProcessor


class _StubTokenizer:
    """Tiny tokenizer stub that splits on spaces."""

    name_or_path = "stub-tokenizer"

    def encode(self, text, add_special_tokens=False, **kwargs):  # noqa: D401, D403
        return [index + 1 for index, _ in enumerate(text.split())]

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        **kwargs,
    ):  # noqa: D401, D403
        tokens = self.encode(text, add_special_tokens=add_special_tokens)
        result = {"input_ids": tokens}
        if return_attention_mask:
            result["attention_mask"] = [1] * len(tokens)
        return result


def test_tokenize_processor_adds_tokens():
    processor = TokenizeProcessor({"text_fields": ["text"]})
    processor.tokenizer = _StubTokenizer()

    example = {"text": "hello world"}
    processed = processor.process_example(example.copy())

    assert processed["input_ids"] == [1, 2]
    assert "attention_mask" in processed
    assert processed["attention_mask"] == [1, 2]


def test_tokenize_processor_handles_dataset_map():
    dataset = Dataset.from_list([
        {"text": "a b c"},
        {"text": "only two"},
    ])

    processor = TokenizeProcessor({"text_fields": ["text"], "add_special_tokens": False})
    processor.tokenizer = _StubTokenizer()

    tokenized_dataset = processor.apply_to_dataset(dataset)

    assert tokenized_dataset[0]["input_ids"] == [1, 2, 3]
    assert tokenized_dataset[1]["input_ids"] == [1, 2]
