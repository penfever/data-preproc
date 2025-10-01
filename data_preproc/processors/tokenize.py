"""Processor that materializes tokenized fields using the configured tokenizer."""

from typing import Any, Dict, List, Optional

from . import DatasetProcessor, register_processor


class TokenizeProcessor(DatasetProcessor):
    """Tokenize raw text fields into `input_ids` (and optional masks) in-place."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.text_fields: List[str] = config.get("text_fields", ["text"])
        self.output_field: str = config.get("output_field", "input_ids")
        self.attention_mask_field: Optional[str] = config.get(
            "attention_mask_field", "attention_mask"
        )
        self.add_special_tokens: bool = config.get("add_special_tokens", False)
        self.truncation: Optional[bool] = config.get("truncation")
        self.max_length: Optional[int] = config.get("max_length")
        self.padding: Optional[Any] = config.get("padding", False)
        self.join_with: str = config.get("join_with", "\n\n")
        self.skip_if_empty: bool = config.get("skip_if_empty", True)
        self.keep_text_fields: bool = config.get("keep_text_fields", True)
        self.return_token_type_ids: bool = config.get("return_token_type_ids", False)

        self.tokenizer = config.get("tokenizer")  # Will be provided by framework

    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Single-example helper used only for map()."""
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            raise ValueError(
                "Tokenizer not available for TokenizeProcessor. Ensure pipeline loads one before running."
            )

        text = self._collect_text_from_example(example)

        if not text and self.skip_if_empty:
            text = ""

        encoding = self._tokenize_texts([text])

        example[self.output_field] = encoding[self.output_field][0]

        if self.attention_mask_field and self.attention_mask_field in encoding:
            example[self.attention_mask_field] = encoding[self.attention_mask_field][0]

        if self.return_token_type_ids and "token_type_ids" in encoding:
            example["token_type_ids"] = encoding["token_type_ids"][0]

        if not self.keep_text_fields:
            for field in self.text_fields:
                example.pop(field, None)

        return example

    def apply_to_dataset(self, dataset):  # type: ignore[override]
        """Apply tokenization using dataset.map to preserve dataset features."""
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            raise ValueError(
                "Tokenizer not available for TokenizeProcessor. Ensure pipeline loads one before running."
            )

        remove_columns: List[str] = []
        if not self.keep_text_fields:
            remove_columns = [
                field for field in self.text_fields if field in dataset.column_names
            ]

        def map_fn(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
            size = self._batch_size(batch)
            texts = [self._collect_text_from_batch(batch, idx) for idx in range(size)]

            encoding = self._tokenize_texts(texts)

            result: Dict[str, Any] = {
                self.output_field: encoding[self.output_field],
            }

            if self.attention_mask_field and self.attention_mask_field in encoding:
                result[self.attention_mask_field] = encoding[self.attention_mask_field]

            if self.return_token_type_ids and "token_type_ids" in encoding:
                result["token_type_ids"] = encoding["token_type_ids"]

            return result

        mapped_dataset = dataset.map(
            map_fn,
            batched=True,
            desc="Tokenizing examples",
            remove_columns=remove_columns if remove_columns else None,
        )

        return mapped_dataset

    def _collect_text_from_example(self, example: Dict[str, Any]) -> str:
        parts: List[str] = []
        for field in self.text_fields:
            value = example.get(field)
            text = self._normalize_text_value(value)
            if text:
                parts.append(text)

        return self.join_with.join(parts).strip()

    def _collect_text_from_batch(self, batch: Dict[str, List[Any]], index: int) -> str:
        parts: List[str] = []
        for field in self.text_fields:
            column = batch.get(field)
            if column is None:
                continue
            value = column[index]
            text = self._normalize_text_value(value)
            if text:
                parts.append(text)

        return self.join_with.join(parts).strip()

    def _normalize_text_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            string_parts = [
                item for item in value if isinstance(item, str) and item is not None
            ]
            return self.join_with.join(string_parts)
        return str(value)

    def _tokenize_texts(self, texts: List[str]) -> Dict[str, List[List[int]]]:
        encoding_kwargs = {
            "add_special_tokens": self.add_special_tokens,
            "return_attention_mask": self.attention_mask_field is not None,
        }

        if self.truncation is not None:
            encoding_kwargs["truncation"] = self.truncation
        if self.max_length is not None:
            encoding_kwargs["max_length"] = self.max_length
        if self.padding:
            encoding_kwargs["padding"] = self.padding
        if self.return_token_type_ids:
            encoding_kwargs["return_token_type_ids"] = True

        encoding = self.tokenizer(texts, **encoding_kwargs)

        result: Dict[str, List[List[int]]] = {
            self.output_field: [self._ensure_flat_list(ids) for ids in encoding["input_ids"]]
        }

        if self.attention_mask_field and "attention_mask" in encoding:
            result[self.attention_mask_field] = [
                self._ensure_flat_list(mask) for mask in encoding["attention_mask"]
            ]

        if self.return_token_type_ids and "token_type_ids" in encoding:
            result["token_type_ids"] = [
                self._ensure_flat_list(tt_ids) for tt_ids in encoding["token_type_ids"]
            ]

        return result

    @staticmethod
    def _batch_size(batch: Dict[str, List[Any]]) -> int:
        for value in batch.values():
            if isinstance(value, list):
                return len(value)
        return 0

    @staticmethod
    def _ensure_flat_list(value: Any) -> List[int]:
        if isinstance(value, list) and value and isinstance(value[0], list):
            return value[0]
        return list(value) if isinstance(value, (list, tuple)) else [int(value)]

    def get_required_columns(self) -> List[str]:
        # Allow processor to run even if some fields are missing; _collect_text handles it.
        return []


register_processor("tokenize", TokenizeProcessor)
