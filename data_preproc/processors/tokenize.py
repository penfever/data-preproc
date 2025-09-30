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
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            raise ValueError(
                "Tokenizer not available for TokenizeProcessor. Ensure pipeline loads one before running."
            )

        collected_text = self._collect_text(example)

        if not collected_text:
            if self.skip_if_empty:
                return example
            return None

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

        encoding = self.tokenizer(collected_text, **encoding_kwargs)

        example[self.output_field] = self._ensure_flat_list(encoding["input_ids"])

        if self.attention_mask_field and "attention_mask" in encoding:
            example[self.attention_mask_field] = self._ensure_flat_list(
                encoding["attention_mask"]
            )

        if self.return_token_type_ids and "token_type_ids" in encoding:
            example["token_type_ids"] = self._ensure_flat_list(encoding["token_type_ids"])

        if not self.keep_text_fields:
            for field in self.text_fields:
                example.pop(field, None)

        return example

    def _collect_text(self, example: Dict[str, Any]) -> str:
        parts: List[str] = []
        for field in self.text_fields:
            value = example.get(field)
            if value is None:
                continue

            if isinstance(value, str):
                if value:
                    parts.append(value)
            elif isinstance(value, list):
                string_parts = [str(item) for item in value if isinstance(item, str)]
                if string_parts:
                    parts.append(self.join_with.join(string_parts))
            else:
                parts.append(str(value))

        return self.join_with.join(part for part in parts if part).strip()

    @staticmethod
    def _ensure_flat_list(value: Any) -> List[int]:
        if isinstance(value, list) and value and isinstance(value[0], list):
            return value[0]
        return list(value) if isinstance(value, (list, tuple)) else [int(value)]

    def get_required_columns(self) -> List[str]:
        # Allow processor to run even if some fields are missing; _collect_text handles it.
        return []


register_processor("tokenize", TokenizeProcessor)
