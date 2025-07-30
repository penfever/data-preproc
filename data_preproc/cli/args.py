"""Module for data preprocessing CLI command arguments."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PreprocessCliArgs:
    """Dataclass with CLI arguments for preprocessing command."""

    debug: bool = field(default=False)
    debug_text_only: bool = field(default=False)
    debug_num_examples: int = field(default=1)
    prompter: Optional[str] = field(default=None)
    download: Optional[bool] = field(default=True)
    iterable: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Use IterableDataset for streaming processing of large datasets"
        },
    )
    hf_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "HuggingFace token for dataset upload (can also use HF_TOKEN env var)"
        },
    )
    limit: Optional[int] = field(
        default=None,
        metadata={
            "help": "Global limit on number of samples to process (enables streaming mode)"
        },
    )