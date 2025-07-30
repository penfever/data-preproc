"""Module containing PromptTokenizingStrategy and related classes"""

import abc
from typing import Callable, Dict, List, Optional

from transformers import BatchEncoding, PreTrainedTokenizer

from data_preproc.prompters import Prompter
from data_preproc.utils.logging import get_logger

LOG = get_logger(__name__)

IGNORE_INDEX = -100
LLAMA_DEFAULT_PAD_TOKEN = "<pad>"
LLAMA_DEFAULT_EOS_TOKEN = "</s>"
LLAMA_DEFAULT_BOS_TOKEN = "<s>"
LLAMA_DEFAULT_UNK_TOKEN = "<unk>"


class InvalidDataException(Exception):
    """Exception raised when the data is invalid"""
    pass


class PromptTokenizingStrategy(abc.ABC):
    """Abstract class for tokenizing strategies"""

    filter_rows: Optional[Callable] = None

    def __init__(
        self,
        prompter: Prompter,
        tokenizer: PreTrainedTokenizer,
        train_on_inputs: bool = False,
        sequence_len: int = 2048,
    ):
        self.prompter = prompter
        self.tokenizer = tokenizer
        self.train_on_inputs = train_on_inputs
        self.sequence_len = sequence_len
        self.max_length = sequence_len

    @abc.abstractmethod
    def tokenize_prompt(self, prompt):
        pass

    @property
    def supports_batched(self):
        return False

    def _tokenize(
        self, prompt: str, add_eos_token: bool = True, strip_bos_token: bool = False
    ) -> BatchEncoding:
        empty = BatchEncoding(data={"input_ids": [], "attention_mask": []})
        if not prompt:
            LOG.warning("Empty text requested for tokenization.")
            return empty

        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        
        if len(result["input_ids"]) == 0:
            LOG.warning("Tokenizer result is empty.")
            return empty

        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.max_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        if result["input_ids"][0] == self.tokenizer.bos_token_id and strip_bos_token:
            result["input_ids"] = result["input_ids"][1:]
            result["attention_mask"] = result["attention_mask"][1:]

        result["labels"] = result["input_ids"].copy()
        return result


class InstructionPromptTokenizingStrategy(PromptTokenizingStrategy):
    """Tokenizing strategy for instruction-based prompts."""

    def tokenize_prompt(self, prompt) -> Dict[str, List[int]]:
        instruction = self.prompter.build_prompt(
            prompt["instruction"],
            prompt.get("input", ""),
            prompt.get("output", ""),
        )
        tokenized = self._tokenize(instruction)
        
        if not self.train_on_inputs:
            # Mask the input portion
            user_prompt = self.prompter.build_prompt(
                prompt["instruction"],
                prompt.get("input", ""),
            )
            tokenized_user_prompt = self._tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            
            # Set labels to IGNORE_INDEX for input portion
            tokenized["labels"][:user_prompt_len] = [IGNORE_INDEX] * user_prompt_len
        
        return tokenized


class AlpacaPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    """Tokenizing strategy for Alpaca-style prompts."""
    pass


class CompletionPromptTokenizingStrategy(PromptTokenizingStrategy):
    """Tokenizing strategy for completion-style prompts."""

    def tokenize_prompt(self, prompt) -> Dict[str, List[int]]:
        text = prompt.get("text", "")
        tokenized = self._tokenize(text)
        return tokenized