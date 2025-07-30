"""Tokenizer loading functionality"""

import os
from transformers import AutoTokenizer

from data_preproc.utils.logging import get_logger

LOG = get_logger(__name__)


def load_tokenizer(cfg):
    """Load and configure the tokenizer based on the provided config."""
    tokenizer_kwargs = {}
    use_fast = True  # this is the default

    if cfg.tokenizer_use_fast is not None:
        use_fast = cfg.tokenizer_use_fast
    if cfg.tokenizer_legacy is not None:
        tokenizer_kwargs["legacy"] = cfg.tokenizer_legacy

    # Use tokenizer_config if specified, otherwise use base_model
    tokenizer_path = cfg.tokenizer_config or cfg.base_model

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=cfg.trust_remote_code or False,
        use_fast=use_fast,
        **tokenizer_kwargs,
    )

    # Add special tokens configuration
    if cfg.special_tokens:
        for k, val in cfg.special_tokens.items():
            if k == "pad_token" and val == "</s>":
                continue
            # Set special tokens
            setattr(tokenizer, k, val)

    # Handle tokenizer padding side
    if cfg.tokenizer_pad_side:
        tokenizer.padding_side = cfg.tokenizer_pad_side
    elif tokenizer.padding_side != "left":
        LOG.warning(
            f"Tokenizer padding side is {tokenizer.padding_side}, not 'left'. "
            f"Consider setting tokenizer_pad_side: left in your config."
        )

    # Add tokens if specified
    if cfg.tokens:
        tokenizer.add_tokens(
            [
                token
                for token in cfg.tokens
                if token not in tokenizer.get_vocab()
            ]
        )

    # Set pad token if not set
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Log tokenizer configuration
    LOG.debug(f"EOS: {tokenizer.eos_token_id} / {tokenizer.eos_token}")
    LOG.debug(f"BOS: {tokenizer.bos_token_id} / {tokenizer.bos_token}")
    LOG.debug(f"PAD: {tokenizer.pad_token_id} / {tokenizer.pad_token}")
    LOG.debug(f"UNK: {tokenizer.unk_token_id} / {tokenizer.unk_token}")

    return tokenizer