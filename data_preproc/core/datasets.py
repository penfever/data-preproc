"""Dataset loading utilities."""

import random
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from datasets import Dataset

from data_preproc.cli.args import PreprocessCliArgs
from data_preproc.loaders import load_processor, load_tokenizer
from data_preproc.utils.data import prepare_dataset
from data_preproc.utils.dict import DictDefault
from data_preproc.utils.logging import get_logger
from data_preproc.utils.tokenization import check_dataset_labels

try:
    from data_preproc.utils.hf_upload import (
        should_upload_to_hf, 
        validate_hf_config, 
        upload_dataset_to_hf,
        HFUploadError
    )
    HAS_HF_UPLOAD = True
except ImportError:
    HAS_HF_UPLOAD = False

LOG = get_logger(__name__)


@dataclass
class TrainDatasetMeta:
    """Dataclass with fields for training and validation datasets and metadata."""

    train_dataset: Dataset
    eval_dataset: Optional[Dataset] = None
    total_num_steps: Optional[int] = None


def sample_dataset(dataset: Dataset, num_samples: int) -> Dataset:
    """
    Randomly sample `num_samples` samples from `dataset`.

    Args:
        dataset: Dataset.
        num_samples: Number of samples to return.

    Returns:
        Random sample (with replacement) of examples in `dataset`.
    """
    if dataset is None:
        raise ValueError("Cannot sample from None dataset. Dataset loading may have failed.")
    
    if len(dataset) == 0:
        raise ValueError("Cannot sample from empty dataset. All examples may have been filtered out.")
    
    if len(dataset) == 1:
        print("Only one sample in dataset, returning it repeated")
        # Special case: only one example, return it repeated
        return dataset.select([0] * min(num_samples, 1))
    
    return dataset.select(
        [random.randrange(0, len(dataset)) for _ in range(min(num_samples, len(dataset)))]
    )


def load_datasets(
    *,
    cfg: DictDefault,
    cli_args: Optional[PreprocessCliArgs] = None,
    debug: bool = False,
) -> TrainDatasetMeta:
    """
    Loads one or more training or evaluation datasets.

    Args:
        cfg: Dictionary mapping config keys to values.
        cli_args: Command-specific CLI arguments.
        debug: Whether to print out tokenization of sample

    Returns:
        Dataclass with fields for training and evaluation datasets.
    """
    tokenizer = load_tokenizer(cfg)
    processor = load_processor(cfg, tokenizer=tokenizer) if cfg.processor_type else None
    preprocess_iterable = (
        cli_args
        and hasattr(cli_args, "iterable")
        and cli_args.iterable is not None
        and cli_args.iterable
    )

    # Extract limit parameter from CLI args
    limit = cli_args.limit if cli_args and hasattr(cli_args, 'limit') else None
    
    train_dataset, eval_dataset, total_num_steps, prompters = prepare_dataset(
        cfg,
        tokenizer,
        processor=processor,
        preprocess_iterable=preprocess_iterable,
        limit=limit,
    )

    if (
        cli_args
        and (
            cli_args.debug
            or cfg.debug
            or cli_args.debug_text_only
            or int(cli_args.debug_num_examples) > 0
        )
    ) or debug:
        LOG.info("Checking dataset labels...")

        num_examples = cli_args.debug_num_examples if cli_args else 1
        text_only = cli_args.debug_text_only if cli_args else False
        train_samples = sample_dataset(train_dataset, num_examples)
        check_dataset_labels(
            train_samples,
            tokenizer,
            num_examples=num_examples,
            text_only=text_only,
        )

        LOG.info("Printing prompters...")
        for prompter in prompters:
            LOG.info(prompter)

    # Check if HuggingFace upload is requested
    if HAS_HF_UPLOAD and should_upload_to_hf(cfg):
        try:
            LOG.info("HuggingFace upload is enabled, validating configuration...")
            hf_config = cfg.hf_upload
            validate_hf_config(hf_config)
            
            # Get the dataset path where data was saved
            dataset_prepared_path = Path(cfg.dataset_prepared_path or "./data/prepared")
            split_name = getattr(cli_args, 'dataset_split', 'train') if cli_args else 'train'
            actual_dataset_path = dataset_prepared_path / "_".join([
                split_name,
                cfg.tokenizer_config.replace("/", "_"),
                f"{cfg.sequence_len}",
            ])
            
            # Load processing statistics if available
            stats = {
                "processed_samples": len(train_dataset) if train_dataset else 0,
                "original_samples": len(train_dataset) if train_dataset else 0,
                "success_rate": 1.0,
                "model": cfg.base_model,
                "sequence_length": cfg.sequence_len
            }
            stats_path = actual_dataset_path.parent / "stats.json"
            if stats_path.exists():
                import json
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
            
            LOG.info("Uploading dataset to HuggingFace Hub...")
            # Get token from CLI args, config, or environment
            token = None
            if cli_args and hasattr(cli_args, 'hf_token') and cli_args.hf_token:
                token = cli_args.hf_token
            elif hf_config.get("token"):
                token = hf_config["token"]
            
            dataset_url = upload_dataset_to_hf(
                dataset_path=actual_dataset_path,
                organization=hf_config["organization"],
                dataset_name=hf_config["dataset_name"],
                config=cfg,
                hf_config=hf_config,
                stats=stats,
                token=token
            )
            
            LOG.info(f"🎉 Dataset successfully uploaded to HuggingFace Hub: {dataset_url}")
            
        except HFUploadError as e:
            LOG.error(f"❌ HuggingFace upload failed: {e}")
            if cfg.get("hf_upload", {}).get("fail_on_upload_error", False):
                raise
        except Exception as e:
            LOG.error(f"❌ Unexpected error during HuggingFace upload: {e}")
            if cfg.get("hf_upload", {}).get("fail_on_upload_error", False):
                raise
    elif not HAS_HF_UPLOAD and cfg.get("hf_upload", {}).get("enabled", False):
        LOG.warning("HuggingFace upload requested but dependencies not available. Install with: pip install datasets huggingface_hub")

    return TrainDatasetMeta(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        total_num_steps=total_num_steps,
    )