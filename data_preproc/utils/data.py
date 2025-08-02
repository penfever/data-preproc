"""Data preparation utilities."""

import logging
import os
import gc
from typing import Optional, Tuple, List, Any, Dict, Union
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets, get_dataset_config_names

from data_preproc.utils.dict import DictDefault
from data_preproc.prompt_strategies import load
from data_preproc.utils.error_handling import ErrorHandlingConfig, ErrorHandlingWrapper, ErrorHandlingMode, safe_dataset_iterator, safe_process_example

LOG = logging.getLogger(__name__)


def load_dataset_with_subset(
    dataset_path: str,
    subset: Union[str, List[str]],
    split: str = "train",
    streaming: bool = False,
    data_files: Optional[Any] = None,
    **kwargs
) -> Dataset:
    """
    Load a dataset with subset support.
    
    Args:
        dataset_path: Path to the dataset
        subset: Single subset name, list of subset names, or "_ALL" for all subsets
        split: Dataset split to load
        streaming: Whether to use streaming mode
        data_files: Optional data files specification
        **kwargs: Additional arguments passed to load_dataset
    
    Returns:
        Dataset with subset(s) loaded and concatenated if multiple
    """
    # Handle "_ALL" special case
    if subset == "_ALL":
        try:
            available_subsets = get_dataset_config_names(dataset_path)
            if not available_subsets:
                LOG.warning(f"No subsets found for dataset {dataset_path}, loading without subset")
                if data_files:
                    return load_dataset(dataset_path, data_files=data_files, split=split, streaming=streaming, **kwargs)
                else:
                    return load_dataset(dataset_path, split=split, streaming=streaming, **kwargs)
            subset = available_subsets
            LOG.info(f"Loading all {len(available_subsets)} subsets for {dataset_path}: {available_subsets}")
        except Exception as e:
            LOG.warning(f"Failed to discover subsets for {dataset_path}: {e}. Loading without subset.")
            if data_files:
                return load_dataset(dataset_path, data_files=data_files, split=split, streaming=streaming, **kwargs)
            else:
                return load_dataset(dataset_path, split=split, streaming=streaming, **kwargs)
    
    # Convert single subset to list for uniform processing
    if isinstance(subset, str):
        subset = [subset]
    
    # Load each subset
    datasets = []
    for subset_name in subset:
        try:
            LOG.info(f"Loading subset '{subset_name}' from {dataset_path}")
            if data_files:
                ds = load_dataset(dataset_path, subset_name, data_files=data_files, split=split, streaming=streaming, **kwargs)
            else:
                ds = load_dataset(dataset_path, subset_name, split=split, streaming=streaming, **kwargs)
            datasets.append(ds)
        except Exception as e:
            LOG.warning(f"Failed to load subset '{subset_name}' from {dataset_path}: {e}")
            # Continue with other subsets if one fails
            continue
    
    # Handle case where no subsets were successfully loaded
    if not datasets:
        LOG.error(f"No subsets could be loaded from {dataset_path}. Trying without subset.")
        if data_files:
            return load_dataset(dataset_path, data_files=data_files, split=split, streaming=streaming, **kwargs)
        else:
            return load_dataset(dataset_path, split=split, streaming=streaming, **kwargs)
    
    # If only one dataset, return it directly
    if len(datasets) == 1:
        return datasets[0]
    
    # Concatenate multiple datasets
    LOG.info(f"Concatenating {len(datasets)} subsets from {dataset_path}")
    return concatenate_datasets(datasets)


def prepare_dataset(
    cfg: DictDefault,
    tokenizer,
    processor=None,
    preprocess_iterable: Optional[bool] = None,
    limit: Optional[int] = None,
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[int], List[Any]]:
    """
    Prepare training and evaluation datasets based on configuration.
        
    Args:
        cfg: Configuration dictionary
        tokenizer: Tokenizer instance
        processor: Optional processor instance
        preprocess_iterable: Whether to use iterable datasets
        limit: Optional global limit on number of samples to process
        
    Returns:
        Tuple of (train_dataset, eval_dataset, total_num_steps, prompters)
    """
    prompters = []
    train_dataset = None
    eval_dataset = None
    total_num_steps = None
    
    # Set up error handling
    error_handling_config = ErrorHandlingConfig.from_dict(cfg.get("error_handling", {}))
    error_handler = ErrorHandlingWrapper(error_handling_config)
    LOG.info(f"Error handling mode: {error_handling_config.mode.value}")
    
    # Validate global limit parameter
    if limit is not None:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"Global limit must be a positive integer, got: {limit}")
        LOG.info(f"Global limit set to: {limit}")
    
    # Load datasets based on configuration
    if cfg.datasets:
        datasets = []
        
        for dataset_config in cfg.datasets:
            # Basic dataset loading
            dataset_path = dataset_config.get("path")
            dataset_type = dataset_config.get("type", "custom")
            split = dataset_config.get("split", "train")
            subset = dataset_config.get("subset")
            
            # Check for limit parameter (per-dataset overrides global)
            dataset_limit = dataset_config.get("limit", limit)
            use_streaming = dataset_limit is not None
            
            # Force streaming mode if error handling is set to log_and_continue or raise_after_k
            # This allows individual problematic samples to be skipped during iteration
            if error_handling_config.mode in [ErrorHandlingMode.LOG_AND_CONTINUE, ErrorHandlingMode.RAISE_AFTER_K]:
                if not use_streaming:
                    LOG.warning(f"Forcing streaming mode for dataset {dataset_path} due to error handling mode: {error_handling_config.mode.value}")
                use_streaming = True
            
            # Validate limit parameter
            if dataset_limit is not None:
                if not isinstance(dataset_limit, int) or dataset_limit <= 0:
                    raise ValueError(f"Dataset limit must be a positive integer, got: {dataset_limit}")
            
            LOG.info(f"Loading dataset: {dataset_path}")
            if dataset_limit:
                LOG.info(f"Using streaming mode with limit: {dataset_limit}")
            
            try:
                # Load the dataset
                if dataset_path:
                    try:
                        # Check if specific data files are specified
                        data_files = dataset_config.get("data_files")
                        
                        # Handle subset loading
                        if subset:
                            ds = load_dataset_with_subset(
                                dataset_path,
                                subset,
                                split=split,
                                streaming=use_streaming,
                                data_files=data_files
                            )
                        else:
                            # Standard loading without subset
                            if data_files:
                                ds = load_dataset(dataset_path, data_files=data_files, split=split, streaming=use_streaming)
                            else:
                                ds = load_dataset(dataset_path, split=split, streaming=use_streaming)
                    except Exception as load_error:
                        # If initial dataset loading fails, try to handle it gracefully
                        should_continue = error_handler.handle_error(load_error, f"initial dataset loading: {dataset_path}")
                        if not should_continue:
                            raise
                        else:
                            LOG.warning(f"Skipping dataset {dataset_path} due to loading error")
                            continue
                    
                    # Apply limit if specified or convert streaming dataset
                    if use_streaming:
                        if dataset_limit:
                            LOG.info(f"Applying limit of {dataset_limit} samples")
                            ds = ds.take(dataset_limit)
                        # Convert back to regular dataset for processor compatibility
                        LOG.info("Converting streaming dataset to regular dataset with error handling...")
                        ds = Dataset.from_list(list(safe_dataset_iterator(ds, error_handler, "streaming dataset conversion")))
                    elif dataset_limit:
                        # For regular datasets, use select()
                        LOG.info(f"Applying limit of {dataset_limit} samples")
                        ds = ds.select(range(min(dataset_limit, len(ds))))
                    
                    # Apply processors if configured
                    if "processors" in dataset_config:
                        from data_preproc.processors import create_processor, get_processor
                        
                        for proc_index, proc_config in enumerate(dataset_config["processors"]):
                            # Support both old and new processor configuration formats
                            try:
                                # Try new modular format first
                                proc = create_processor(proc_config)
                                proc_type = proc_config.get("type", "unknown")
                                proc_name = proc_config.get("name", proc_type)
                                proc_identifier = f"{proc_name} ({proc_type})" if proc_name != proc_type else proc_type
                            except (ValueError, KeyError):
                                # Fall back to old format for backward compatibility
                                proc_type = proc_config.get("type")
                                if not proc_type:
                                    LOG.error(f"Processor configuration missing 'type' field: {proc_config}")
                                    continue
                                proc = get_processor(proc_type, proc_config)
                                proc_identifier = proc_type
                            
                            initial_count = len(ds)
                            LOG.info(f"")
                            LOG.info(f"🔄 Processing Stage {proc_index + 1}/{len(dataset_config.get('processors', []))}: {proc_identifier}")
                            LOG.info(f"   Input: {initial_count} examples")
                            
                            # Special handling for HF filter processor
                            if hasattr(proc, 'apply_to_dataset'):
                                proc.tokenizer = tokenizer
                                ds = proc.apply_to_dataset(ds)
                                final_count = len(ds)
                                filtered_count = initial_count - final_count
                                filter_rate = (filtered_count / initial_count * 100) if initial_count > 0 else 0
                                LOG.info(f"   Output: {final_count} examples ({filtered_count} filtered, {filter_rate:.1f}% reduction)")
                            # Special handling for sample_packer processor
                            elif hasattr(proc, 'process_dataset'):
                                ds = proc.process_dataset(ds)
                                final_count = len(ds)
                                LOG.info(f"   Output: {final_count} packed examples from {initial_count} original examples")
                            else:
                                # Apply standard processor with streaming/batched processing
                                ds = _apply_processor_streaming(ds, proc, proc_identifier, error_handler)
                                if ds is not None:
                                    final_count = len(ds)
                                    filtered_count = initial_count - final_count
                                    filter_rate = (filtered_count / initial_count * 100) if initial_count > 0 else 0
                                    LOG.info(f"   Output: {final_count} examples ({filtered_count} filtered, {filter_rate:.1f}% reduction)")
                                else:
                                    LOG.error(f"❌ All {initial_count} examples filtered out by {proc_identifier} processor for dataset {dataset_path}")
                                    
                                    # Provide helpful suggestions based on processor type
                                    if proc_type == "hf_filter":
                                        LOG.error("Suggestion: Check min_tokens/max_tokens settings. Use --debug to see token counts per example.")
                                    elif proc_type == "advanced_mapping":
                                        LOG.error("Suggestion: Check mapping configuration and required fields. Use --debug to see transformation details.")
                                    elif proc_type == "image_count_filter":
                                        LOG.error("Suggestion: Check min_images/max_images settings and image field names.")
                                    
                                    raise ValueError(f"All examples filtered out by {proc_identifier} processor. Check processor configuration and input data.")
                    
                    # Log processor pipeline summary
                    if "processors" in dataset_config:
                        final_dataset_count = len(ds)
                        LOG.info(f"")
                        LOG.info(f"✅ Processor Pipeline Complete for {dataset_path}")
                        LOG.info(f"   Final dataset size: {final_dataset_count} examples")
                        LOG.info(f"   Processors applied: {len(dataset_config['processors'])}")
                        LOG.info(f"")
                    
                    # Handle prompt strategies if specified
                    if dataset_type and dataset_type != "custom":
                        strategy = load(dataset_type, tokenizer, cfg, dataset_config, processor=processor)
                        if strategy:
                            # Apply the strategy to transform the dataset
                            if hasattr(strategy, 'wrap_dataset'):
                                ds = strategy.wrap_dataset(ds)
                            elif hasattr(strategy, 'process'):
                                # Process each example with the strategy
                                processed_examples = []
                                for i, example in enumerate(safe_dataset_iterator(ds, error_handler, "strategy processing")):
                                    processed = safe_process_example(example, strategy.process, error_handler, "strategy processing", i)
                                    if processed:
                                        processed_examples.append(processed)
                                if processed_examples:
                                    ds = Dataset.from_list(processed_examples)
                    
                    if ds is not None:
                        datasets.append(ds)
                    else:
                        LOG.warning(f"Dataset {dataset_path} resulted in None after processing")
                    
            except Exception as e:
                should_continue = error_handler.handle_error(e, f"dataset loading: {dataset_path}")
                if not should_continue:
                    raise
        
        # Concatenate multiple datasets if needed
        if len(datasets) > 1:
            LOG.info("Merging multiple datasets...")
            train_dataset = concatenate_datasets(datasets)
        elif len(datasets) == 1:
            train_dataset = datasets[0]
        else:
            LOG.warning("No datasets loaded successfully")
            
        # Handle validation split
        val_set_size = cfg.get("val_set_size", 0)
        if train_dataset and val_set_size > 0:
            if val_set_size < 1:
                # Percentage split
                split_datasets = train_dataset.train_test_split(test_size=val_set_size, seed=cfg.get("seed", 42))
            else:
                # Absolute number split
                split_datasets = train_dataset.train_test_split(test_size=int(val_set_size), seed=cfg.get("seed", 42))
            
            train_dataset = split_datasets["train"]
            eval_dataset = split_datasets["test"]
            
        # Calculate total steps if needed
        if train_dataset and cfg.get("num_epochs"):
            batch_size = cfg.get("micro_batch_size", 1) * cfg.get("gradient_accumulation_steps", 1)
            total_num_steps = (len(train_dataset) // batch_size) * cfg.get("num_epochs", 1)
        
        # Save dataset to disk if we have a prepared path
        if train_dataset and cfg.get("dataset_prepared_path"):
            prepared_ds_path = Path(cfg.dataset_prepared_path)
            
            # Create a specific path based on dataset configuration
            tokenizer_name = cfg.get("tokenizer_config", "unknown").replace("/", "_")
            sequence_len = cfg.get("sequence_len", "unknown")
            split_name = "train"
            
            # Build the full path
            full_path = prepared_ds_path / f"{split_name}_{tokenizer_name}_{sequence_len}"
            
            LOG.info(f"Saving merged prepared dataset to disk... {full_path}")
            os.makedirs(full_path, exist_ok=True)
            train_dataset.save_to_disk(str(full_path))
            
            # Also save eval dataset if it exists
            if eval_dataset:
                eval_path = prepared_ds_path / f"eval_{tokenizer_name}_{sequence_len}"
                LOG.info(f"Saving eval dataset to disk... {eval_path}")
                os.makedirs(eval_path, exist_ok=True)
                eval_dataset.save_to_disk(str(eval_path))
    
    return train_dataset, eval_dataset, total_num_steps, prompters


def _apply_processor_streaming(dataset, processor, proc_type, error_handler, batch_size=1000):
    """Apply processor with memory-efficient streaming and batching with conditional support."""
    processed_batches = []
    batch = []
    filtered_count = 0
    skipped_count = 0
    filter_reasons = {}
    
    LOG.info(f"Processing {len(dataset)} examples with {proc_type} processor using batched streaming...")
    
    for i, example in enumerate(safe_dataset_iterator(dataset, error_handler, f"{proc_type} processor iteration")):
        # Check if processor should process this example (supports conditional execution)
        should_process = True
        if hasattr(processor, 'should_process'):
            try:
                should_process = processor.should_process(example)
            except Exception as e:
                LOG.warning(f"Error checking processor condition for example {i}: {e}")
                should_process = True  # Default to processing if condition check fails
        
        if should_process:
            # Process the example
            processed = safe_process_example(
                example, 
                processor.process_example, 
                error_handler, 
                f"{proc_type} processor", 
                i
            )
            
            if processed is not None:
                batch.append(processed)
            else:
                filtered_count += 1
                # Track filter reason if available
                reason = getattr(processor, '_last_filter_reason', 'processing failed')
                filter_reasons[reason] = filter_reasons.get(reason, 0) + 1
        else:
            # Skip processing but keep the original example
            batch.append(example)
            skipped_count += 1
            filter_reasons['condition not met'] = filter_reasons.get('condition not met', 0) + 1
        
        # Process batch when it reaches target size or at end
        if len(batch) >= batch_size or i == len(dataset) - 1:
            if batch:
                # Create dataset from batch and add to list
                batch_ds = Dataset.from_list(batch)
                processed_batches.append(batch_ds)
                LOG.debug(f"Processed batch {len(processed_batches)}: {len(batch)} examples")
                
                # Clear batch and force garbage collection
                batch.clear()
                del batch_ds
                gc.collect()
    
    # Combine all batches into final dataset
    if processed_batches:
        LOG.info(f"Combining {len(processed_batches)} batches into final dataset...")
        final_dataset = concatenate_datasets(processed_batches)
        
        # Clean up batch datasets
        for batch_ds in processed_batches:
            del batch_ds
        processed_batches.clear()
        gc.collect()
        
        processing_summary = f"{len(final_dataset)} examples kept"
        if filtered_count > 0:
            processing_summary += f", {filtered_count} filtered"
        if skipped_count > 0:
            processing_summary += f", {skipped_count} skipped (conditions not met)"
        LOG.info(f"Streaming processing complete: {processing_summary}")
        
        if error_handler.error_count > 0:
            LOG.info(f"Errors encountered during processing: {error_handler.error_count}")
            error_summary = error_handler.get_error_summary()
            if error_summary["error_types"]:
                LOG.info(f"Error types: {error_summary['error_types']}")
        return final_dataset
    else:
        LOG.warning(f"No examples passed {proc_type} processor")
        if filter_reasons:
            reason_summary = "; ".join([f"{reason}: {count}" for reason, count in filter_reasons.items()])
            LOG.warning(f"Filter reasons: {reason_summary}")
        return None