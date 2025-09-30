"""CLI to run preprocessing of a dataset."""

from pathlib import Path
import sys
from typing import List
from typing import Union

import fire
import transformers
from dotenv import load_dotenv

from data_preproc.cli.args import PreprocessCliArgs
from data_preproc.cli.config import load_cfg
from data_preproc.core.datasets import load_datasets
from data_preproc.utils.dict import DictDefault
from data_preproc.utils.logging import get_logger

LOG = get_logger(__name__)

DEFAULT_DATASET_PREPARED_PATH = "data/prepared"


def do_preprocess(cfg: DictDefault, cli_args: PreprocessCliArgs) -> None:
    """
    Preprocesses dataset specified in config.

    Args:
        cfg: Dictionary mapping config keys to values.
        cli_args: Preprocessing-specific CLI arguments.
    """
    import logging
    
    # Configure logging format to be cleaner for processor output
    # Default to WARNING to suppress most logs, but we'll selectively enable INFO for our processors
    logging.basicConfig(
        level=logging.WARNING if not cli_args.debug else logging.DEBUG,
        format='%(message)s',  # Simplified format for cleaner output
        force=True
    )
    
    # Suppress PIL debug logging
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('PIL.Image').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
    
    # Ensure processor loggers are at INFO level minimum
    processor_loggers = [
        'data_preproc.processors.image_count_filter',
        'data_preproc.processors.hf_filter', 
        'data_preproc.processors.deduplicator',
        'data_preproc.processors.image_transform',
        'data_preproc.utils.data'
    ]
    
    # Always show INFO level for our processor loggers
    for logger_name in processor_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        # Add a handler with INFO level to ensure the messages are shown
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
            logger.propagate = False
    
    if cli_args.debug:
        LOG.info("Debug logging enabled")

    if not cfg.dataset_prepared_path:
        LOG.warning(
            f"preprocess CLI called without dataset_prepared_path set, "
            f"using default path: {DEFAULT_DATASET_PREPARED_PATH}"
        )
        cfg.dataset_prepared_path = DEFAULT_DATASET_PREPARED_PATH

    # Load and preprocess datasets
    load_datasets(cfg=cfg, cli_args=cli_args)

    LOG.info(
        f"Success! Preprocessed data path: `dataset_prepared_path: {cfg.dataset_prepared_path}`"
    )


def do_cli(
    config: Union[Path, str] = Path("config.yaml"),
    **kwargs,
) -> None:
    """
    Parses config, CLI args, and calls `do_preprocess`.

    Args:
        config: Path to config YAML file.
        kwargs: Additional keyword arguments to override config file values.
    """
    parsed_cfg = load_cfg(config, **kwargs)
    parsed_cfg.is_preprocess = True
    parser = transformers.HfArgumentParser(PreprocessCliArgs)
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    do_preprocess(parsed_cfg, parsed_cli_args)


def main():
    """Main entry point."""
    load_dotenv()
    # Normalize short flag aliases before Fire parses arguments
    # Map `-c value` and `-c=value` to `--config value`
    argv = sys.argv[1:]
    normalized: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "-c":
            normalized.append("--config")
            # If a value follows, attach it; Fire will handle missing value errors
            if i + 1 < len(argv):
                normalized.append(argv[i + 1])
                i += 2
            else:
                i += 1
            continue
        if arg.startswith("-c="):
            normalized.append("--config")
            normalized.append(arg.split("=", 1)[1])
            i += 1
            continue
        normalized.append(arg)
        i += 1

    sys.argv = [sys.argv[0], *normalized]

    fire.Fire(do_cli)


if __name__ == "__main__":
    main()
