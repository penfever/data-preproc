"""Vision Language prompt strategy for multimodal datasets."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    from ..mm_plugin import get_mm_plugin, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER, AUDIO_PLACEHOLDER
    from PIL import Image
    HAS_MM_PLUGIN = True
    HAS_PIL = True
except ImportError:
    HAS_MM_PLUGIN = False
    HAS_PIL = False

from ..utils.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from ..utils.dict import DictDefault

LOG = get_logger(__name__)


def verify_image(image: "Image.Image") -> bool:
    """Verify that an image is valid and not corrupted."""
    if not HAS_PIL or image is None:
        return False
    try:
        if image.size[0] == 0 or image.size[1] == 0:
            return False
        image.load()
        return True
    except Exception as e:
        LOG.warning(f"Image verification failed: {e}")
        return False


def process_image(image: "Image.Image", config: Dict[str, Any]) -> Optional["Image.Image"]:
    """Process image according to configuration."""
    if not HAS_PIL or image is None:
        return None
    
    try:
        # Verify image if requested
        if config.get("verify_images", True) and not verify_image(image):
            return None
        
        # Convert to RGB if requested
        if config.get("convert_to_rgb", True) and image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize if max_size specified
        max_size = config.get("max_size")
        if max_size and isinstance(max_size, (list, tuple)) and len(max_size) == 2:
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Check minimum size
        min_size = config.get("min_size")
        if min_size and isinstance(min_size, (list, tuple)) and len(min_size) == 2:
            if image.size[0] < min_size[0] or image.size[1] < min_size[1]:
                return None
        
        return image
    except Exception as e:
        LOG.warning(f"Image processing failed: {e}")
        return None


def load(tokenizer: "PreTrainedTokenizer", cfg: "DictDefault", ds_cfg: Optional[Dict[str, Any]] = None):
    """Load vision language prompt strategy."""
    
    if not HAS_MM_PLUGIN:
        raise ImportError("MM plugin not available. Make sure datasets and other dependencies are installed.")
    
    # Ensure ds_cfg is available for the inner function
    ds_cfg = ds_cfg or {}
    
    def process_vl_example(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Process vision language examples."""
        results = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "images": [],
            "videos": [], 
            "audios": [],
            "metadata": []
        }
        
        # Get processor if available
        processor = getattr(cfg, 'processor', None)
        
        # Get MM plugin configuration
        mm_plugin_name = ds_cfg.get("mm_plugin", "base")
        image_token = ds_cfg.get("image_token", "<image>")
        video_token = ds_cfg.get("video_token", "<video>")
        audio_token = ds_cfg.get("audio_token", "<audio>")
        
        # Get VL processing configuration
        vl_config = cfg.get("vl_config", {})
        image_config = vl_config.get("image_processing", {})
        video_config = vl_config.get("video_processing", {})
        audio_config = vl_config.get("audio_processing", {})
        
        # Apply dataset-specific overrides
        if ds_cfg.get("filter_corrupted_images"):
            image_config = {**image_config, "verify_images": True}
        if ds_cfg.get("max_image_size"):
            image_config = {**image_config, "max_size": ds_cfg["max_image_size"]}
        
        mm_plugin = get_mm_plugin(
            mm_plugin_name,
            image_token=image_token,
            video_token=video_token, 
            audio_token=audio_token
        )
        
        # Determine dataset format and get examples
        if "messages" in examples:
            # Messages format
            num_examples = len(examples["messages"])
            def get_example_data(i):
                messages = examples["messages"][i]
                raw_images = examples.get("images", [None] * num_examples)[i] or []
                raw_videos = examples.get("videos", [None] * num_examples)[i] or []
                raw_audios = examples.get("audios", [None] * num_examples)[i] or []
                return messages, raw_images, raw_videos, raw_audios
        else:
            # Q&A format - convert on the fly
            first_key = next(iter(examples.keys()))
            num_examples = len(examples[first_key])
            def get_example_data(i):
                # Convert Q&A to messages format
                question = examples.get("problem", [None] * num_examples)[i] or \
                          examples.get("question", [None] * num_examples)[i] or \
                          examples.get("original_question", [None] * num_examples)[i]
                answer = examples.get("solution", [None] * num_examples)[i] or \
                        examples.get("answer", [None] * num_examples)[i] or \
                        examples.get("original_answer", [None] * num_examples)[i]
                
                if not question or not answer:
                    raise ValueError(f"Missing question or answer in example {i}")
                
                messages = [
                    {"role": "user", "content": str(question)},
                    {"role": "assistant", "content": str(answer)}
                ]
                
                # Get images
                raw_images = []
                if "image" in examples and examples["image"][i] is not None:
                    raw_images = [examples["image"][i]]
                elif "images" in examples and examples["images"][i] is not None:
                    raw_images = examples["images"][i] if isinstance(examples["images"][i], list) else [examples["images"][i]]
                
                raw_videos = examples.get("videos", [[] for _ in range(num_examples)])[i] or []
                raw_audios = examples.get("audios", [[] for _ in range(num_examples)])[i] or []
                
                return messages, raw_images, raw_videos, raw_audios
        
        for i in range(num_examples):
            try:
                messages, raw_images, raw_videos, raw_audios = get_example_data(i)
                
                # Process media according to configuration
                processed_images = []
                for img in raw_images:
                    processed_img = process_image(img, image_config)
                    if processed_img is not None:
                        processed_images.append(processed_img)
                
                # Skip if no valid images after processing (if images were required)
                if raw_images and not processed_images and image_config.get("verify_images", True):
                    LOG.debug(f"Skipping sample {i}: no valid images after processing")
                    continue
                
                # TODO: Add video and audio processing
                processed_videos = raw_videos  # Pass through for now
                processed_audios = raw_audios  # Pass through for now
                
                # Process messages through MM plugin
                processed_messages = mm_plugin.process_messages(
                    messages, processed_images, processed_videos, processed_audios, processor
                )
                
                # Convert messages to conversation text
                conversation_text = ""
                for message in processed_messages:
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    
                    if role == "user":
                        conversation_text += f"Human: {content}\n\n"
                    elif role == "assistant":
                        conversation_text += f"Assistant: {content}\n\n"
                    elif role == "system":
                        conversation_text = f"System: {content}\n\n" + conversation_text
                
                # Tokenize the conversation
                tokenized = tokenizer(
                    conversation_text.strip(),
                    truncation=True,
                    max_length=cfg.get("sequence_len", 2048),
                    padding=False,
                    return_tensors=None
                )
                
                # Track original length for metadata
                original_tokenized = tokenizer(
                    conversation_text.strip(),
                    truncation=False,
                    padding=False,
                    return_tensors=None
                )
                original_length = len(original_tokenized["input_ids"])
                was_truncated = original_length > cfg.get("sequence_len", 2048)
                
                # Process token IDs through MM plugin
                input_ids, labels = mm_plugin.process_token_ids(
                    tokenized["input_ids"],
                    tokenized["input_ids"].copy(),  # Use input_ids as labels for now
                    processed_images,
                    processed_videos, 
                    processed_audios,
                    tokenizer,
                    processor
                )
                
                # Create metadata
                metadata = {
                    "original_length": original_length,
                    "truncated": was_truncated,
                    "num_images": len(processed_images),
                    "num_videos": len(processed_videos),
                    "num_audios": len(processed_audios),
                    "mm_plugin": mm_plugin_name
                }
                
                results["input_ids"].append(input_ids)
                results["attention_mask"].append([1] * len(input_ids))
                results["labels"].append(labels)
                results["images"].append(processed_images)
                results["videos"].append(processed_videos)
                results["audios"].append(processed_audios)
                results["metadata"].append(metadata)
                
            except Exception as e:
                LOG.warning(f"Error processing VL example {i}: {e}")
                # Instead of skipping, add empty placeholders to maintain array alignment
                results["input_ids"].append([])
                results["attention_mask"].append([])
                results["labels"].append([])
                results["images"].append([])
                results["videos"].append([])
                results["audios"].append([])
                results["metadata"].append({"error": str(e)})
        
        return results
    
    return process_vl_example