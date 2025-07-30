"""Multimodal plugin for vision language preprocessing."""

import inspect
import math
import os
import re
import gc
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, Optional, TypedDict, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers.image_utils import get_image_size, is_valid_image, to_numpy_array
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from typing_extensions import override

from .utils.logging import get_logger

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from PIL import Image
    from PIL.Image import Image as ImageObject
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False

# Placeholder constants
IMAGE_PLACEHOLDER = "<image>"
VIDEO_PLACEHOLDER = "<video>"
AUDIO_PLACEHOLDER = "<audio>"
IGNORE_INDEX = -100

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
    from transformers.image_processing_utils import BaseImageProcessor

    class EncodedImage(TypedDict):
        path: Optional[str]
        bytes: Optional[bytes]

    ImageInput = Union[str, bytes, EncodedImage, BinaryIO, ImageObject]
    VideoInput = Union[str, BinaryIO, list[list[ImageInput]]]
    AudioInput = Union[str, BinaryIO, NDArray]

    class MMProcessor(ProcessorMixin):
        patch_size: int
        image_seq_length: int
        num_additional_image_tokens: int
        vision_feature_select_strategy: Literal["default", "full"]

LOG = get_logger(__name__)


def _check_video_is_nested_images(video: "VideoInput") -> bool:
    """Check if the video is nested images."""
    return isinstance(video, list) and all(isinstance(frame, (str, BinaryIO, dict)) for frame in video)


def _make_batched_images(images: list["ImageObject"], imglens: list[int]) -> list[list["ImageObject"]]:
    """Make nested list of images."""
    batch_images = []
    for imglen in imglens:
        batch_images.append(images[:imglen])
        images = images[imglen:]
    return batch_images


@dataclass
class MMPluginMixin:
    image_token: Optional[str]
    video_token: Optional[str]
    audio_token: Optional[str]
    expand_mm_tokens: bool = True

    def _validate_input(
        self,
        processor: Optional["MMProcessor"],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> None:
        """Validate if this model accepts the input modalities."""
        if not HAS_PIL and len(images) > 0:
            raise ImportError("PIL is required for image processing. Install with: pip install Pillow")
        
        if not HAS_AV and len(videos) > 0:
            raise ImportError("PyAV is required for video processing. Install with: pip install av")
        
        if not HAS_LIBROSA and len(audios) > 0:
            raise ImportError("librosa is required for audio processing. Install with: pip install librosa")

        image_processor = getattr(processor, "image_processor", None) if processor else None
        video_processor = getattr(processor, "video_processor", 
                                 getattr(processor, "image_processor", None)) if processor else None
        feature_extractor = getattr(processor, "feature_extractor", None) if processor else None

        if len(images) != 0 and self.image_token is None:
            raise ValueError(
                "This model does not support image input. Please check whether the correct template is used."
            )

        if len(videos) != 0 and self.video_token is None:
            raise ValueError(
                "This model does not support video input. Please check whether the correct template is used."
            )

        if len(audios) != 0 and self.audio_token is None:
            raise ValueError(
                "This model does not support audio input. Please check whether the correct template is used."
            )

        # Note: In data preprocessing, we may not have processors available
        # Only validate if we actually need to process media
        if self.image_token is not None and processor is None and len(images) > 0:
            LOG.debug("No processor found for images - using text tokens only")

        if self.video_token is not None and video_processor is None and len(videos) > 0:
            LOG.debug("No video processor found - using text tokens only")

        if self.audio_token is not None and feature_extractor is None and len(audios) > 0:
            LOG.debug("No audio feature extractor found - using text tokens only")

    def _validate_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ):
        """Validate if the number of images, videos and audios match the number of placeholders in messages."""
        num_image_tokens, num_video_tokens, num_audio_tokens = 0, 0, 0
        for message in messages:
            num_image_tokens += message["content"].count(IMAGE_PLACEHOLDER)
            num_video_tokens += message["content"].count(VIDEO_PLACEHOLDER)
            num_audio_tokens += message["content"].count(AUDIO_PLACEHOLDER)

        if len(images) != num_image_tokens:
            raise ValueError(
                f"The number of images does not match the number of {IMAGE_PLACEHOLDER} tokens in {messages}."
            )

        if len(videos) != num_video_tokens:
            raise ValueError(
                f"The number of videos does not match the number of {VIDEO_PLACEHOLDER} tokens in {messages}."
            )

        if len(audios) != num_audio_tokens:
            raise ValueError(
                f"The number of audios does not match the number of {AUDIO_PLACEHOLDER} tokens in {messages}."
            )

    def _preprocess_image(
        self, image: "ImageObject", image_max_pixels: int, image_min_pixels: int, **kwargs
    ) -> "ImageObject":
        """Pre-process a single image."""
        original_image = image
        needs_cleanup = False
        
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            if needs_cleanup and hasattr(image, 'close'):
                image.close()
            image = original_image.resize((width, height))
            needs_cleanup = True

        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            if needs_cleanup and hasattr(image, 'close'):
                image.close()
            image = (image if not needs_cleanup else original_image).resize((width, height))
            needs_cleanup = True

        if image.mode != "RGB":
            old_image = image
            image = image.convert("RGB")
            # Clean up the old image if it was created during processing
            if needs_cleanup and hasattr(old_image, 'close') and old_image != original_image:
                old_image.close()

        return image

    def _regularize_images(self, images: list["ImageInput"], **kwargs) -> dict[str, list["ImageObject"]]:
        """Regularize images to avoid error. Including reading and pre-processing."""
        if not HAS_PIL:
            raise ImportError("PIL is required for image processing. Install with: pip install Pillow")

        results = []
        for i, image in enumerate(images):
            try:
                if isinstance(image, (str, BinaryIO)):
                    image = Image.open(image)
                elif isinstance(image, bytes):
                    image = Image.open(BytesIO(image))
                elif isinstance(image, dict):
                    if image["bytes"] is not None:
                        image = Image.open(BytesIO(image["bytes"]))
                    else:
                        image = Image.open(image["path"])

                if not isinstance(image, Image.Image):
                    raise ValueError(f"Expect input is a list of images, but got {type(image)}.")

                processed_image = self._preprocess_image(image, **kwargs)
                results.append(processed_image)
                
                # Periodic garbage collection for large image batches
                if (i + 1) % 50 == 0:
                    gc.collect()
                    
            except Exception as e:
                LOG.warning(f"Error processing image {i}: {e}")
                # Clean up any partially loaded image
                if 'image' in locals() and hasattr(image, 'close'):
                    image.close()
                raise

        return {"images": results}

    def _get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: "MMProcessor",
        imglens: Optional[list[int]] = None,
    ) -> dict[str, "torch.Tensor"]:
        """Process visual inputs."""
        mm_inputs = {}
        if len(images) != 0:
            image_processor = getattr(processor, "image_processor", None)
            images = self._regularize_images(
                images,
                image_max_pixels=getattr(processor, "image_max_pixels", 768 * 768),
                image_min_pixels=getattr(processor, "image_min_pixels", 32 * 32),
            )["images"]
            if imglens is not None:  # if imglens are provided, make batched images
                images = _make_batched_images(images, imglens)

            mm_inputs.update(image_processor(images, return_tensors="pt"))

        return mm_inputs


@dataclass
class BasePlugin(MMPluginMixin):
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        """Pre-process input messages before tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return messages

    def process_token_ids(
        self,
        input_ids: list[int],
        labels: Optional[list[int]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["MMProcessor"],
    ) -> tuple[list[int], Optional[list[int]]]:
        """Pre-process token ids after tokenization for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return input_ids, labels

    def get_mm_inputs(
        self,
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        imglens: list[int],
        vidlens: list[int],
        audlens: list[int],
        batch_ids: list[list[int]],
        processor: Optional["MMProcessor"],
    ) -> dict[str, Union[list[int], "torch.Tensor"]]:
        """Build batched multimodal inputs for VLMs."""
        self._validate_input(processor, images, videos, audios)
        return self._get_mm_inputs(images, videos, audios, processor)


@dataclass
class LlavaPlugin(BasePlugin):
    @override
    def process_messages(
        self,
        messages: list[dict[str, str]],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
        processor: Optional["MMProcessor"],
    ) -> list[dict[str, str]]:
        self._validate_input(processor, images, videos, audios)
        self._validate_messages(messages, images, videos, audios)
        messages = deepcopy(messages)
        if self.expand_mm_tokens:
            mm_inputs = self._get_mm_inputs(images, videos, audios, processor)
            if "pixel_values" in mm_inputs:
                height, width = get_image_size(to_numpy_array(mm_inputs["pixel_values"][0]))
                image_seqlen = (height // processor.patch_size) * (
                    width // processor.patch_size
                ) + processor.num_additional_image_tokens
                if processor.vision_feature_select_strategy == "default":
                    image_seqlen -= 1
        else:
            image_seqlen = 1

        for message in messages:
            content = message["content"]
            while IMAGE_PLACEHOLDER in content:
                content = content.replace(IMAGE_PLACEHOLDER, "{{image}}" * image_seqlen, 1)

            message["content"] = content.replace("{{image}}", self.image_token)

        return messages


# Available plugins
PLUGINS = {
    "base": BasePlugin,
    "llava": LlavaPlugin,
}


def register_mm_plugin(name: str, plugin_class: type["BasePlugin"]) -> None:
    """Register a multimodal plugin."""
    if name in PLUGINS:
        raise ValueError(f"Multimodal plugin {name} already exists.")

    PLUGINS[name] = plugin_class


def get_mm_plugin(
    name: str,
    image_token: Optional[str] = None,
    video_token: Optional[str] = None,
    audio_token: Optional[str] = None,
) -> "BasePlugin":
    """Get plugin for multimodal inputs."""
    if name not in PLUGINS:
        raise ValueError(f"Multimodal plugin `{name}` not found.")

    return PLUGINS[name](image_token, video_token, audio_token)