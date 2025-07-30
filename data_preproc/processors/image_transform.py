"""Image transformation processor with torchvision-style transforms."""

import logging
import math
from typing import Dict, Any, List, Optional, Union, Tuple
from copy import deepcopy

from . import DatasetProcessor, register_processor

LOG = logging.getLogger(__name__)

# PIL imports
try:
    from PIL import Image, ImageEnhance, ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    LOG.warning("PIL not available for image processing")

# Torchvision imports (optional)
try:
    import torch
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as F
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    LOG.warning("Torchvision not available, using PIL-only transforms")

# Numpy import for tensor operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    LOG.warning("Numpy not available")


class ImageTransformProcessor(DatasetProcessor):
    """Processor that applies torchvision-style image transformations.
    
    Supports:
    - Resize, crop, and resized crop operations
    - Color space transformations (grayscale, ColorJitter)
    - Normalization and format conversions
    - Multiple image fields
    - Both PIL and tensor inputs/outputs
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        if not HAS_PIL:
            raise ImportError("PIL is required for image transformations")
        
        # Support both complex transforms and simple resize parameters
        self.transforms = config.get("transforms", [])
        self.image_fields = config.get("image_fields", ["image"])
        self.output_format = config.get("output_format", "pil")  # "pil" or "tensor"
        self.skip_on_error = config.get("skip_on_error", True)
        
        # Simple resize parameters (alternative to transforms)
        self.max_size = config.get("max_size")
        self.resize_mode = config.get("resize_mode", "keep_aspect_ratio")
        
        # If max_size is provided, create a simple resize transform
        if self.max_size and not self.transforms:
            self.transforms = [{
                "type": "resize",
                "size": self.max_size,
                "mode": self.resize_mode
            }]
        
        # Validate transforms
        self._validate_transforms()
        
        LOG.info(f"Initialized ImageTransformProcessor with {len(self.transforms)} transforms")
        LOG.info(f"Image fields: {self.image_fields}")
        LOG.info(f"Output format: {self.output_format}")
        if self.max_size:
            LOG.info(f"Max size: {self.max_size}, resize mode: {self.resize_mode}")
    
    def _validate_transforms(self):
        """Validate transform configurations."""
        supported_transforms = {
            "resize", "center_crop", "random_crop", "resized_crop",
            "grayscale", "color_jitter", "normalize", "to_tensor", "to_pil"
        }
        
        for i, transform in enumerate(self.transforms):
            transform_type = transform.get("type")
            if not transform_type:
                raise ValueError(f"Transform {i} missing 'type' parameter")
            
            if transform_type not in supported_transforms:
                raise ValueError(f"Unsupported transform type: {transform_type}")
            
            # Validate specific transform parameters
            self._validate_transform_params(transform_type, transform)
    
    def _validate_transform_params(self, transform_type: str, params: Dict[str, Any]):
        """Validate parameters for specific transform types."""
        if transform_type == "resize":
            size = params.get("size")
            if not size:
                raise ValueError("Resize transform requires 'size' parameter")
            
        elif transform_type in ["center_crop", "random_crop"]:
            size = params.get("size")
            if not size:
                raise ValueError(f"{transform_type} transform requires 'size' parameter")
            
        elif transform_type == "resized_crop":
            required_params = ["top", "left", "height", "width", "size"]
            for param in required_params:
                if param not in params:
                    raise ValueError(f"resized_crop requires '{param}' parameter")
        
        elif transform_type == "normalize":
            mean = params.get("mean")
            std = params.get("std")
            if not mean or not std:
                raise ValueError("Normalize transform requires 'mean' and 'std' parameters")
    
    def apply_to_dataset(self, dataset):
        """Apply image transformations directly to the dataset with detailed logging."""
        initial_count = len(dataset)
        
        # Track transformation statistics
        transform_stats = {
            'success': 0,
            'skipped_missing_field': 0,
            'skipped_none_image': 0,
            'error_count': 0,
            'total_processed': 0
        }
        
        def process_function(example):
            """Process function that gets applied to each example."""
            transform_stats['total_processed'] += 1
            result = example.copy()
            
            for field in self.image_fields:
                if field not in result:
                    transform_stats['skipped_missing_field'] += 1
                    continue
                
                image = result[field]
                if image is None:
                    transform_stats['skipped_none_image'] += 1
                    continue
                
                try:
                    # Apply transformations
                    original_size = getattr(image, 'size', None)
                    transformed_image = self._apply_transforms(image)
                    new_size = getattr(transformed_image, 'size', None)
                    
                    result[field] = transformed_image
                    transform_stats['success'] += 1
                    
                    # Log size changes if we have size info
                    if original_size and new_size and original_size != new_size:
                        LOG.debug(f"Resized image from {original_size} to {new_size}")
                    
                except Exception as e:
                    transform_stats['error_count'] += 1
                    # Provide detailed error information for debugging
                    image_type = type(example[field]).__name__
                    if isinstance(example[field], dict):
                        dict_keys = list(example[field].keys())
                        LOG.warning(f"Error transforming image in field '{field}': {e}")
                        LOG.warning(f"Image is dict with keys: {dict_keys}")
                        # Log a sample of dict values for debugging
                        for key in dict_keys[:3]:  # Show first 3 keys
                            value_type = type(example[field][key]).__name__
                            LOG.debug(f"  {key}: {value_type}")
                    else:
                        LOG.warning(f"Error transforming image in field '{field}': {e}")
                        LOG.warning(f"Image type: {image_type}")
                    
                    if not self.skip_on_error:
                        return None
            
            return result
        
        LOG.info(f"🔍 ImageTransform: Processing {initial_count} examples")
        if self.max_size:
            LOG.info(f"  Max size: {self.max_size}, mode: {self.resize_mode}")
        LOG.info(f"  Image fields: {self.image_fields}")
        LOG.info(f"  Transforms: {len(self.transforms)}")
        
        # Apply transformations
        transformed_dataset = dataset.map(process_function)
        final_count = len(transformed_dataset)
        
        # Log detailed results
        LOG.info(f"📊 ImageTransform Results:")
        LOG.info(f"  ✅ Successfully transformed: {transform_stats['success']}")
        LOG.info(f"  ⏭️  Skipped (missing field): {transform_stats['skipped_missing_field']}")
        LOG.info(f"  ⏭️  Skipped (null image): {transform_stats['skipped_none_image']}")
        LOG.info(f"  ❌ Errors: {transform_stats['error_count']}")
        LOG.info(f"  📈 Total processed: {transform_stats['total_processed']}")
        
        return transformed_dataset
    
    def process_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply image transformations to the example."""
        result = example.copy()
        
        for field in self.image_fields:
            if field not in result:
                LOG.debug(f"Image field '{field}' not found in example")
                continue
            
            image = result[field]
            if image is None:
                LOG.debug(f"Image field '{field}' is None")
                continue
            
            try:
                # Apply transformations
                transformed_image = self._apply_transforms(image)
                result[field] = transformed_image
                LOG.debug(f"Successfully transformed image in field '{field}'")
                
            except Exception as e:
                LOG.warning(f"Error transforming image in field '{field}': {e}")
                if self.skip_on_error:
                    LOG.debug(f"Skipping transformation for field '{field}' due to error")
                    continue
                else:
                    return None
        
        return result
    
    def _apply_transforms(self, image: Any) -> Any:
        """Apply all transforms to a single image."""
        current_image = image
        
        # Convert to PIL if needed
        if not isinstance(current_image, Image.Image):
            current_image = self._to_pil_image(current_image)
        
        # Apply each transform
        for transform in self.transforms:
            current_image = self._apply_single_transform(current_image, transform)
        
        # Convert to desired output format
        if self.output_format == "tensor":
            current_image = self._to_tensor(current_image)
        elif self.output_format == "pil":
            if not isinstance(current_image, Image.Image):
                current_image = self._to_pil_image(current_image)
        
        return current_image
    
    def _apply_single_transform(self, image: Any, transform: Dict[str, Any]) -> Any:
        """Apply a single transform to an image."""
        transform_type = transform["type"]
        
        if transform_type == "resize":
            return self._resize(image, transform)
        elif transform_type == "center_crop":
            return self._center_crop(image, transform)
        elif transform_type == "random_crop":
            return self._random_crop(image, transform)
        elif transform_type == "resized_crop":
            return self._resized_crop(image, transform)
        elif transform_type == "grayscale":
            return self._grayscale(image, transform)
        elif transform_type == "color_jitter":
            return self._color_jitter(image, transform)
        elif transform_type == "normalize":
            return self._normalize(image, transform)
        elif transform_type == "to_tensor":
            return self._to_tensor(image)
        elif transform_type == "to_pil":
            return self._to_pil_image(image)
        else:
            LOG.warning(f"Unknown transform type: {transform_type}")
            return image
    
    def _resize(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """Resize image."""
        size = params["size"]
        interpolation = params.get("interpolation", "bilinear")
        mode = params.get("mode", "exact")
        
        # Convert size to tuple if needed
        if isinstance(size, int):
            # Resize smallest edge to this size
            w, h = image.size
            if w < h:
                size = (size, int(size * h / w))
            else:
                size = (int(size * w / h), size)
        elif isinstance(size, list):
            size = tuple(size)
        
        # Map interpolation names to PIL constants
        interp_map = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        
        pil_interpolation = interp_map.get(interpolation.lower(), Image.BILINEAR)
        
        # Handle different resize modes
        if mode == "keep_aspect_ratio":
            # Use thumbnail to maintain aspect ratio
            w, h = image.size
            max_w, max_h = size
            
            # Only resize if image is larger than max size
            if w > max_w or h > max_h:
                image_copy = image.copy()
                image_copy.thumbnail((max_w, max_h), pil_interpolation)
                return image_copy
            else:
                return image
        else:
            # Exact resize (default behavior)
            return image.resize(size, pil_interpolation)
    
    def _center_crop(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """Center crop image."""
        size = params["size"]
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, list):
            size = tuple(size)
        
        return ImageOps.fit(image, size, Image.LANCZOS, centering=(0.5, 0.5))
    
    def _random_crop(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """Random crop image (for this implementation, we'll do center crop)."""
        # Note: True random crop would require random number generation
        # For deterministic processing, we'll use center crop
        LOG.debug("Using center crop for random_crop (deterministic processing)")
        return self._center_crop(image, params)
    
    def _resized_crop(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """Crop and resize image."""
        top = params["top"]
        left = params["left"]
        height = params["height"]
        width = params["width"]
        size = params["size"]
        
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, list):
            size = tuple(size)
        
        # Crop then resize
        cropped = image.crop((left, top, left + width, top + height))
        return cropped.resize(size, Image.BILINEAR)
    
    def _grayscale(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """Convert image to grayscale."""
        num_output_channels = params.get("num_output_channels", 1)
        
        grayscale_img = ImageOps.grayscale(image)
        
        if num_output_channels == 3:
            # Convert back to RGB but with grayscale values
            return grayscale_img.convert("RGB")
        
        return grayscale_img
    
    def _color_jitter(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """Apply color jitter to image."""
        brightness = params.get("brightness", 0)
        contrast = params.get("contrast", 0)
        saturation = params.get("saturation", 0)
        hue = params.get("hue", 0)
        
        # For deterministic processing, we'll apply fixed adjustments
        # In a real scenario, these would be random
        result = image
        
        # Apply brightness
        if brightness > 0:
            brightness_factor = 1.0 + (brightness * 0.5)  # Use middle of range
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(brightness_factor)
        
        # Apply contrast
        if contrast > 0:
            contrast_factor = 1.0 + (contrast * 0.5)
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(contrast_factor)
        
        # Apply saturation
        if saturation > 0:
            saturation_factor = 1.0 + (saturation * 0.5)
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(saturation_factor)
        
        # Note: Hue adjustment is more complex and would need HSV conversion
        if hue > 0:
            LOG.debug("Hue adjustment not implemented in PIL-only mode")
        
        return result
    
    def _normalize(self, image: Any, params: Dict[str, Any]) -> Any:
        """Normalize image tensor."""
        mean = params["mean"]
        std = params["std"]
        
        if not HAS_TORCHVISION:
            LOG.warning("Normalize requires torchvision, skipping")
            return image
        
        # Convert to tensor if needed
        if isinstance(image, Image.Image):
            image = self._to_tensor(image)
        
        # Apply normalization
        if HAS_TORCHVISION:
            return F.normalize(image, mean, std)
        else:
            return image
    
    def _to_tensor(self, image: Any) -> Any:
        """Convert PIL image to tensor."""
        if isinstance(image, Image.Image):
            if HAS_TORCHVISION:
                return F.to_tensor(image)
            elif HAS_NUMPY:
                # Fallback to numpy
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
                return img_array.astype(np.float32) / 255.0
            else:
                LOG.warning("Cannot convert to tensor without torchvision or numpy")
                return image
        else:
            return image
    
    def _to_pil_image(self, image: Any) -> Image.Image:
        """Convert tensor, array, bytes, or dict to PIL image."""
        if isinstance(image, Image.Image):
            return image
        
        # Handle dictionary format (common in HuggingFace datasets)
        if isinstance(image, dict):
            try:
                # Try common dictionary keys for image data
                if "bytes" in image:
                    import io
                    return Image.open(io.BytesIO(image["bytes"]))
                elif "path" in image:
                    return Image.open(image["path"])
                elif "file" in image:
                    return Image.open(image["file"])
                elif "image" in image:
                    # Recursive call if nested
                    return self._to_pil_image(image["image"])
                elif "data" in image:
                    import io
                    return Image.open(io.BytesIO(image["data"]))
                else:
                    # Try to find any bytes-like value in the dict
                    for key, value in image.items():
                        if isinstance(value, bytes):
                            import io
                            return Image.open(io.BytesIO(value))
                    
                    LOG.warning(f"Dictionary image format not recognized. Keys: {list(image.keys())}")
                    raise ValueError(f"Cannot find image data in dictionary with keys: {list(image.keys())}")
            except Exception as e:
                LOG.warning(f"Failed to convert dictionary to PIL image: {e}")
                raise ValueError(f"Cannot convert dictionary to PIL image: {e}")
        
        # Handle bytes format (common in HuggingFace datasets)
        if isinstance(image, bytes):
            try:
                import io
                return Image.open(io.BytesIO(image))
            except Exception as e:
                LOG.warning(f"Failed to convert bytes to PIL image: {e}")
                raise ValueError(f"Cannot convert bytes to PIL image: {e}")
        
        # Handle file-like objects or paths
        if isinstance(image, str):
            try:
                return Image.open(image)
            except Exception as e:
                LOG.warning(f"Failed to open image from path '{image}': {e}")
                raise ValueError(f"Cannot open image from path: {e}")
        
        if HAS_TORCHVISION and hasattr(image, "shape"):
            try:
                return F.to_pil_image(image)
            except Exception as e:
                LOG.debug(f"Failed to convert with torchvision: {e}")
        
        if HAS_NUMPY and hasattr(image, "shape"):
            try:
                img_array = np.array(image)
                if len(img_array.shape) == 3 and img_array.shape[0] in [1, 3]:
                    # CHW to HWC
                    img_array = img_array.transpose(1, 2, 0)
                
                # Handle different data types
                if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                
                return Image.fromarray(img_array)
            except Exception as e:
                LOG.debug(f"Failed to convert with numpy: {e}")
        
        # Fallback: assume it's already a PIL image or compatible
        if hasattr(image, "size"):
            return image
        
        LOG.warning(f"Cannot convert image of type {type(image)} to PIL")
        raise ValueError(f"Cannot convert image of type {type(image)} to PIL")
    
    def get_required_columns(self) -> List[str]:
        """Return required image field columns."""
        return self.image_fields


# Register the processor
register_processor("image_transform", ImageTransformProcessor)