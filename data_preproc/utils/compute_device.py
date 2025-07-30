"""Compute device utility for selecting optimal compute devices."""

import torch
import logging

LOG = logging.getLogger(__name__)


class ComputeDeviceUtils:
    """Utility class for managing compute device selection."""
    
    CPU_DEVICE_KEY = 'cpu'
    CUDA_DEVICE_KEY = 'cuda'
    XPU_DEVICE_KEY = 'xpu'  # ROCm
    MPS_DEVICE_KEY = 'mps'  # Apple Silicon

    @staticmethod
    def default_device():
        """
        Get the default compute device based on availability.
        
        Priority order:
        1. CUDA GPU (if available)
        2. Intel XPU (if available)
        3. Apple Silicon MPS (if available)
        4. CPU (fallback)
        
        Returns:
            torch.device: The default device to use
        """
        device = ComputeDeviceUtils.CPU_DEVICE_KEY

        # Check for CUDA GPU
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = ComputeDeviceUtils.CUDA_DEVICE_KEY
        # Check for Intel XPU
        elif (hasattr(torch, ComputeDeviceUtils.XPU_DEVICE_KEY) and
                torch.xpu.is_available()):
            device = ComputeDeviceUtils.XPU_DEVICE_KEY
        # Check for Apple Silicon MPS
        elif (hasattr(torch.backends, ComputeDeviceUtils.MPS_DEVICE_KEY) and
                torch.backends.mps.is_available()):
            device = ComputeDeviceUtils.MPS_DEVICE_KEY
        else:
            # No GPU detected. Using CPU.
            pass

        result = torch.device(device)
        LOG.debug(f"Selected compute device: {result}")
        return result

    @staticmethod
    def cpu_device():
        """Get CPU device."""
        return torch.device('cpu')

    @staticmethod
    def get_device_info():
        """
        Get information about available compute devices.

        Returns:
            dict: A dictionary containing device information:
                - device: The default device that would be used (cpu, cuda, mps, etc.)
                - device_name: Human-readable device name
                - cuda_available: Boolean indicating if CUDA is available
                - cuda_device_count: Number of CUDA devices available
                - mps_available: Boolean indicating if Apple MPS is available
                - xpu_available: Boolean indicating if Intel XPU is available
        """
        device = ComputeDeviceUtils.default_device()
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        mps_available = (hasattr(torch.backends, ComputeDeviceUtils.MPS_DEVICE_KEY) and
                         torch.backends.mps.is_available())
        xpu_available = (hasattr(torch, ComputeDeviceUtils.XPU_DEVICE_KEY) and
                         torch.xpu.is_available())

        # Generate human-readable device name
        device_name = device.type
        if device.type == 'cuda' and cuda_device_count > 0:
            device_name = f"CUDA GPU ({cuda_device_count} device{'s' if cuda_device_count > 1 else ''})"
        elif device.type == 'mps':
            device_name = "Apple Silicon MPS"
        elif device.type == 'xpu':
            device_name = "Intel XPU"
        elif device.type == 'cpu':
            device_name = "CPU"

        return {
            'device': device.type,
            'device_name': device_name,
            'cuda_available': cuda_available,
            'cuda_device_count': cuda_device_count,
            'mps_available': mps_available,
            'xpu_available': xpu_available
        }

    @staticmethod
    def move_to_device(tensor_or_model, device=None):
        """
        Move tensor or model to the specified device.
        
        Args:
            tensor_or_model: PyTorch tensor or model to move
            device: Target device (if None, uses default_device())
            
        Returns:
            The tensor or model moved to the specified device
        """
        if device is None:
            device = ComputeDeviceUtils.default_device()
        
        if hasattr(tensor_or_model, 'to'):
            return tensor_or_model.to(device)
        elif hasattr(tensor_or_model, 'cuda') and device.type == 'cuda':
            return tensor_or_model.cuda()
        else:
            return tensor_or_model

    @staticmethod
    def log_device_info():
        """Log information about available compute devices."""
        info = ComputeDeviceUtils.get_device_info()
        LOG.info(f"🔧 Compute Device Info:")
        LOG.info(f"  Selected device: {info['device_name']}")
        LOG.info(f"  CUDA available: {info['cuda_available']}")
        if info['cuda_available']:
            LOG.info(f"  CUDA devices: {info['cuda_device_count']}")
        LOG.info(f"  MPS available: {info['mps_available']}")
        LOG.info(f"  XPU available: {info['xpu_available']}")