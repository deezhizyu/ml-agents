import os
import sys
from typing import Any, Optional, TypeVar

# Configure CUDA memory allocator BEFORE importing torch
# This must be set before PyTorch initializes CUDA
# expandable_segments is unsupported on Windows (both CUDA and ROCm builds warn
# "not supported on this platform"), and forcing it there can leave the caching
# allocator in a bad state that hangs on interpreter shutdown.
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from packaging.version import Version  # noqa: E402
import importlib.metadata  # noqa: E402
from mlagents.torch_utils import cpu_utils  # noqa: E402
from mlagents.trainers.settings import TorchSettings  # noqa: E402
from mlagents_envs.logging_util import get_logger  # noqa: E402

logger = get_logger(__name__)

# Type variable for torch.compile
T = TypeVar("T")


def assert_torch_installed():
    # Check that torch version 1.6.0 or later has been installed. If not, refer
    # user to the PyTorch webpage for install instructions.
    torch_version = None
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        pass
    assert torch_version is not None and Version(torch_version) >= Version("1.6.0"), (
        "A compatible version of PyTorch was not installed. Please visit the PyTorch homepage "
        + "(https://pytorch.org/get-started/locally/) and follow the instructions to install. "
        + "Version 1.6.0 and later are supported."
    )


assert_torch_installed()

# This should be the only place that we import torch directly.
# Everywhere else is caught by the banned-modules setting for flake8
import torch  # noqa I201

torch.set_num_threads(cpu_utils.get_num_threads_to_use())
os.environ["KMP_BLOCKTIME"] = "0"


_device = torch.device("cpu")
_torch_settings: Optional[TorchSettings] = None
_amp_enabled = False
_compile_enabled = False
_fused_optimizer_available = False


def set_torch_config(torch_settings: TorchSettings) -> None:
    global _device, _torch_settings, _amp_enabled, _compile_enabled, _fused_optimizer_available

    _torch_settings = torch_settings

    if torch_settings.device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = torch_settings.device

    _device = torch.device(device_str)

    if _device.type == "cuda":
        torch.set_default_device(_device.type)
        torch.set_default_dtype(torch.float32)

        # Enable cudnn.benchmark for faster training with consistent input sizes
        if torch_settings.enable_cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            logger.debug("Enabled cudnn.benchmark for faster training")

        # Enable TF32 on Ampere+ GPUs for faster matmul operations
        if torch_settings.enable_tf32:
            if hasattr(torch.backends.cuda, "matmul") and hasattr(
                torch.backends.cuda.matmul, "allow_tf32"
            ):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
            logger.debug("Enabled TF32 for faster matrix operations on Ampere+ GPUs")

        # Check if AMP is enabled
        _amp_enabled = torch_settings.enable_amp
        if _amp_enabled:
            logger.info("Automatic Mixed Precision (AMP) training enabled")

        # Check if torch.compile is available and enabled (PyTorch 2.0+)
        _compile_enabled = torch_settings.enable_compile and hasattr(torch, "compile")
        if torch_settings.enable_compile and not hasattr(torch, "compile"):
            logger.warning(
                "torch.compile requested but not available (requires PyTorch 2.0+)"
            )
        elif _compile_enabled:
            logger.info("torch.compile enabled for model optimization")

        # Check if fused optimizer is available (PyTorch 2.0+)
        _fused_optimizer_available = torch_settings.enable_fused_optimizer
        if _fused_optimizer_available and getattr(torch.version, "hip", None) is not None:
            # Constructing a fused CUDA optimizer (even without calling .step())
            # has been observed to leave the process unable to exit on some
            # ROCm/GPU combinations (not even os._exit() escapes it), so skip
            # the feature - and its self-test below - entirely on ROCm builds.
            _fused_optimizer_available = False
            logger.debug(
                "Fused optimizer disabled on ROCm (known to hang on process exit "
                "on some GPUs), falling back to standard optimizer"
            )
        elif _fused_optimizer_available:
            # Test if fused is actually supported
            try:
                # Create a small test to verify fused optimizer works
                test_param = torch.nn.Parameter(torch.zeros(1, device=_device))
                torch.optim.Adam([test_param], lr=0.001, fused=True)
                logger.debug("Fused optimizer available and enabled")
            except (TypeError, RuntimeError):
                _fused_optimizer_available = False
                logger.debug(
                    "Fused optimizer not available, falling back to standard optimizer"
                )
    else:
        torch.set_default_dtype(torch.float32)
        _amp_enabled = False
        _compile_enabled = False
        _fused_optimizer_available = False

    logger.debug(f"default Torch device: {_device}")


# Initialize to default settings
set_torch_config(TorchSettings(device=None))

nn = torch.nn


def default_device():
    return _device


def is_amp_enabled() -> bool:
    """Check if Automatic Mixed Precision is enabled."""
    return _amp_enabled


def is_compile_enabled() -> bool:
    """Check if torch.compile is enabled."""
    return _compile_enabled


def is_fused_optimizer_available() -> bool:
    """Check if fused optimizer is available."""
    return _fused_optimizer_available


def maybe_compile(model: T, mode: str = "reduce-overhead") -> T:
    """
    Optionally compile a model using torch.compile if enabled and available.

    :param model: The model to potentially compile.
    :param mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune').
    :return: The compiled model if compilation is enabled, otherwise the original model.
    """
    if _compile_enabled and hasattr(torch, "compile"):
        try:
            return torch.compile(model, mode=mode)
        except Exception as e:
            logger.warning(f"torch.compile failed, using uncompiled model: {e}")
            return model
    return model


def create_optimizer(params: Any, lr: float, **kwargs: Any) -> "torch.optim.Adam":
    """
    Create an Adam optimizer with optional fused optimization.

    :param params: Model parameters to optimize.
    :param lr: Learning rate.
    :param kwargs: Additional optimizer arguments.
    :return: An Adam optimizer instance.
    """
    if _fused_optimizer_available and _device.type == "cuda":
        return torch.optim.Adam(params, lr=lr, fused=True, **kwargs)
    return torch.optim.Adam(params, lr=lr, **kwargs)
