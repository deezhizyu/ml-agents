"""
ML-Agents Doctor - Diagnostic tool for environment and setup

Usage:
    mlagents-doctor
    python -m mlagents.trainers.cli_doctor
"""

import sys
from typing import List, Tuple, Callable, Any, Optional
from mlagents_envs import logging_util

logger = logging_util.get_logger(__name__)


class DiagnosticCheck:
    """Represents a single diagnostic check"""

    def __init__(
        self,
        name: str,
        check_func: Callable[[], Tuple[bool, str]],
        fix_suggestion: str = "",
    ):
        self.name = name
        self.check_func = check_func
        self.fix_suggestion = fix_suggestion
        self.result: Optional[Tuple[bool, str]] = None
        self.message = ""

    def run(self) -> bool:
        """Run the diagnostic check"""
        try:
            self.result = self.check_func()
            return self.result[0] if isinstance(self.result, tuple) else self.result
        except Exception as e:
            import traceback

            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.result = (False, error_msg)
            logger.error(
                f"Diagnostic check '{self.name}' failed with exception: {e}",
                exc_info=True,
            )
            return False


class MLAgentsDoctor:
    """Diagnostic tool for ML-Agents environment"""

    def __init__(self):
        self.checks: List[DiagnosticCheck] = []
        self._register_checks()

    def _register_checks(self):
        """Register all diagnostic checks"""
        self.checks.append(
            DiagnosticCheck(
                "Python Version",
                self._check_python_version,
                "Install Python 3.10 or 3.11 for best compatibility",
            )
        )

        self.checks.append(
            DiagnosticCheck(
                "PyTorch Installation",
                self._check_pytorch,
                "Install PyTorch: pip install torch",
            )
        )

        self.checks.append(
            DiagnosticCheck(
                "NumPy Installation",
                self._check_numpy,
                "Install NumPy: pip install numpy",
            )
        )

        self.checks.append(
            DiagnosticCheck(
                "ML-Agents Installation",
                self._check_mlagents,
                "Install ML-Agents: pip install -e ./ml-agents",
            )
        )

        self.checks.append(
            DiagnosticCheck(
                "GPU Availability",
                self._check_gpu,
                "Install CUDA-enabled PyTorch for GPU support",
            )
        )

        self.checks.append(
            DiagnosticCheck(
                "TorchScript Support",
                self._check_torchscript,
                "Update PyTorch: pip install --upgrade torch",
            )
        )

    def _check_python_version(self) -> Tuple[bool, str]:
        """Check Python version"""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major == 3 and version.minor in [10, 11]:
            return (True, f"Python {version_str} ✓ (Optimal)")
        elif version.major == 3 and version.minor == 12:
            return (True, f"Python {version_str} ✓ (May have compatibility issues)")
        elif version.major == 3 and version.minor >= 9:
            return (True, f"Python {version_str} ✓")
        else:
            return (False, f"Python {version_str} ✗ (Requires Python 3.10-3.11)")

    def _check_pytorch(self) -> Tuple[bool, str]:
        """Check PyTorch installation"""
        try:
            import torch

            return (True, f"PyTorch {torch.__version__} ✓")
        except ImportError:
            return (False, "PyTorch not found ✗")

    def _check_numpy(self) -> Tuple[bool, str]:
        """Check NumPy installation"""
        try:
            import numpy as np

            return (True, f"NumPy {np.__version__} ✓")
        except ImportError:
            return (False, "NumPy not found ✗")

    def _check_mlagents(self) -> Tuple[bool, str]:
        """Check ML-Agents installation"""
        try:
            import mlagents.trainers

            return (True, f"ML-Agents {mlagents.trainers.__version__} ✓")
        except ImportError:
            return (False, "ML-Agents not found ✗")

    def _check_gpu(self) -> Tuple[bool, str]:
        """Check GPU availability"""
        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                return (True, f"GPU available: {gpu_name} ({gpu_count} device(s)) ✓")
            else:
                return (True, "GPU not available (CPU only) ℹ")
        except Exception as e:
            return (False, f"Cannot check GPU status: {str(e)} ✗")

    def _check_torchscript(self) -> Tuple[bool, str]:
        """Check TorchScript support"""
        try:
            import torch
            import torch.nn as nn

            # Test basic TorchScript compilation
            class TestModel(nn.Module):
                def forward(self, x):
                    return x * 2

            model = TestModel()
            traced = torch.jit.trace(model, torch.randn(1))

            return (True, "TorchScript support ✓")
        except Exception as e:
            # Log full error, return truncated version for display
            logger.error(f"TorchScript check failed: {e}", exc_info=True)
            error_preview = str(e)[:100] + ("..." if len(str(e)) > 100 else "")
            return (False, f"TorchScript not working: {error_preview} ✗")

    def run_diagnostics(self) -> bool:
        """Run all diagnostic checks"""
        print("=" * 60)
        print("ML-Agents Environment Diagnostics")
        print("=" * 60)
        print()

        all_passed = True

        for check in self.checks:
            success = check.run()
            result = check.result

            if isinstance(result, tuple):
                status, message = result
            else:
                status = result
                message = "Check completed"

            print(f"{check.name}:")
            print(f"  {message}")

            if not status:
                all_passed = False
                if check.fix_suggestion:
                    print(f"  Suggestion: {check.fix_suggestion}")

            print()

        print("=" * 60)
        if all_passed:
            print("✓ All checks passed! Environment is ready.")
        else:
            print("✗ Some checks failed. Please review suggestions above.")
        print("=" * 60)

        return all_passed

    def quick_check(self) -> bool:
        """Run quick essential checks only"""
        essential = ["Python Version", "PyTorch Installation", "NumPy Installation"]

        print("Quick Environment Check")
        print("-" * 40)

        all_passed = True
        for check in self.checks:
            if check.name in essential:
                success = check.run()
                result = check.result

                if isinstance(result, tuple):
                    status, message = result
                    print(f"{check.name}: {message}")
                    if not status:
                        all_passed = False

        return all_passed


def main():
    """Main entry point for mlagents-doctor command"""
    doctor = MLAgentsDoctor()

    # Check for command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = doctor.quick_check()
    else:
        success = doctor.run_diagnostics()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
