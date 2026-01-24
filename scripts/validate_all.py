#!/usr/bin/env python3
"""Complete validation script for all improvements"""

import sys
import subprocess
import os


def run_command(cmd, description):
    """Run command and report result"""
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=120
        )

        if result.returncode == 0:
            print(f"✓ PASSED - {description}")
            return True
        else:
            print(f"✗ FAILED - {description}")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT - {description}")
        return False
    except Exception as e:
        print(f"✗ ERROR - {description}: {e}")
        return False


def main():
    print("=" * 70)
    print("ML-AGENTS COMPLETE VALIDATION")
    print("=" * 70)

    tests = [
        # Environment
        ("python -m mlagents.trainers.cli_doctor --quick", "Environment Setup"),
        # Phase 2 Tests
        (
            "python -m pytest ml-agents/mlagents/trainers/tests/test_torchscript_optimization.py -v -x",
            "Phase 2: TorchScript Tests",
        ),
        (
            "python -m pytest ml-agents/mlagents/trainers/tests/test_profiling.py -v -x",
            "Phase 2: Profiling Tests",
        ),
        # Phase 3 Tests
        (
            "python -m pytest ml-agents/mlagents/trainers/tests/test_curriculum_scheduler.py -v -x",
            "Phase 3: Curriculum Scheduler Tests",
        ),
        (
            "python -m pytest ml-agents/mlagents/trainers/tests/test_cli_doctor.py -v -x",
            "Phase 3: CLI Doctor Tests",
        ),
    ]

    results = []
    for cmd, desc in tests:
        results.append(run_command(cmd, desc))

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    for i, (_, desc) in enumerate(tests):
        status = "✓ PASS" if results[i] else "✗ FAIL"
        print(f"  {status} - {desc}")

    print("=" * 70)

    if passed == total:
        print("\n✓ ALL VALIDATIONS PASSED - PRODUCTION READY!")
        return 0
    else:
        print(f"\n✗ {total - passed} VALIDATIONS FAILED - REVIEW NEEDED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
