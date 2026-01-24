#!/usr/bin/env python3
"""
Script to profile ML-Agents training performance

Usage:
    python scripts/profile_training.py --config config/ppo/3DBall.yaml --run-id profile_test

With py-spy:
    py-spy record -o profile.svg -- python scripts/profile_training.py --config config/ppo/3DBall.yaml
"""

import argparse
import sys
import time
import cProfile
import pstats
import io
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "ml-agents"))

from mlagents.trainers.learn import run_cli
from mlagents.trainers.utils.profiling import (
    PerformanceMonitor,
    get_global_monitor,
    enable_profiling,
)


def profile_with_cprofile(args):
    """Profile using Python's cProfile"""
    print("Profiling with cProfile...")

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        # Run training for limited steps
        run_cli(args)
    finally:
        profiler.disable()

        # Print results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(50)  # Top 50 functions

        print("\n" + "=" * 80)
        print("TOP 50 FUNCTIONS BY CUMULATIVE TIME")
        print("=" * 80)
        print(s.getvalue())

        # Save to file
        profiler.dump_stats("training_profile.prof")
        print(f"\nProfile saved to: training_profile.prof")
        print("Visualize with: snakeviz training_profile.prof")


def profile_with_custom_monitor():
    """Profile using custom performance monitor"""
    print("Custom performance monitoring enabled")
    enable_profiling(True)
    monitor = get_global_monitor()

    # Monitor will collect metrics during training
    # Log summary at the end
    return monitor


def main():
    parser = argparse.ArgumentParser(
        description="Profile ML-Agents training performance"
    )
    parser.add_argument(
        "--config",
        default="config/ppo/3DBall.yaml",
        help="Path to training config",
    )
    parser.add_argument(
        "--run-id",
        default="profile_test",
        help="Run ID for this training session",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum training steps (for profiling)",
    )
    parser.add_argument(
        "--profiler",
        choices=["cprofile", "custom", "none"],
        default="custom",
        help="Profiler to use",
    )
    parser.add_argument(
        "--env",
        help="Path to Unity environment executable",
    )

    args = parser.parse_args()

    # Build arguments for mlagents-learn
    train_args = [
        args.config,
        "--run-id",
        args.run_id,
        "--force",  # Overwrite existing run
    ]

    if args.env:
        train_args.extend(["--env", args.env])

    # Add max steps
    train_args.extend(["--max-steps", str(args.max_steps)])

    print(f"Profiling training with config: {args.config}")
    print(f"Run ID: {args.run_id}")
    print(f"Max steps: {args.max_steps}")
    print(f"Profiler: {args.profiler}")
    print("=" * 80)

    if args.profiler == "cprofile":
        profile_with_cprofile(train_args)
    elif args.profiler == "custom":
        monitor = profile_with_custom_monitor()
        try:
            run_cli(train_args)
        finally:
            monitor.log_summary()
    else:
        # No profiling
        run_cli(train_args)

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)

    if args.profiler == "cprofile":
        print("\nAnalyze with:")
        print("  snakeviz training_profile.prof")
        print("  python -m pstats training_profile.prof")

    print("\nFor detailed profiling, use py-spy:")
    print(f"  py-spy record -o profile.svg -- python {' '.join(sys.argv)}")
    print(f"  py-spy top -- python {' '.join(sys.argv)}")


if __name__ == "__main__":
    main()
