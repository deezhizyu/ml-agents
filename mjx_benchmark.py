#!/usr/bin/env python3
"""
MuJoCo MJX + Playground Benchmark Script

Run this in WSL2:
    source ~/mjx-env/bin/activate
    python3 mjx_benchmark.py
"""

import time
import sys

# Suppress warp deprecation warnings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*warp.*")

print("=" * 70)
print("MuJoCo MJX + Playground GPU Training Benchmark")
print("=" * 70)

# Import JAX and check GPU
import jax
import jax.numpy as jnp
from functools import partial

print(f"\nJAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Check if GPU is available
gpu_available = any(
    "cuda" in str(d).lower() or "gpu" in str(d).lower() for d in jax.devices()
)
if not gpu_available:
    print("WARNING: No GPU detected! Training will be slow.")
else:
    print("GPU detected: YES")

print("=" * 70)

# Import MuJoCo MJX and Playground
from mujoco import mjx
from mujoco_playground import dm_control_suite

print("\nImports successful!")
print("   - MuJoCo MJX: GPU-accelerated physics")
print("   - MuJoCo Playground dm_control_suite: Classic RL environments")

# List available dm_control_suite environments
DM_CONTROL_ENVS = {
    # Classic locomotion
    "HumanoidWalk": "Humanoid Walking (21 DoF)",
    "HumanoidRun": "Humanoid Running (21 DoF)",
    "WalkerWalk": "Walker Walking (6 DoF)",
    "WalkerRun": "Walker Running (6 DoF)",
    "CheetahRun": "Half Cheetah Running (6 DoF)",
    "HopperHop": "Hopper Hopping (4 DoF)",
    # Simpler tasks
    "CartpoleSwingup": "Cartpole Swingup (1 DoF)",
    "CartpoleBalance": "Cartpole Balance (1 DoF)",
    "ReacherEasy": "Reacher Easy (2 DoF)",
    "ReacherHard": "Reacher Hard (2 DoF)",
    "PendulumSwingup": "Pendulum Swingup (1 DoF)",
    "AcrobotSwingup": "Acrobot Swingup (2 DoF)",
}

print("\nAvailable DM Control Suite Environments:")
for env_id, desc in DM_CONTROL_ENVS.items():
    print(f"   - {env_id}: {desc}")


def run_benchmark(env_name: str, num_timesteps: int = 1_000_000, num_envs: int = 2048):
    """Run a training benchmark using MuJoCo MJX physics."""
    print(f"\n{'-' * 60}")
    print(f"Training: {env_name}")
    print(f"   Physics: MuJoCo MJX (GPU-accelerated)")
    print(f"   Timesteps: {num_timesteps:,}")
    print(f"   Parallel Envs: {num_envs}")
    print(f"{'-' * 60}")

    try:
        # Load environment from MuJoCo Playground dm_control_suite
        env = dm_control_suite.load(env_name)

        print(f"   Observation size: {env.observation_size}")
        print(f"   Action size: {env.action_size}")

        # JIT compile reset and step functions
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        # Vectorize for parallel environments
        batch_reset = jax.vmap(jit_reset)
        batch_step = jax.vmap(jit_step)

        # Initialize random keys
        rng = jax.random.PRNGKey(0)
        rng, *reset_keys = jax.random.split(rng, num_envs + 1)
        reset_keys = jnp.array(reset_keys)

        # Warm-up / JIT compilation
        print("   JIT compiling (first run)...")
        warmup_start = time.time()
        states = batch_reset(reset_keys)

        # Take one step to compile step function
        rng, action_key = jax.random.split(rng)
        actions = jax.random.uniform(
            action_key, (num_envs, env.action_size), minval=-1, maxval=1
        )
        states = batch_step(states, actions)
        jax.block_until_ready(states)
        warmup_time = time.time() - warmup_start
        print(f"   JIT compilation done in {warmup_time:.1f}s")

        # Benchmark: Run many environment steps
        num_steps = num_timesteps // num_envs
        print(
            f"   Running {num_steps:,} batched steps ({num_timesteps:,} total env steps)..."
        )

        start_time = time.time()
        total_reward = 0.0

        for step in range(num_steps):
            # Generate random actions
            rng, action_key = jax.random.split(rng)
            actions = jax.random.uniform(
                action_key, (num_envs, env.action_size), minval=-1, maxval=1
            )

            # Step environments
            states = batch_step(states, actions)

            # Track rewards (sample every 100 steps to reduce overhead)
            if step % 100 == 0:
                total_reward += float(jnp.mean(states.reward))

            # Progress update
            if step > 0 and step % (num_steps // 5) == 0:
                elapsed = time.time() - start_time
                steps_done = step * num_envs
                rate = steps_done / elapsed
                print(f"   Step {step:,}/{num_steps:,}: {rate:,.0f} steps/sec")

        # Wait for all computations to finish
        jax.block_until_ready(states)

        elapsed = time.time() - start_time
        steps_per_sec = num_timesteps / elapsed
        avg_reward = total_reward / (num_steps // 100)

        print(f"\n   Benchmark Complete!")
        print(f"   Time: {elapsed:.2f} seconds")
        print(f"   Avg Reward (random policy): {avg_reward:.3f}")
        print(f"   Throughput: {steps_per_sec:,.0f} steps/sec")
        print(f"   Throughput: {steps_per_sec/1e6:.2f} M steps/sec")

        return {
            "env": env_name,
            "timesteps": num_timesteps,
            "num_envs": num_envs,
            "time": elapsed,
            "steps_per_sec": steps_per_sec,
            "avg_reward": avg_reward,
            "success": True,
        }

    except Exception as e:
        print(f"\n   Error: {e}")
        import traceback

        traceback.print_exc()
        return {
            "env": env_name,
            "timesteps": num_timesteps,
            "num_envs": num_envs,
            "time": 0,
            "steps_per_sec": 0,
            "avg_reward": 0,
            "success": False,
            "error": str(e),
        }


def warmup_gpu():
    """Warm up JAX/GPU with a simple computation."""
    print("\nWarming up GPU...")
    x = jnp.ones((1000, 1000))
    for _ in range(3):
        x = jnp.dot(x, x)
    x.block_until_ready()
    print("   GPU warm-up complete!")


def main():
    # GPU warmup
    warmup_gpu()

    print("\n" + "=" * 70)
    print("Choose benchmark type:")
    print("  1. Quick benchmark (Cheetah, ~30 seconds)")
    print("  2. Full benchmark (multiple environments)")
    print("  3. Single custom environment")
    print("=" * 70)

    try:
        choice = input("Enter choice (1/2/3) [default: 1]: ").strip() or "1"
    except EOFError:
        choice = "1"

    results = []

    if choice == "1":
        # Quick benchmark with Cheetah
        print("\nRunning quick Cheetah benchmark...")
        results.append(
            run_benchmark("CheetahRun", num_timesteps=10_000_000, num_envs=4096)
        )

    elif choice == "2":
        # Full benchmark
        print("\nRunning full benchmark suite...")
        benchmarks = [
            ("CheetahRun", 10_000_000, 4096),
            ("WalkerRun", 10_000_000, 4096),
            ("HumanoidWalk", 10_000_000, 2048),
        ]
        for env_name, steps, envs in benchmarks:
            results.append(run_benchmark(env_name, steps, envs))

    elif choice == "3":
        # Custom environment
        print("\nAvailable environments:")
        for i, (env_id, desc) in enumerate(DM_CONTROL_ENVS.items(), 1):
            print(f"  {i}. {env_id}: {desc}")

        try:
            env_choice = (
                input("Enter environment name [default: CheetahRun]: ").strip()
                or "CheetahRun"
            )
            steps = int(
                input("Enter timesteps [default: 10000000]: ").strip() or "10000000"
            )
            envs = int(input("Enter parallel envs [default: 4096]: ").strip() or "4096")
        except (EOFError, ValueError):
            env_choice, steps, envs = "CheetahRun", 10_000_000, 4096

        results.append(run_benchmark(env_choice, steps, envs))

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(
        f"{'Environment':<20} {'Steps':<14} {'Envs':<8} {'Time':<10} {'Steps/sec':<18}"
    )
    print("-" * 70)

    total_steps_per_sec = []
    for r in results:
        if r["success"]:
            print(
                f"{r['env']:<20} {r['timesteps']:>12,} {r['num_envs']:>6} {r['time']:>8.1f}s {r['steps_per_sec']:>16,.0f}"
            )
            total_steps_per_sec.append(r["steps_per_sec"])
        else:
            print(
                f"{r['env']:<20} {r['timesteps']:>12,} {r['num_envs']:>6} {'FAILED':<10} {'-':<18}"
            )

    print("-" * 70)

    if total_steps_per_sec:
        avg_throughput = sum(total_steps_per_sec) / len(total_steps_per_sec)
        print(
            f"\nAverage Throughput: {avg_throughput:,.0f} steps/sec ({avg_throughput/1e6:.2f} M steps/sec)"
        )

    print("=" * 70)
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
