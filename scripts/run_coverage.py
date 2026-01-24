#!/usr/bin/env python3
"""
Script to run test coverage analysis and identify gaps

Usage:
    python scripts/run_coverage.py
    python scripts/run_coverage.py --html  # Generate HTML report
"""
import subprocess
import sys
import argparse
from pathlib import Path


def run_coverage(html_report=False, xml_report=False):
    """Run pytest with coverage"""
    project_root = Path(__file__).parent.parent
    
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=ml-agents/mlagents",
        "--cov=ml-agents-envs/mlagents_envs",
        "--cov-report=term-missing",
        "-m",
        "not slow",
        "--tb=short",
        "-v",
    ]
    
    if html_report:
        cmd.append("--cov-report=html")
    
    if xml_report:
        cmd.append("--cov-report=xml")
    
    print("Running test coverage analysis...")
    print(f"Command: {' '.join(cmd)}")
    print("="*60)
    
    result = subprocess.run(cmd, cwd=project_root)
    
    if html_report:
        html_dir = project_root / "htmlcov"
        print("\n" + "="*60)
        print(f"HTML report generated in: {html_dir}")
        print(f"Open: {html_dir / 'index.html'}")
        print("="*60)
    
    return result.returncode


def parse_coverage_output(output: str) -> dict:
    """Parse coverage percentage from output"""
    lines = output.split('\n')
    total_line = [line for line in lines if 'TOTAL' in line]
    
    if total_line:
        # Extract coverage percentage
        parts = total_line[0].split()
        if len(parts) >= 4:
            coverage_pct = parts[-1].replace('%', '')
            try:
                return {'coverage': float(coverage_pct)}
            except ValueError:
                pass
    
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run test coverage analysis")
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--xml",
        action="store_true",
        help="Generate XML coverage report for CI"
    )
    
    args = parser.parse_args()
    
    exit_code = run_coverage(
        html_report=args.html,
        xml_report=args.xml
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
