#!/usr/bin/env python3
"""
Python replacement for the Ruby-based pre-commit-search-and-replace hook.
This script performs search and replace operations on files based on rules
defined in .pre-commit-search-and-replace.yaml.
"""
import re
import sys
from pathlib import Path

import yaml


def load_rules(config_path: str = ".pre-commit-search-and-replace.yaml") -> list:
    """Load search and replace rules from the config file."""
    config_file = Path(config_path)
    if not config_file.exists():
        return []
    
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or []


def parse_search_pattern(search: str, insensitive: bool = False) -> re.Pattern:
    """Parse the search pattern, handling regex syntax."""
    # Check if it's a regex pattern (surrounded by /)
    if search.startswith("/") and search.endswith("/"):
        pattern = search[1:-1]
    else:
        # Treat as literal string, escape regex special chars
        pattern = re.escape(search)
    
    flags = re.IGNORECASE if insensitive else 0
    return re.compile(pattern, flags)


def process_file(filepath: str, rules: list) -> bool:
    """
    Process a single file with all rules.
    Returns True if the file was modified.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except (IOError, UnicodeDecodeError):
        return False
    
    original_content = content
    
    for rule in rules:
        search = rule.get("search", "")
        replacement = rule.get("replacement", "")
        insensitive = rule.get("insensitive", False)
        
        if not search:
            continue
        
        pattern = parse_search_pattern(search, insensitive)
        content = pattern.sub(replacement, content)
    
    if content != original_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    
    return False


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: search_and_replace.py <file1> [file2] ...")
        return 0
    
    rules = load_rules()
    if not rules:
        # No rules defined, nothing to do
        return 0
    
    modified_files = []
    for filepath in sys.argv[1:]:
        if process_file(filepath, rules):
            modified_files.append(filepath)
    
    if modified_files:
        print(f"Modified {len(modified_files)} file(s):")
        for f in modified_files:
            print(f"  - {f}")
    
    # Always return 0 on success - pre-commit detects changes by file content
    return 0


if __name__ == "__main__":
    sys.exit(main())
