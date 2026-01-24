"""
Configuration Upgrade Tool for ML-Agents

This tool helps migrate deprecated configuration options to their current equivalents.
It scans YAML configuration files and automatically updates deprecated fields.

Usage:
    python -m mlagents.trainers.upgrade_config <config_file.yaml> [--output <output_file.yaml>] [--dry-run]

Examples:
    # Show what would be changed without modifying the file
    python -m mlagents.trainers.upgrade_config my_config.yaml --dry-run
    
    # Upgrade in place (backs up original to .backup)
    python -m mlagents.trainers.upgrade_config my_config.yaml
    
    # Upgrade and save to new file
    python -m mlagents.trainers.upgrade_config old.yaml --output new.yaml
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)


class ConfigUpgrader:
    """Upgrades deprecated ML-Agents configuration fields to current versions."""
    
    # Deprecation timeline: These fields will be REMOVED in version 5.0
    REMOVAL_VERSION = "5.0"
    CURRENT_VERSION = "4.0"
    
    DEPRECATED_REWARD_SIGNAL_FIELDS = {
        "encoding_size": {
            "replacement": "network_settings.hidden_units",
            "migration": lambda v: {"network_settings": {"hidden_units": v}},
            "message": "'encoding_size' is deprecated. Use 'network_settings.hidden_units' instead.",
        }
    }
    
    DEPRECATED_ROOT_FIELDS = {
        "framework": {
            "replacement": None,  # No replacement - PyTorch only now
            "migration": lambda v: {},
            "message": "'framework' field is deprecated. ML-Agents now uses PyTorch only.",
        }
    }
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.changes: List[str] = []
        self.warnings: List[str] = []
    
    def upgrade_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upgrade configuration dictionary, replacing deprecated fields.
        
        :param config: Configuration dictionary to upgrade
        :return: Upgraded configuration dictionary
        """
        upgraded = config.copy()
        
        # Check root-level deprecated fields
        for field_name, deprecation in self.DEPRECATED_ROOT_FIELDS.items():
            if field_name in upgraded:
                self._handle_deprecation(
                    upgraded, field_name, deprecation, path="root"
                )
        
        # Check behavior-level configs
        if "behaviors" in upgraded:
            for behavior_name, behavior_config in upgraded["behaviors"].items():
                self._upgrade_behavior_config(
                    behavior_config, path=f"behaviors.{behavior_name}"
                )
        
        return upgraded
    
    def _upgrade_behavior_config(self, config: Dict[str, Any], path: str) -> None:
        """Upgrade a single behavior configuration."""
        
        # Check reward signals
        if "reward_signals" in config:
            for signal_name, signal_config in config["reward_signals"].items():
                self._upgrade_reward_signal(
                    signal_config, path=f"{path}.reward_signals.{signal_name}"
                )
    
    def _upgrade_reward_signal(self, config: Dict[str, Any], path: str) -> None:
        """Upgrade reward signal configuration."""
        for field_name, deprecation in self.DEPRECATED_REWARD_SIGNAL_FIELDS.items():
            if field_name in config:
                self._handle_deprecation(config, field_name, deprecation, path=path)
    
    def _handle_deprecation(
        self, 
        config: Dict[str, Any], 
        field_name: str, 
        deprecation: Dict[str, Any],
        path: str
    ) -> None:
        """
        Handle a deprecated field by migrating it to new format.
        
        :param config: Configuration dictionary containing the deprecated field
        :param field_name: Name of the deprecated field
        :param deprecation: Deprecation information dictionary
        :param path: Path to the field in the config (for logging)
        """
        old_value = config[field_name]
        
        # Log the deprecation
        warning = f"{path}.{field_name}: {deprecation['message']}"
        logger.warning(warning)
        self.warnings.append(warning)
        
        # Apply migration
        if deprecation["migration"]:
            new_fields = deprecation["migration"](old_value)
            
            # Merge new fields into config
            for key, value in new_fields.items():
                if isinstance(value, dict) and key in config:
                    # Merge nested dictionaries
                    config[key] = self._merge_dicts(config[key], value)
                else:
                    config[key] = value
            
            change = f"  Migrated {path}.{field_name}={old_value} -> {deprecation['replacement']}"
            self.changes.append(change)
        
        # Remove deprecated field
        del config[field_name]
        self.changes.append(f"  Removed deprecated field: {path}.{field_name}")
    
    @staticmethod
    def _merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two dictionaries, with dict2 taking precedence."""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigUpgrader._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    def upgrade_file(self, input_path: Path, output_path: Path) -> bool:
        """
        Upgrade a configuration file.
        
        :param input_path: Path to input configuration file
        :param output_path: Path to output configuration file
        :return: True if changes were made, False otherwise
        """
        # Load config
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration file {input_path}: {e}")
            return False
        
        if config is None:
            logger.error(f"Configuration file {input_path} is empty")
            return False
        
        # Upgrade
        logger.info(f"Upgrading configuration: {input_path}")
        upgraded_config = self.upgrade_config(config)
        
        # Report changes
        if not self.changes:
            logger.info("✅ No deprecated fields found. Configuration is up to date!")
            return False
        
        logger.info(f"Found {len(self.warnings)} deprecated field(s):")
        for warning in self.warnings:
            logger.info(f"  {warning}")
        
        logger.info(f"\nChanges to be made:")
        for change in self.changes:
            logger.info(change)
        
        # Save if not dry run
        if not self.dry_run:
            # Backup original if overwriting
            if input_path == output_path:
                backup_path = Path(str(input_path) + ".backup")
                logger.info(f"\nBacking up original to: {backup_path}")
                shutil.copy2(input_path, backup_path)
            
            # Write upgraded config
            logger.info(f"Writing upgraded configuration to: {output_path}")
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(upgraded_config, f, default_flow_style=False, sort_keys=False)
            
            logger.info("✅ Configuration upgraded successfully!")
        else:
            logger.info("\n[DRY RUN] No files were modified.")
        
        return True


def main():
    """Main entry point for the configuration upgrade tool."""
    parser = argparse.ArgumentParser(
        description="Upgrade ML-Agents configuration files by migrating deprecated fields.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to configuration file to upgrade"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path to output file (default: overwrite input file)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.config_file.exists():
        logger.error(f"Configuration file not found: {args.config_file}")
        sys.exit(1)
    
    # Default output to input file
    output_path = args.output if args.output else args.config_file
    
    # Create upgrader and process file
    upgrader = ConfigUpgrader(dry_run=args.dry_run)
    
    try:
        had_changes = upgrader.upgrade_file(args.config_file, output_path)
        
        if had_changes and not args.dry_run:
            logger.info(
                f"\n⚠️  DEPRECATION NOTICE: "
                f"These deprecated fields will be REMOVED in ML-Agents {ConfigUpgrader.REMOVAL_VERSION}. "
                f"Please update your configurations before upgrading."
            )
        
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to upgrade configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
