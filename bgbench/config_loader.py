import argparse
import logging
import os
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_yaml_config(file_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file.

    Args:
        file_path: The path to the YAML configuration file.

    Returns:
        A dictionary containing the configuration loaded from the file.
        Returns an empty dictionary if the file doesn't exist or is empty.

    Raises:
        FileNotFoundError: If the file_path does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
        Exception: For other file reading errors.
    """
    # No need to log existence check unless debugging
    # if not os.path.exists(file_path):
    #     logger.error(f"Configuration file not found: {file_path}")
    #     raise FileNotFoundError(f"Configuration file not found: {file_path}")
    try:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
            # Return the loaded config or an empty dict if the file is empty
            return config if config else {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise  # Re-raise the error after logging
    except Exception as e:
        logger.error(f"Error reading configuration file {file_path}: {e}")
        raise


def merge_configs(
    yaml_config: Dict[str, Any],
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> Dict[str, Any]:
    """Merges YAML configuration with command-line arguments.

    Command-line arguments take precedence over YAML settings if they are
    explicitly provided (i.e., not their default value). Action flags like
    --list, --export, --export-experiment are ignored during merge as they
    are handled directly in main.py.

    Args:
        yaml_config: Configuration loaded from the YAML file.
        args: Parsed command-line arguments namespace.
        parser: The ArgumentParser instance used to parse args.

    Returns:
        A dictionary containing the final merged configuration.
    """
    logger.debug("Starting config merge process") # Changed to debug
    logger.debug(f"YAML config: {yaml_config}") # Changed to debug
    logger.debug(f"Args: {vars(args)}") # Changed to debug
    
    merged_config = yaml_config.copy()
    args_dict = vars(args)

    # Define action flags that should not be merged from YAML or CLI defaults
    # These are handled directly based on their presence in the CLI args.
    action_flags = {"list", "export", "export_experiment"}

    # First, start with YAML config values
    logger.debug(f"Starting with YAML config: {merged_config}") # Changed to debug
    
    # Then, add any CLI arguments that aren't in the YAML config
    # or override YAML values with CLI values when appropriate
    for action in parser._actions:
        arg_name = action.dest
        # Skip help action and specific action flags
        if arg_name == "help" or arg_name in action_flags:
            continue

        cli_value = args_dict.get(arg_name)
        default_value = parser.get_default(arg_name)
        
        logger.debug(f"Processing arg: {arg_name}, CLI value: {cli_value}, Default: {default_value}") # Changed to debug
        
        # Check if this is a CLI-provided value (not default)
        is_cli_provided = False
        
        # For boolean flags
        if isinstance(action, argparse._StoreTrueAction):
            is_cli_provided = cli_value is True
            logger.debug(f"  StoreTrueAction: is_cli_provided={is_cli_provided}") # Changed to debug
        elif isinstance(action, argparse._StoreFalseAction):
            is_cli_provided = cli_value is False
            logger.debug(f"  StoreFalseAction: is_cli_provided={is_cli_provided}") # Changed to debug
        else:
            # For non-boolean arguments, we need a different approach
            # In tests, we can't detect if an argument was explicitly provided
            # So we'll check if the CLI value is different from the default
            is_cli_provided = cli_value != default_value
            logger.debug(f"  Regular arg: is_cli_provided={is_cli_provided} (cli_value != default_value)") # Changed to debug

        # Apply the correct precedence rules:
        # 1. If CLI value was explicitly provided, use it (highest precedence)
        # 2. Otherwise, if value exists in YAML config, use it
        # 3. Otherwise, use the default value from argparse
        
        if is_cli_provided:
            # CLI value was explicitly provided, override YAML
            merged_config[arg_name] = cli_value
            logger.debug(f"  Using CLI value for {arg_name}: {cli_value}") # Changed to debug
        elif arg_name in yaml_config:
            # YAML value exists and CLI didn't override it
            # Keep the YAML value (already in merged_config from the copy)
            logger.debug(f"  Using YAML value for {arg_name}: {yaml_config[arg_name]}") # Changed to debug
        else:
            # Neither CLI nor YAML provided a value, use default
            merged_config[arg_name] = default_value
            logger.debug(f"  Using default value for {arg_name}: {default_value}") # Changed to debug

    logger.debug(f"Final merged config: {merged_config}") # Changed to debug
    return merged_config


def load_and_merge_config(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Dict[str, Any]:
    """Loads YAML config if specified and merges it with CLI arguments.

    Args:
        parser: The ArgumentParser instance.
        args: The initially parsed command-line arguments (must include 'config_file').

    Returns:
        The final configuration dictionary. Returns args as dict if no config file.

    Raises:
        FileNotFoundError: If the specified config_file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    logger.debug("=== Starting load_and_merge_config ===") # Changed to debug
    logger.debug(f"Initial args: {vars(args)}") # Changed to debug
    
    yaml_config = {}
    config_file_path = getattr(args, 'config_file', None) # Use getattr for safety

    if config_file_path:
        logger.info(f"Loading configuration from: {config_file_path}") # Keep info level for this
        # load_yaml_config will raise errors if file not found or invalid
        yaml_config = load_yaml_config(config_file_path)
        logger.debug(f"Loaded YAML config: {yaml_config}") # Changed to debug
    else:
        logger.info("No configuration file specified (--config). Using command-line arguments and defaults.") # Keep info level
        # If no config file, the "merged" config starts as just the args defaults
        # The merge_configs function will then overlay any non-default CLI args.
        # We pass an empty yaml_config here.

    logger.debug("Calling merge_configs...") # Changed to debug
    final_config = merge_configs(yaml_config, args, parser)
    logger.debug(f"Final configuration after merge: {final_config}") # Changed to debug

    return final_config
