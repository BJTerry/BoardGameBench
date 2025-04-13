import os
import tempfile
import pytest
import argparse
import yaml
from typing import Dict, Any

from bgbench.config_loader import load_yaml_config, merge_configs, load_and_merge_config


@pytest.fixture
def sample_parser():
    """Create a sample ArgumentParser for testing config merging."""
    parser = argparse.ArgumentParser()
    # Game has choices but no default in main.py, reflecting that here.
    parser.add_argument("--game", choices=["chess", "nim", "go"]) # Add choices for realism
    parser.add_argument("--players", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--parallel-games", type=int, default=3)
    parser.add_argument("--cost-budget", type=float, default=2.0)
    parser.add_argument("--debug", action="store_true", default=False)
    return parser


def create_yaml_file(content: Dict[str, Any]) -> str:
    """Create a temporary YAML file with given content."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as temp_file:
        yaml.safe_dump(content, temp_file)
        temp_file_path = temp_file.name
    return temp_file_path


def test_load_yaml_config_valid():
    """Test loading a valid YAML configuration file."""
    yaml_content = {
        "game": "chess",
        "players": "players.json",
        "parallel_games": 5
    }
    yaml_path = create_yaml_file(yaml_content)
    
    try:
        loaded_config = load_yaml_config(yaml_path)
        assert loaded_config == yaml_content
    finally:
        os.unlink(yaml_path)


def test_load_yaml_config_empty():
    """Test loading an empty YAML configuration file."""
    yaml_path = create_yaml_file({})
    
    try:
        loaded_config = load_yaml_config(yaml_path)
        assert loaded_config == {}
    finally:
        os.unlink(yaml_path)


def test_load_yaml_config_nonexistent():
    """Test loading a non-existent YAML configuration file."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config("/path/to/nonexistent/config.yaml")


def test_merge_configs_yaml_only(sample_parser):
    """Test merging configuration with only YAML settings."""
    yaml_config = {
        "game": "chess",
        "players": "players.json",
        "parallel_games": 5
    }
    
    args = sample_parser.parse_args([])
    merged_config = merge_configs(yaml_config, args, sample_parser)
    
    assert merged_config['game'] == 'chess'
    assert merged_config['players'] == 'players.json'
    assert merged_config['parallel_games'] == 5
    assert merged_config['cost_budget'] == 2.0  # Default from parser
    assert merged_config['debug'] is False  # Default from parser


def test_merge_configs_cli_override(sample_parser):
    """Test merging configuration with CLI arguments overriding YAML."""
    yaml_config = {
        "game": "chess",
        "players": "players.json",
        "parallel_games": 5
    }
    
    args = sample_parser.parse_args(["--game", "nim", "--parallel-games", "7"])
    merged_config = merge_configs(yaml_config, args, sample_parser)
    
    assert merged_config['game'] == 'nim'  # CLI overrides YAML
    assert merged_config['players'] == 'players.json'  # YAML value preserved
    assert merged_config['parallel_games'] == 7  # CLI overrides YAML
    assert merged_config['cost_budget'] == 2.0  # Default from parser


def test_merge_configs_cli_boolean_flags(sample_parser):
    """Test merging configuration with boolean flags."""
    yaml_config = {
        "debug": True
    }
    
    # No CLI flag
    args = sample_parser.parse_args([])
    merged_config = merge_configs(yaml_config, args, sample_parser)
    assert merged_config['debug'] is True  # YAML value used
    
    # CLI flag overrides YAML
    args = sample_parser.parse_args(["--debug"])
    merged_config = merge_configs(yaml_config, args, sample_parser)
    assert merged_config['debug'] is True  # CLI flag takes precedence


def test_load_and_merge_config(sample_parser, monkeypatch):
    """Test the load_and_merge_config function."""
    yaml_content = {
        "game": "chess",
        "players": "players.json",
        "parallel_games": 5
    }
    yaml_path = create_yaml_file(yaml_content)
    
    try:
        # Create args with config_file set
        args = sample_parser.parse_args([])
        args.config_file = yaml_path
        
        # Test with config file
        config = load_and_merge_config(sample_parser, args)
        assert config['game'] == 'chess'
        assert config['players'] == 'players.json'
        assert config['parallel_games'] == 5
        
        # Test with CLI override
        args = sample_parser.parse_args(["--game", "nim"])
        args.config_file = yaml_path
        config = load_and_merge_config(sample_parser, args)
        assert config['game'] == 'nim'  # CLI overrides YAML
        
        # Test without config file
        args = sample_parser.parse_args(["--game", "nim"])
        args.config_file = None
        config = load_and_merge_config(sample_parser, args)
        assert config['game'] == 'nim'  # CLI value used
        assert config['parallel_games'] == 3  # Default used
    finally:
        os.unlink(yaml_path)
