# Experiment Configuration Design

## Goal

To simplify the configuration and execution of `bgbench` experiments by allowing users to define experiment parameters in a YAML file. This approach makes configurations easier to manage, version control, and share, while still allowing for quick overrides using command-line arguments.

## Design Overview

We will introduce a mechanism to load experiment settings from a YAML configuration file. Command-line arguments will serve as overrides to the settings specified in the YAML file. If a setting is not provided in either the YAML file or via a command-line argument, a predefined default value will be used.

### Key Components

1.  **YAML Configuration File:** A user-provided file (e.g., `experiment.yaml`) containing key-value pairs for experiment settings. The structure will mirror the existing command-line arguments (e.g., `game`, `players`, `parallel_games`, `cost_budget`, etc.).
2.  **Configuration Loader (`bgbench/config_loader.py`):** A new Python module responsible for:
    *   Parsing the YAML configuration file using the `PyYAML` library.
    *   Merging the settings loaded from YAML with the arguments provided via the command line (`argparse`).
    *   Handling the precedence rules (CLI > YAML > Defaults).
    *   Returning a single configuration dictionary containing the final settings.
3.  **Main Script (`bgbench/main.py`):**
    *   Will be updated to include a new `--config` command-line argument to specify the path to the YAML configuration file.
    *   Will utilize the `config_loader` module to obtain the final configuration settings after parsing initial arguments.
    *   Will be refactored to use the configuration dictionary returned by the loader instead of directly accessing the `argparse` namespace for most settings.

### Precedence Rules

The final configuration settings will be determined based on the following order of precedence:

1.  **Command-Line Arguments:** Any argument explicitly provided on the command line will override settings from the YAML file and default values.
2.  **YAML Configuration File:** Settings defined in the YAML file specified by `--config` will be used if not overridden by a command-line argument.
3.  **Default Values:** Predefined default values (managed within `argparse` or the `config_loader`) will be used for any setting not specified via the command line or the YAML file.

## Implementation Plan

Here is a step-by-step plan to implement this feature:

*   [ ] **1. Add Dependency:**
    *   Add `PyYAML` to the project dependencies using `poetry add pyyaml`.
*   [ ] **2. Create Configuration Loader Module (`bgbench/config_loader.py`):**
    *   Create the new file `bgbench/config_loader.py`.
    *   Implement a function `load_yaml_config(file_path)` to safely load and parse the YAML file.
    *   Implement a function `merge_configs(yaml_config, args, parser)` to merge YAML settings with `argparse` arguments, respecting precedence rules. This function needs access to the `ArgumentParser` instance to check defaults.
    *   Implement an orchestrator function `load_and_merge_config(parser, args)` that calls the loading and merging functions.
*   [ ] **3. Update Main Script (`bgbench/main.py`):**
    *   Import the necessary functions from `bgbench.config_loader`.
    *   Add a `--config` argument (e.g., `dest="config_file"`) to the `ArgumentParser`.
    *   Perform an initial parse of arguments to get the `config_file` path (and potentially the `debug` flag).
    *   Call `load_and_merge_config` to get the final configuration dictionary.
    *   Refactor the rest of `main.py` to use the returned configuration dictionary (e.g., `config['game']`, `config.get('parallel_games')`) instead of `args.game`, `args.parallel_games`, etc.
    *   Review and potentially adjust where default values are defined (consider centralizing them in `parser.set_defaults` before the merge if removed from `add_argument`).
*   [ ] **4. Add Example Configuration:**
    *   Create an example YAML configuration file (e.g., `examples/experiment_config.yaml`) demonstrating the structure and available settings.
*   [ ] **5. Write Unit Tests:**
    *   Add unit tests for the `bgbench/config_loader.py` module to verify YAML loading and merging logic under various scenarios (CLI overrides, YAML only, defaults only, missing file).
*   [ ] **6. Update Documentation:**
    *   Update this document (`docs/EXPERIMENT_CONFIG.md`) with usage instructions and the example.
    *   Update the main `README.md` to explain the new `--config` option and the YAML configuration format.

## Example YAML Structure

```yaml
# examples/experiment_config.yaml
game: "nim"
players: "configs/players/basic_players.json"
name: "nim_experiment_yaml"
# description: "Optional description" # Example of an optional field

# Arena settings
parallel_games: 5
cost_budget: 1.50
confidence_threshold: 0.75
max_games_per_pair: 15
max_concurrent_games_per_pair: 2
# selected_players: "player1,player3" # Example of a list-like string

# Other settings
debug: false
```
