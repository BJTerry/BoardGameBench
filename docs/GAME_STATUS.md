# Enhanced Experiment Status Logging Design

## Problem

The current logging during an active `bgbench` experiment run provides limited real-time visibility. Key issues include:
-   **Infrequent Updates:** Standings and pairwise confidences only update when a match finishes, making it hard to gauge progress during long matches.
-   **Lack of Ongoing Match Details:** It's impossible to tell which matches are currently running, their progress (e.g., turn number), or their individual costs without digging into verbose logs or the database.
-   **Ambiguous Error/Info Messages:** Log messages like "Invalid move" or "Player conceded" don't specify which match or player they belong to, making debugging difficult, especially with parallel games.
-   **Repetitive Budget Logging:** When the cost budget is hit, the "budget reached" message can log excessively, cluttering the output.

## Proposed Solution

Implement several enhancements to the logging within `bgbench.experiment.arena.Arena` and related components to provide clearer, more frequent, and contextualized status updates.

### 1. Periodic Status Updates

-   **Mechanism:** Introduce a dedicated asynchronous task (`_periodic_status_logger`) within the `Arena` class that runs concurrently with the main `evaluate_all` loop.
-   **Frequency:** This task will wake up every 60 seconds (configurable).
-   **Content:**
    -   **Timestamp:** Log the current time for each update.
    -   **Budget Status:** Display the current total experiment cost against the allocated budget (e.g., `$40.10 / $45.00 (89.1% used)`). This replaces the repetitive logging when the budget is hit.
    -   **Ongoing Matches Table:** Query the database for active (non-complete) `GameMatch` records associated with the current experiment. For each active match, retrieve and display in a `tabulate` table:
        -   Match ID
        -   Player 1 Name
        -   Player 2 Name
        -   Current Turn Number (Fetch the latest `MatchState`, deserialize `game_state` to get turn, or add `turn_number` to `MatchStateData`). *Initial implementation might omit turn number if state deserialization is too complex/slow.*
        -   Current Match Cost (Sum of `LLMInteraction.cost` for this `game_id`).
-   **Control:** The periodic logger task will be started within `evaluate_all`, managed within `_active_tasks`, and stopped gracefully during shutdown.

### 2. Tabulated Standings and Results

-   **Goal:** Improve the readability of player standings both during the experiment (`log_standings`) and in the final results (`print_results`).
-   **Implementation:**
    -   Modify `Arena.log_standings` to format the output using `tabulate`. Include columns for Rank, Player Name, Rating (Mu ± 95% CI), Matches Played, Concessions, and Cost.
    -   Modify `main.print_results` to use a similar `tabulate` format for the final standings section and the skill comparison table.

### 3. Refined Budget Logging

-   **Goal:** Prevent excessive logging when the budget is exceeded while still providing clear status.
-   **Implementation:**
    -   The periodic logger (see point 1) will continuously display the budget status (e.g., `$45.00 / $45.00 (100.0% used)`).
    -   Retain the single, initial log message in `Arena.evaluate_all` when the budget is first detected as exceeded (e.g., "Cost budget of $45.00 reached. Stopping evaluation.").
    -   Remove the separate cost logging logic from the main loop in `evaluate_all` that triggers on cost changes.

### 4. Contextual Logging for Match Events

-   **Goal:** Add context (Match ID, Player Name) to log messages originating from within a specific match run (e.g., invalid moves, concessions, errors).
-   **Implementation:**
    -   **Requires changes primarily in `bgbench/match/runner.py`**.
    -   Pass the `match_id` (already available as `game_id` in `MatchRunner`) and potentially a `logging.LoggerAdapter` or similar mechanism to `MatchRunner`.
    -   Modify logging calls within `MatchRunner` (e.g., in `_get_llm_response`, `play_turn`, error handling) to automatically include `[Match:{match_id}] [Player:{player_name}]` in the log record's metadata or message format. The specific player name would be determined based on whose turn it is or which player's action/response is being processed.

## Implementation Details

-   **Dependencies:** Add `tabulate` to project dependencies (`pyproject.toml` or `requirements.txt`).
-   **Database Queries:** Optimize queries for the periodic status update to minimize performance impact, especially fetching active matches and their costs. Consider adding indices if needed.
-   **Turn Number:** Decide on the approach for getting the turn number for ongoing matches (deserialization vs. adding to `MatchStateData`). If adding to `MatchStateData`, ensure `MatchRunner` updates it when saving state.
-   **Task Management:** Ensure the `_periodic_status_logger` task is properly created, added to `_active_tasks`, and cancelled during `Arena.handle_sigint` and at the end of `evaluate_all`.
-   **Logging Configuration:** May need minor adjustments to the logging format in `bgbench/logging_config.py` if using `LoggerAdapter` for contextual logging.

## Benefits

-   **Improved Visibility:** Clear, regular updates on experiment progress, budget, and active games.
-   **Easier Debugging:** Quickly identify which match and player are associated with errors or specific events.
-   **Enhanced Readability:** Standings and results presented in a clean, tabular format.
-   **Reduced Log Noise:** Less clutter from repetitive budget messages.

## TODO List

1.  **Add Dependency:** Add `tabulate` to the project's dependencies. (Already present)
2.  **Implement Periodic Logger Task:** **DONE (Turn number omitted)**
    -   Create the `_periodic_status_logger` async function in `Arena`.
    -   Implement database queries to fetch active matches, player names, and costs.
    -   Implement logic to fetch/calculate turn numbers (or omit initially). -> *Omitted for now*
    -   Format output using `tabulate`.
    -   Integrate the task into `evaluate_all` loop and `_active_tasks` management.
    -   Ensure graceful shutdown.
3.  **Refactor Budget Logging:** **DONE**
    -   Remove the cost-change-based logging from the `evaluate_all` main loop.
    -   Ensure the single "budget reached" message remains.
    -   Rely on the periodic logger for continuous budget status updates.
4.  **Implement Tabulated Standings:** **DONE**
    -   Modify `Arena.log_standings` to use `tabulate`.
    -   Modify `main.print_results` to use `tabulate` for standings and skill comparison tables.
5.  **Implement Contextual Match Logging:** **DONE**
    -   Ask the user to add `bgbench/match/runner.py` to the chat. (Done)
    -   Modify `MatchRunner.__init__` to accept/create a context-aware logger (e.g., using `LoggerAdapter`). -> *Used simple message prepending instead.*
    -   Update logging calls within `MatchRunner` to use the contextual logger. (Done)
6.  **Testing:** **Partially Done** Manually run an experiment to verify the new logging format, frequency, and context. Check edge cases like experiment start, budget exceeded, and shutdown. -> *Identified and fixed exit condition, ongoing match reporting, and final results format.*
7.  **Documentation:** Update any relevant documentation regarding experiment monitoring or logging output. **(Pending)**
