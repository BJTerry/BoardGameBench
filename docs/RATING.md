# Rating System Documentation

The BoardGameBench rating system implements a Bayesian approach to Elo-like ratings that quantifies uncertainty and supports multi-outcome games (wins/draws).

## Key Components

- **PlayerRating**: Encapsulates a player's rating and uncertainty.
  - `name`: Identifier for the player
  - `rating`: Mean of the player's skill posterior (similar to Elo rating)
  - `sigma`: Standard deviation (uncertainty) of the player's skill posterior
  - `games_played`: Number of games the player has participated in

- **GameResult**: Represents a single game outcome.
  - `player_0`: Name of first player
  - `player_1`: Name of second player
  - `winner`: Name of winning player, or `None` for draws

- **EloSystem**: Updates ratings via MCMC sampling.
  - Uses PyMC for Bayesian inference
  - Models player skills as latent variables with priors
  - Draws are explicitly modeled using a draw parameter
  - Posterior distributions provide uncertainty quantification

## Key Methods

- **update_ratings()**: Runs MCMC to update all player ratings based on match history.
  - Returns a dictionary mapping player names to `PlayerRating` objects.

- **probability_stronger(player_a, player_b)**: Estimates probability that player_a's skill is greater than player_b's.
  - Uses posterior samples from the most recent MCMC run.
  - Returns a value between 0 and 1.

- **is_match_needed(player_a, player_b)**: Determines if more matches between players are needed.
  - Based on the `confidence_threshold` (default 0.70).
  - Returns `True` if confidence is below threshold.

## Mathematical Model

The system uses a modified Bradley-Terry model with a draw parameter:

```
score_i = 10^(skill_i/400)
score_j = 10^(skill_j/400)
denom   = score_i + score_j + c

P(player_i wins) = score_i / denom
P(player_j wins) = score_j / denom
P(draw)          = c        / denom
```

Where `c` is a latent parameter representing draw propensity.

## Usage Example

```python
# Create rating system
elo = EloSystem(confidence_threshold=0.70)

# Update ratings based on match history
history = [
    GameResult("model_a", "model_b", "model_a"),
    GameResult("model_a", "model_c", "model_c"),
    GameResult("model_b", "model_c", None)  # Draw
]
ratings = elo.update_ratings(history, ["model_a", "model_b", "model_c"])

# Check if more games are needed between two players
need_more_games = elo.is_match_needed("model_a", "model_b")

# Get probability that model_a is stronger than model_b
prob = elo.probability_stronger("model_a", "model_b")
```

For implementation details, see `bgbench/rating.py`.