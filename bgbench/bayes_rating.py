"""
Bayesian Elo System Module
--------------------------
This module implements a Bayesian approach to Elo-like rating updates, allowing
for draws and providing a posterior distribution over skill parameters. Each
player's skill is modeled as a latent variable with a prior. Observed match outcomes
are used to update posterior beliefs about each player's skill via MCMC sampling.

Usage:
  1. Collect a list of `GameResult` entries (who played whom and who won or if draw).
  2. Create an `EloSystem` object.
  3. Call `update_ratings(...)` with the match history and the set of player names
     you want ratings for. It returns a dictionary mapping each requested player name
     to a `PlayerRating`.
  4. Optionally, call `probability_stronger(a, b)` to get the posterior probability
     that `a`'s skill is higher than `b`'s.
  5. Use `is_match_needed(a, b)` to see if the model is sufficiently confident about
     which player is stronger, based on `confidence_threshold`.
"""

import math
from typing import List, Dict, Optional, Tuple, Any, cast
from dataclasses import dataclass
import numpy as np

# These are imported only if you use a Bayesian library like PyMC.
# If you don't have PyMC installed, install it via:
#   pip install pymc
import pymc as pm
import arviz as az
from arviz.data.inference_data import InferenceData


@dataclass
class PlayerRating:
    """
    Encapsulates a player's Bayesian skill rating.

    :param name: Name/identifier of the player.
    :param rating: Mean of the player's skill posterior distribution.
                   Interpreted similarly to an Elo rating (e.g., around 1500).
    :param sigma: Standard deviation (uncertainty) of the player's skill posterior.
    :param games_played: Number of games the player has participated in.
    """
    name: str
    rating: float
    sigma: float
    games_played: int


@dataclass
class GameResult:
    """
    Represents a single game between two players.

    :param player_0: Name of the first player.
    :param player_1: Name of the second player.
    :param winner: Name of the winning player, or None if the game was a draw.
    """
    player_0: str
    player_1: str
    winner: Optional[str]


class EloSystem:
    """
    A Bayesian Elo system using a Davidson-style extension to Bradley-Terry
    for handling draws. Each player's skill is a latent variable with a normal prior.
    Match outcomes are modeled with probabilities:

      score_i = 10^(skill_i/400)
      score_j = 10^(skill_j/400)
      denom   = score_i + score_j + c

      P(player_i wins) = score_i / denom
      P(player_j wins) = score_j / denom
      P(draw)          = c        / denom

    where `c` is a latent parameter that represents the "draw propensity." If draws
    are impossible, it can be set to a very small prior center.

    This model is fit via MCMC each time you call `update_ratings()`, returning
    posterior means and standard deviations for each player's skill parameter.
    """

    def __init__(self, confidence_threshold: float = 0.70, draws_parameter_prior: float = 10.0):
        """
        Initialize the EloSystem with a given confidence threshold for comparisons
        and a prior scale for the draw parameter.

        :param confidence_threshold: Probability threshold for deciding whether
                                     one player's skill is likely higher than
                                     another's.
        :param draws_parameter_prior: Prior scale for the latent "draw" parameter c.
                                      Larger values increase the model's prior belief
                                      in frequent draws.
        """
        self.confidence_threshold = confidence_threshold
        self.draws_parameter_prior = draws_parameter_prior
        # We'll store these after each update
        self._trace: Any = None  # Will hold InferenceData when available
        self._player_index_map: Optional[Dict[str, int]] = None
        self._all_players: Optional[List[str]] = None


    def update_ratings(self,
                       history: List[GameResult],
                       player_names: List[str]) -> Dict[str, PlayerRating]:
        """
        Runs an MCMC procedure over all players who have appeared in the provided
        match history, then returns a dictionary mapping each requested player name
        to a `PlayerRating` derived from the posterior distribution.

        :param history: A list of GameResult entries capturing who played whom,
                        and who (if anyone) won.
        :param player_names: The list of players for whom we want final rating results.
        :return: A dictionary mapping each requested player name to a PlayerRating.
        """

        # Check if history is empty
        if not history:
            # Return default values for all requested players
            return {p: PlayerRating(name=p, rating=1500.0, sigma=400.0, games_played=0)
                   for p in player_names}
        
        # 1) Gather all unique players from the match history
        players_set = set()
        for g in history:
            players_set.add(g.player_0)
            players_set.add(g.player_1)
        all_players = sorted(list(players_set))

        # 2) Build index maps for players
        player_index_map = {p: i for i, p in enumerate(all_players)}
        n_players = len(all_players)

        # 3) Convert each game to an integer "outcome":
        #    0 => player_0 wins, 1 => player_1 wins, 2 => draw
        outcome_list = []
        p0_idx_list = []
        p1_idx_list = []

        games_played = {p: 0 for p in all_players}

        for g in history:
            i0 = player_index_map[g.player_0]
            i1 = player_index_map[g.player_1]
            p0_idx_list.append(i0)
            p1_idx_list.append(i1)
            if g.winner is None:
                outcome_list.append(2)  # draw
            elif g.winner == g.player_0:
                outcome_list.append(0)
            else:
                outcome_list.append(1)

            # Count games for each player
            games_played[g.player_0] += 1
            games_played[g.player_1] += 1

        # Ensure we have at least one game
        if not outcome_list:
            # If the history was not empty but we somehow have no outcomes,
            # return default values
            return {p: PlayerRating(name=p, rating=1500.0, sigma=400.0, games_played=0)
                   for p in player_names}
            
        outcome_array = np.array(outcome_list, dtype=int)
        p0_array = np.array(p0_idx_list, dtype=int)
        p1_array = np.array(p1_idx_list, dtype=int)

        # 4) Build a PyMC model
        #    skill[i] ~ Normal(1500, 400) for each player
        #    c ~ Exponential(lambda) for draws
        #    Then for each match:
        #      p0 = 10^(skill[p0]/400)
        #      p1 = 10^(skill[p1]/400)
        #      denom = p0 + p1 + c
        #      P(outcome=0 => p0 wins) = p0 / denom
        #      P(outcome=1 => p1 wins) = p1 / denom
        #      P(outcome=2 => draw)    = c  / denom

        with pm.Model():
            # skill for each player
            skill = pm.Normal("skill", mu=1500.0, sigma=400.0, shape=n_players)

            # draw parameter c
            # Using an exponential prior centered on draws_parameter_prior
            # e.g., if draws_parameter_prior=10 => mean(c)=10
            c = pm.Exponential("c", lam=1.0 / self.draws_parameter_prior)

            # Evaluate logistic-like expressions
            p0_ = 10 ** (skill[p0_array] / 400.0)
            p1_ = 10 ** (skill[p1_array] / 400.0)

            denom = p0_ + p1_ + c

            p_win0 = p0_ / denom
            p_win1 = p1_ / denom
            p_draw = c   / denom

            # Observed outcomes: these are 0,1,2
            # Use proper stacking for PyMC 5.x (replacing pm.stack with pytensor.tensor.stack)
            import pytensor.tensor as pt
            outcome = pm.Categorical(
                "outcome",
                p=pt.stack([p_win0, p_win1, p_draw], axis=1),
                observed=outcome_array
            )

            # 5) Sample from the posterior
            #    Using 4 chains as recommended for robust convergence diagnostics
            trace = pm.sample(draws=1000, tune=1000, chains=4, target_accept=0.9, progressbar=False)

        # Store the trace and indexes for joint queries
        self._trace = trace  # Type already defined in __init__
        self._player_index_map = player_index_map
        self._all_players = all_players

        # 6) Summarize the posterior for each player's skill
        summary = az.summary(trace, var_names=["skill"], round_to=None)
        # 'summary' is a dataframe with columns like mean, sd, hdi_3%, hdi_97%, etc.

        # Prepare result dict
        result: Dict[str, PlayerRating] = {}

        for p in player_names:
            if p not in player_index_map:
                # This player had no matches in the history
                # We'll set them to some default
                result[p] = PlayerRating(name=p, rating=1500.0, sigma=400.0, games_played=0)
            else:
                i = player_index_map[p]
                # The "skill" summary row for skill[i]
                row = summary.iloc[i]
                mean_skill = row["mean"]
                std_skill = row["sd"]
                gp = games_played[p]
                result[p] = PlayerRating(name=p,
                                         rating=float(mean_skill),
                                         sigma=float(std_skill),
                                         games_played=gp)

        return result

    def probability_stronger(self, name_a: str, name_b: str) -> float:
        """
        Computes P(skill_A > skill_B) using the joint posterior samples
        from the most recent MCMC run (stored in self._trace).

        :param name_a: Name of the first player
        :param name_b: Name of the second player
        :return: Probability that A's skill is greater than B's skill under normal assumptions.
        """
        if self._trace is None or self._player_index_map is None:
            # That means we haven't run update_ratings yet
            raise RuntimeError("No trace available. Call update_ratings first.")

        if name_a not in self._player_index_map or name_b not in self._player_index_map:
            # If one didn't appear in the history, you won't have any samples
            # so you can return 0.5 or raise an exception
            return 0.5

        idx_a = self._player_index_map[name_a]
        idx_b = self._player_index_map[name_b]

        # Extract skill samples from the trace; shape = (chain, draw, player_index)
        # Cast to Any to avoid type checking issues, as InferenceData's structure is dynamic
        trace_data = cast(Any, self._trace)
        skill_samples = trace_data.posterior["skill"].values  # xarray data
        # Flatten chain/draw dims into one for convenience
        flat_samples = skill_samples.reshape(-1, skill_samples.shape[-1])

        # skill_A / skill_B are now 1D arrays of the same length
        skill_A = flat_samples[:, idx_a]
        skill_B = flat_samples[:, idx_b]

        # Probability that A is stronger is fraction of draws where skill_A > skill_B
        return float(np.mean(skill_A > skill_B))
        
    def is_match_needed(self, name_a: str, name_b: str) -> bool:
        """
        Determines if more matches between these two players should be played to
        surpass the confidence threshold. If the probability that one is stronger
        than the other is under 'confidence_threshold', we want more data.

        :param name_a: Name of the first player.
        :param name_b: Name of the second player.
        :return: True if an additional match is needed to be more certain who is stronger.
        """
        prob = self.probability_stronger(name_a, name_b)
        return prob < self.confidence_threshold

    def get_credible_intervals(self, player_names: Optional[List[str]] = None) -> Dict[str, Tuple[float, float]]:
        """
        Returns a dictionary of 95% credible intervals for each player's skill, based
        on the joint posterior samples from the most recent MCMC run.

        If `player_names` is None, intervals are computed for all players that appeared
        in the last call to `update_ratings`.

        :param player_names: Optional list of player names to retrieve intervals for.
                            If None, returns intervals for all players in the trace.
        :return: A dictionary mapping player name -> (lower_bound, upper_bound) of
                the 95% credible interval for that player's skill.
        :raises RuntimeError: If no posterior trace is available (i.e., if
                            `update_ratings` hasn't been called yet).
        """
        if self._trace is None or self._player_index_map is None:
            raise RuntimeError("No trace available. Call update_ratings first.")

        if player_names is None:
            player_names = list(self._player_index_map.keys())

        intervals = {}
        # Cast to Any to avoid type checking issues, as InferenceData's structure is dynamic
        trace_data = cast(Any, self._trace)
        skill_posterior = trace_data.posterior["skill"]  # shape = (chain, draw, player_index)

        # Flatten chain and draw dimensions into a single dimension for easy quantile calculation
        # skill_samples.shape will then be (total_samples, n_players)
        skill_samples = skill_posterior.values.reshape(-1, skill_posterior.shape[-1])

        for name in player_names:
            if name not in self._player_index_map:
                # If this player wasn't in the history, there's no posterior for them
                intervals[name] = (None, None)
            else:
                idx = self._player_index_map[name]
                # Extract all samples for this player's skill
                samples = skill_samples[:, idx]
                # Calculate 95% credible interval via quantiles
                lower, upper = np.quantile(samples, [0.025, 0.975])
                intervals[name] = (float(lower), float(upper))

        return intervals