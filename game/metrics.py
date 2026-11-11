"""Social welfare functions for 2x2 games: utilitarian (sum), Rawlsian (min) and
Nash social welfare (product of gains over each player's own minimum payoff), plus
helpers to find the best/worst welfare value over a payoff matrix.
"""

from typing import Callable

quadrant = tuple[float, float]  

def utilitarian_payoff(self_payoff: float, other_payoff: float) -> float:
    """Calculate the utilitarian social welfare from payoffs.
    Args:
        self_payoff: The payoff for the self agent.
        other_payoff: The payoff for the other agent.
    Returns:
        The total utilitarian social welfare.
    """
    return self_payoff + other_payoff

def rawlsian_payoff(self_payoff: float, other_payoff: float) -> float:
    """Calculate the Rawlsian social welfare from payoffs (fairness)
    Args:
        self_payoff: The payoff for the self agent.
        other_payoff: The payoff for the other agent.
    Returns:
        The minimum payoff representing Rawlsian social welfare.
    """
    return min(self_payoff, other_payoff)


def player_minima(payoffs) -> tuple[float, float]:
    """Each player's own worst payoff across the four profiles (their security
    level), used as the disagreement point for Nash social welfare."""
    cells = [payoffs[i][j] for i in range(2) for j in range(2)]
    return min(c[0] for c in cells), min(c[1] for c in cells)


def nash_social_payoff(self_payoff: float, other_payoff: float,
                       minima: tuple[float, float]) -> float:
    """Nash social welfare of one profile: prod_i (u_i - d_i), with d_i the
    player's own minimum over the game (see `player_minima`).

    The disagreement point is required on purpose: the raw product u1*u2 is not
    a valid NSW once payoffs can be negative ((-8,-8) would beat (5,5)), and the
    older positive-orthant rule (product if both > 0 else 0) disagrees with the
    targets. Bind it per game with `nash_social_payoff_fn(payoffs)`.
    """
    return (self_payoff - minima[0]) * (other_payoff - minima[1])


def nash_social_payoff_fn(payoffs) -> Callable[[float, float], float]:
    """Two-argument NSW for this game, for use with general_evaluator_*."""
    m = player_minima(payoffs)
    return lambda a, b: nash_social_payoff(a, b, m)

def general_evaluator_max(payoffs: tuple[tuple[quadrant, quadrant], tuple[quadrant, quadrant]],
                      social_welfare_fn: Callable[[float, float], float]) -> float:
    """General evaluator for social welfare functions.
    Args:
        payoffs: A tuple containing the payoffs for two agents in two scenarios.
                 Each agent's payoffs are represented as a tuple of quadrants,
                 where each quadrant is a tuple of (self_payoff, other_payoff).
        social_welfare_fn: A function that computes social welfare given payoffs.
    """
    
    max_utilitarian = float('-inf')
    for i in range(2):
        for j in range(2):
            self_payoff, other_payoff = payoffs[i][j]
            total_payoff = social_welfare_fn(self_payoff, other_payoff)
            if total_payoff > max_utilitarian:
                max_utilitarian = total_payoff
    return max_utilitarian


def general_evaluator_min(payoffs: tuple[tuple[quadrant, quadrant], tuple[quadrant, quadrant]],
                      social_welfare_fn: Callable[[float, float], float]) -> float:
    """General evaluator for social welfare functions using minimum.
    Args:
        payoffs: A tuple containing the payoffs for two agents in two scenarios.
                 Each agent's payoffs are represented as a tuple of quadrants,
                 where each quadrant is a tuple of (self_payoff, other_payoff).
        social_welfare_fn: A function that computes social welfare given payoffs.
    """
    min_utilitarian = float('inf')
    for i in range(2):
        for j in range(2):
            self_payoff, other_payoff = payoffs[i][j]
            total_payoff = social_welfare_fn(self_payoff, other_payoff)
            if total_payoff < min_utilitarian:
                min_utilitarian = total_payoff
    return min_utilitarian


def utilitarian(payoffs: tuple[tuple[quadrant, quadrant], tuple[quadrant, quadrant]]) -> float:
    """Calculate the maximum utilitarian social welfare from payoffs.
    Args:
        payoffs: A tuple containing the payoffs for two agents in two scenarios.
                 Each agent's payoffs are represented as a tuple of quadrants,
                 where each quadrant is a tuple of (self_payoff, other_payoff).
    Returns:
        The maximum utilitarian social welfare.
    """
    return general_evaluator_max(payoffs, utilitarian_payoff)

def rawlsian(payoffs: tuple[tuple[quadrant, quadrant], tuple[quadrant, quadrant]]) -> float:
    """Calculate the maximum Rawlsian social welfare from payoffs.
    Args:
        payoffs: A tuple containing the payoffs for two agents in two scenarios.
                 Each agent's payoffs are represented as a tuple of quadrants,
                 where each quadrant is a tuple of (self_payoff, other_payoff).
    Returns:
        The maximum Rawlsian social welfare.
    """  
    return general_evaluator_max(payoffs, rawlsian_payoff)

fairness = rawlsian # alias for clarity

def nash_social_welfare(payoffs: tuple[tuple[quadrant, quadrant], tuple[quadrant, quadrant]]) -> float:
    """Calculate the maximum Nash social welfare from payoffs.
    Args:
        payoffs: A tuple containing the payoffs for two agents in two scenarios.
                 Each agent's payoffs are represented as a tuple of quadrants,
                 where each quadrant is a tuple of (self_payoff, other_payoff).
    Returns:
        The maximum Nash social welfare.
    """

    return general_evaluator_max(payoffs, nash_social_payoff_fn(payoffs))