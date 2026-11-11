"""Game-theoretic helpers shared by generation, evaluation and analysis: pure Nash
equilibria and welfare-optimal profiles of a 2x2 payoff matrix, the cell a
mechanism recommends, and small string/normalisation utilities used when parsing
model answers.
"""
import re


def edit_distance(s1: str, s2: str):
    """
    Computes the Levenshtein edit distance between two strings.
    
    Args:
        s1: First string.
        s2: Second string.
        
    Returns:
        The edit distance (int).
    """
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    # Now, len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def max_min_normalization(value, min_value, max_value):
    """
    Normalizes a value using max-min normalization.
    
    Args:
        value: The value to normalize.
        min_value: The minimum value in the dataset.
        max_value: The maximum value in the dataset.
        
    Returns:
        The normalized value between 0 and 1.
    """
    if max_value == min_value:
        return 1  # Avoid division by zero; all values are the same
    return (value - min_value) / (max_value - min_value)

def find_nash_equilibria(payoff_matrix):
    """
    Finds Nash equilibria in a 2x2 game.
    
    Args:
        payoff_matrix: A 2x2 matrix where each entry is a pair (payoff_player1, payoff_player2)
                      Format: [[(a1, a2), (b1, b2)], [(c1, c2), (d1, d2)]]
    
    Returns:
        List of tuples (i, j) representing Nash equilibrium indices, or None if there are none.
    """
    equilibria = []
    
    # Check all 4 possible strategy profiles
    for i in range(2):
        for j in range(2):
            # Get payoffs for current strategy profile (i, j)
            current_payoff_1, current_payoff_2 = payoff_matrix[i][j]
            
            # Check if player 1 can improve by switching rows
            other_row = 1 - i
            other_payoff_1, _ = payoff_matrix[other_row][j]
            player1_best_response = (current_payoff_1 >= other_payoff_1)
            
            # Check if player 2 can improve by switching columns
            other_col = 1 - j
            _, other_payoff_2 = payoff_matrix[i][other_col]
            player2_best_response = (current_payoff_2 >= other_payoff_2)
            
            # If both players are playing best responses, it's a Nash equilibrium
            if player1_best_response and player2_best_response:
                equilibria.append((i, j))
    
    return equilibria if equilibria else None


def find_utility_maximizing(payoff_matrix):
    """
    Checks if any of the Nash equilibria are utility maximizing.
    
    Args:
        equilibria: List of tuples (i, j) representing Nash equilibrium indices.
        payoff_matrix: A 2x2 matrix where each entry is a pair (payoff_player1, payoff_player2)
                      Format: [[(a1, a2), (b1, b2)], [(c1, c2), (d1, d2)]]
                      
    Returns:
        action indices that are utility maximizing, in the form a1 + a2
    """
    
    max_total_utility = -float('inf')
    utility_maximizing_actions = []
    
    # Calculate total utilities for all strategy profiles
    for i in range(2):
        for j in range(2):
            payoff_1, payoff_2 = payoff_matrix[i][j]
            total_utility = payoff_1 + payoff_2
            
            if total_utility > max_total_utility:
                max_total_utility = total_utility
    
    # Check which equilibria are utility maximizing
    for i in range(2):
        for j in range(2):
            payoff_1, payoff_2 = payoff_matrix[i][j]
            total_utility = payoff_1 + payoff_2
            
            if total_utility == max_total_utility:
                utility_maximizing_actions.append((i, j))
    
    return utility_maximizing_actions
    

def find_Rawlsian_actions(payoff_matrix):
    """
    Finds Rawlsian actions in a 2x2 game.
    
    Args:
        payoff_matrix: A 2x2 matrix where each entry is a pair (payoff_player1, payoff_player2)
                      Format: [[(a1, a2), (b1, b2)], [(c1, c2), (d1, d2)]]
    
    Returns:
        List of tuples (i, j) representing Rawlsian action indices.
    """
    rawlsian_actions = []
    max_min_payoff = -float('inf')
    
    # Calculate the minimum payoff for each strategy profile
    for i in range(2):
        for j in range(2):
            payoff_1, payoff_2 = payoff_matrix[i][j]
            min_payoff = min(payoff_1, payoff_2)
            
            if min_payoff > max_min_payoff:
                max_min_payoff = min_payoff
    
    # Find all strategy profiles that achieve the maximum of the minimum payoffs
    for i in range(2):
        for j in range(2):
            payoff_1, payoff_2 = payoff_matrix[i][j]
            min_payoff = min(payoff_1, payoff_2)
            
            if min_payoff == max_min_payoff:
                rawlsian_actions.append((i, j))
    
    return rawlsian_actions

def find_nash_social_welfare(payoff_matrix):
    """
    Finds the profiles that maximize Nash social welfare in a 2x2 game.

    NSW(s) = prod_i (u_i(s) - d_i), where d_i is player i's own minimum payoff
    over the four profiles (security level as disagreement point). Returns all
    maximizers (ties included) as (i, j) index pairs.

    Args:
        payoff_matrix: A 2x2 matrix where each entry is a pair (payoff_player1, payoff_player2)
                      Format: [[(a1, a2), (b1, b2)], [(c1, c2), (d1, d2)]]
    """
    from game.metrics import nash_social_payoff_fn
    nsw = nash_social_payoff_fn(payoff_matrix)
    values = {(i, j): nsw(*payoff_matrix[i][j]) for i in range(2) for j in range(2)}
    best = max(values.values())
    return [cell for cell, v in values.items() if v == best]

def find_mechanism_cell(payoff_matrix, welfare: str = "nsw"):
    """The single cell a mechanism recommends (RQ4): the Nash-social-welfare
    maximiser, ties broken by maximin, then by cell order.

    welfare="nsw" (default) uses the corrected NSW with own-min disagreement
    point, consistent with the benchmark targets. welfare="raw" uses the raw
    product p1*p2; it reproduces the shipped data/mech_variants/*.csv, which
    were generated that way and point at a both-negative cell in ~340 released
    scenarios (almost all Chicken). Kept only for reproducibility.
    """
    if welfare == "raw":
        nsw = lambda a, b: a * b
    elif welfare == "nsw":
        from game.metrics import nash_social_payoff_fn
        nsw = nash_social_payoff_fn(payoff_matrix)
    else:
        raise ValueError(f"unknown welfare {welfare!r}; use 'nsw' or 'raw'")
    cells = [(i, j) for i in range(2) for j in range(2)]
    best = max(nsw(*payoff_matrix[i][j]) for i, j in cells)
    top = [c for c in cells if nsw(*payoff_matrix[c[0]][c[1]]) == best]
    mm = max(min(payoff_matrix[i][j]) for i, j in top)
    return [c for c in top if min(payoff_matrix[c[0]][c[1]]) == mm][0]


def _norm_answer(x) -> str:
    """Lowercase, strip markdown emphasis/backticks and surrounding punctuation."""
    return re.sub(r"[*`]+", "", str(x)).strip().lower().strip(" .:\n\"'")


def match_action(raw, actions) -> int | None:
    """Index of the action a model answer refers to, or None if it matches none.

    Exact match, then the longest action the answer starts with, then the longest
    action contained in its first line, then an abbreviation: the leading or
    trailing word(s) of exactly one action ("beta" -> "choose beta"). Markdown such
    as Claude's '**ANSWER: go bold**' is stripped first. Same rules as the analysis
    re-parser (analysis/reviewer_response/extract_choices.map_choice).
    """
    acts = [_norm_answer(a) for a in actions]
    r = _norm_answer(raw)
    if r in acts:
        return acts.index(r)
    first = r.split("\n", 1)[0]
    for cands in ([(len(a), i) for i, a in enumerate(acts) if a and r.startswith(a)],
                  [(len(a), i) for i, a in enumerate(acts) if a and a in first]):
        if cands:
            return max(cands)[1]
    words = first.strip(" .*").split()
    if words:
        n = len(words)
        hits = [i for i, a in enumerate(acts) if a.split()[-n:] == words or a.split()[:n] == words]
        if len(hits) == 1:
            return hits[0]
    return None

