"""Tests for game-aware action normalization."""

from __future__ import annotations

from eval.plots.data.processor import (
    DIAGONAL_NASH_GAMES,
    SKIP_NORMALIZATION_GAMES,
    UTILITARIAN_NORMALIZED_GAMES,
    compute_normalization_swaps,
    standardize_action,
)


class TestComputeNormalizationSwaps:
    """Tests for compute_normalization_swaps function."""

    # ==========================================================================
    # Utilitarian-normalized games (Prisoner's Dilemma, Chicken, Stag Hunt, No Conflict)
    # Best utilitarian outcome should be at (0,0) = UP-LEFT
    # ==========================================================================

    def test_prisoners_dilemma_already_normalized(self):
        """Prisoner's Dilemma with best outcome already at (0,0)."""
        # Standard PD: Cooperate-Cooperate (0,0) gives (3,3), best utilitarian = 6
        # Defect-Defect (1,1) gives (1,1), worst = 2
        rewards_matrix = [
            [(3, 3), (0, 5)],  # Row 0: CC, CD
            [(5, 0), (1, 1)],  # Row 1: DC, DD
        ]
        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Prisoner's Dilemma")

        assert swap_rows is False
        assert swap_cols is False

    def test_prisoners_dilemma_needs_row_swap(self):
        """Prisoner's Dilemma with best diagonal at (1,1) - needs both swaps."""
        # Diagonal (0,0) = 1+1 = 2
        # Diagonal (1,1) = 3+3 = 6 (best cooperative outcome)
        # Off-diagonal outcomes are asymmetric and not relevant for normalization
        rewards_matrix = [
            [(1, 1), (0, 5)],  # Row 0: DD, DC
            [(3, 3), (5, 0)],  # Row 1: CD, CC (best diagonal at 1,1)
        ]
        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Prisoner's Dilemma")

        # Need both swaps to move best diagonal from (1,1) to (0,0)
        assert swap_rows is True
        assert swap_cols is True

    def test_prisoners_dilemma_needs_col_swap(self):
        """Prisoner's Dilemma with best diagonal at (0,1) is impossible - this tests (1,1)."""
        # Diagonal (0,0) = 0+5 = 5
        # Diagonal (1,1) = 5+0 = 5 (equal, but we use >= so picks (0,0))
        # Actually this doesn't test col swap. Let me make a better test:
        # Diagonal (0,0) = 2+2 = 4
        # Diagonal (1,1) = 5+5 = 10 (best)
        rewards_matrix = [
            [(2, 2), (0, 5)],  # Row 0: CC, CD
            [(5, 0), (5, 5)],  # Row 1: DC, DD (best diagonal at 1,1)
        ]
        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Prisoner's Dilemma")

        # Need both swaps to move best diagonal from (1,1) to (0,0)
        assert swap_rows is True
        assert swap_cols is True

    def test_prisoners_dilemma_needs_both_swaps(self):
        """Prisoner's Dilemma with best outcome at (1,1) - needs both swaps."""
        # Best utilitarian at (1,1)
        rewards_matrix = [
            [(1, 1), (5, 0)],  # Row 0
            [(0, 5), (3, 3)],  # Row 1: best at (1,1)
        ]
        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Prisoner's Dilemma")

        assert swap_rows is True
        assert swap_cols is True

    def test_chicken_normalization(self):
        """Chicken game with best outcome at wrong position."""
        # Standard Chicken: Both swerve (cooperation) = best outcome
        # Best at (1,1), needs both swaps
        rewards_matrix = [
            [(0, 0), (-1, 1)],   # Row 0: crash, player 1 chickens
            [(1, -1), (2, 2)],   # Row 1: player 2 chickens, both swerve (best)
        ]
        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Chicken")

        assert swap_rows is True
        assert swap_cols is True

    def test_chicken_real_case_cover_intent(self):
        """Chicken game from real data - 'cover intent' should map to UP/LEFT.

        This is a regression test for the bug where cooperative outcomes
        were not correctly normalized. The matrix has:
        - (0,0) = (-8, -8): Both crash, worst diagonal
        - (1,1) = (5, 5): Both cooperate, best diagonal

        We should compare only diagonal cells and swap to move (1,1) to (0,0).
        """
        rewards_matrix = [
            [(-8, -8), (8, 3)],   # Row 0: crash-crash, row cooperates
            [(3, 8), (5, 5)],      # Row 1: col cooperates, both cooperate (best)
        ]
        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Chicken")

        # Diagonal comparison:
        # diag_00 = -8 + -8 = -16
        # diag_11 = 5 + 5 = 10
        # Best is at (1,1), so need both swaps to move it to (0,0)
        assert swap_rows is True, "Row swap needed to move best diagonal from (1,1) to (0,0)"
        assert swap_cols is True, "Col swap needed to move best diagonal from (1,1) to (0,0)"

    def test_stag_hunt_normalization(self):
        """Stag Hunt with best diagonal outcome at (1,1)."""
        # Diagonal (0,0) = 1+1 = 2 (both hunt hare - safe but suboptimal)
        # Diagonal (1,1) = 4+4 = 8 (both hunt stag - best cooperative outcome)
        rewards_matrix = [
            [(1, 1), (0, 3)],  # Row 0: both hare, mixed
            [(3, 0), (4, 4)],  # Row 1: mixed, both stag (best at 1,1)
        ]
        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Stag hunt")

        # Need both swaps to move best diagonal from (1,1) to (0,0)
        assert swap_rows is True
        assert swap_cols is True

    def test_no_conflict_normalization(self):
        """No Conflict game - best diagonal outcome should be moved to (0,0)."""
        # Diagonal (0,0) = 2+2 = 4
        # Diagonal (1,1) = 3+3 = 6 (best cooperative outcome)
        rewards_matrix = [
            [(2, 2), (1, 1)],  # Row 0: option A, mixed
            [(1, 1), (3, 3)],  # Row 1: mixed, option B (best at 1,1)
        ]
        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "No conflict")

        # Need both swaps to move best diagonal from (1,1) to (0,0)
        assert swap_rows is True
        assert swap_cols is True

    # ==========================================================================
    # Diagonal Nash games (Coordination, Bach or Stravinski)
    # Nash equilibria should be on main diagonal (0,0) and (1,1)
    # ==========================================================================

    def test_coordination_already_normalized(self):
        """Coordination game with Nash equilibria already on main diagonal."""
        # Nash equilibria at (0,0) and (1,1)
        rewards_matrix = [
            [(2, 2), (0, 0)],  # (0,0) is Nash
            [(0, 0), (1, 1)],  # (1,1) is Nash
        ]
        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Coordination")

        assert swap_rows is False
        assert swap_cols is False

    def test_coordination_needs_swap(self):
        """Coordination game with Nash equilibria on anti-diagonal."""
        # Nash equilibria at (0,1) and (1,0) - anti-diagonal
        rewards_matrix = [
            [(0, 0), (2, 2)],  # (0,1) is Nash
            [(1, 1), (0, 0)],  # (1,0) is Nash
        ]
        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Coordination")

        # Should swap cols to move Nash to main diagonal
        assert swap_rows is False
        assert swap_cols is True

    def test_bach_or_stravinski_already_normalized(self):
        """Bach or Stravinski with Nash equilibria on main diagonal."""
        # BoS: Both prefer to coordinate, but disagree on which
        # Nash at (0,0) and (1,1)
        rewards_matrix = [
            [(3, 2), (0, 0)],  # (0,0) is Nash (Bach)
            [(0, 0), (2, 3)],  # (1,1) is Nash (Stravinsky)
        ]
        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Bach or Stravinski")

        assert swap_rows is False
        assert swap_cols is False

    def test_bach_or_stravinski_needs_swap(self):
        """Bach or Stravinski with Nash equilibria on anti-diagonal."""
        # Nash at (0,1) and (1,0)
        rewards_matrix = [
            [(0, 0), (3, 2)],  # (0,1) is Nash
            [(2, 3), (0, 0)],  # (1,0) is Nash
        ]
        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Bach or Stravinski")

        assert swap_rows is False
        assert swap_cols is True

    # ==========================================================================
    # Skip normalization games (Matching Pennies)
    # ==========================================================================

    def test_matching_pennies_no_normalization(self):
        """Matching Pennies should not be normalized."""
        rewards_matrix = [
            [(1, -1), (-1, 1)],
            [(-1, 1), (1, -1)],
        ]
        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Matching pennies")

        assert swap_rows is False
        assert swap_cols is False

    def test_matching_pennies_case_insensitive(self):
        """Matching Pennies detection should work case-insensitively."""
        rewards_matrix = [
            [(1, -1), (-1, 1)],
            [(-1, 1), (1, -1)],
        ]
        # Various case variations
        for game_type in ["Matching Pennies", "matching pennies", "MATCHING PENNIES"]:
            swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, game_type)
            assert swap_rows is False
            assert swap_cols is False

    # ==========================================================================
    # Edge cases
    # ==========================================================================

    def test_none_rewards_matrix(self):
        """Handle None rewards matrix gracefully."""
        swap_rows, swap_cols = compute_normalization_swaps(None, "Chicken")

        assert swap_rows is False
        assert swap_cols is False

    def test_invalid_matrix_shape(self):
        """Handle invalid matrix shapes."""
        # 1x2 matrix
        rewards_matrix = [[(1, 1), (2, 2)]]
        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Chicken")

        assert swap_rows is False
        assert swap_cols is False

    def test_unknown_game_type(self):
        """Unknown game types should not be normalized."""
        rewards_matrix = [
            [(3, 3), (0, 5)],
            [(5, 0), (1, 1)],
        ]
        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Unknown Game")

        assert swap_rows is False
        assert swap_cols is False


class TestStandardizeActionWithSwap:
    """Tests for standardize_action with swap parameter."""

    def test_no_swap_row(self):
        """Test row standardization without swap."""
        actions = ["Cooperate", "Defect"]

        assert standardize_action("Cooperate", actions, is_row=True, swap=False) == "UP"
        assert standardize_action("Defect", actions, is_row=True, swap=False) == "DOWN"

    def test_swap_row(self):
        """Test row standardization with swap."""
        actions = ["Cooperate", "Defect"]

        # With swap, mapping is inverted
        assert standardize_action("Cooperate", actions, is_row=True, swap=True) == "DOWN"
        assert standardize_action("Defect", actions, is_row=True, swap=True) == "UP"

    def test_no_swap_col(self):
        """Test column standardization without swap."""
        actions = ["Left", "Right"]

        assert standardize_action("Left", actions, is_row=False, swap=False) == "LEFT"
        assert standardize_action("Right", actions, is_row=False, swap=False) == "RIGHT"

    def test_swap_col(self):
        """Test column standardization with swap."""
        actions = ["Left", "Right"]

        # With swap, mapping is inverted
        assert standardize_action("Left", actions, is_row=False, swap=True) == "RIGHT"
        assert standardize_action("Right", actions, is_row=False, swap=True) == "LEFT"

    def test_case_insensitive(self):
        """Test that action matching is case insensitive."""
        actions = ["Cooperate", "Defect"]

        assert standardize_action("cooperate", actions, is_row=True, swap=False) == "UP"
        assert standardize_action("DEFECT", actions, is_row=True, swap=False) == "DOWN"

    def test_action_not_found(self):
        """Test handling of unknown actions."""
        actions = ["Cooperate", "Defect"]

        assert standardize_action("Unknown", actions, is_row=True, swap=False) is None

    def test_none_action(self):
        """Test handling of None action."""
        actions = ["Cooperate", "Defect"]

        assert standardize_action(None, actions, is_row=True, swap=False) is None

    def test_empty_action_list(self):
        """Test handling of empty action list."""
        assert standardize_action("Cooperate", [], is_row=True, swap=False) is None


class TestNormalizationIntegration:
    """Integration tests for full normalization pipeline."""

    def test_prisoners_dilemma_full_normalization(self):
        """Test full normalization for Prisoner's Dilemma."""
        # PD with best outcome at (1,1) instead of (0,0)
        rewards_matrix = [
            [(1, 1), (5, 0)],  # Row 0: DD, DC
            [(0, 5), (3, 3)],  # Row 1: CD, CC (best at 1,1)
        ]
        actions_row = ["Defect", "Cooperate"]
        actions_col = ["Defect", "Cooperate"]

        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Prisoner's Dilemma")

        # Both swaps needed
        assert swap_rows is True
        assert swap_cols is True

        # Cooperate should become UP (because swap=True inverts the mapping)
        # Original: Cooperate is idx=1 -> DOWN
        # With swap: idx=1 becomes 0 -> UP
        std_row = standardize_action("Cooperate", actions_row, is_row=True, swap=swap_rows)
        std_col = standardize_action("Cooperate", actions_col, is_row=False, swap=swap_cols)

        assert std_row == "UP"
        assert std_col == "LEFT"

        # After normalization, Cooperate-Cooperate should be at UP-LEFT
        # which corresponds to the best utilitarian outcome

    def test_chicken_full_normalization(self):
        """Test full normalization for Chicken game."""
        # Chicken with both swerve (best) at (0,0)
        rewards_matrix = [
            [(2, 2), (1, -1)],  # Row 0: both swerve (best), player 1 swerves
            [(-1, 1), (0, 0)],  # Row 1: player 2 swerves, both straight
        ]
        actions_row = ["Swerve", "Straight"]
        actions_col = ["Swerve", "Straight"]

        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Chicken")

        # No swaps needed - best is already at (0,0)
        assert swap_rows is False
        assert swap_cols is False

    def test_coordination_full_normalization(self):
        """Test full normalization for Coordination game."""
        # Coordination with Nash on anti-diagonal
        rewards_matrix = [
            [(0, 0), (2, 2)],  # Nash at (0,1)
            [(1, 1), (0, 0)],  # Nash at (1,0)
        ]
        actions_row = ["A", "B"]
        actions_col = ["X", "Y"]

        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Coordination")

        # Col swap needed to move Nash to main diagonal
        assert swap_rows is False
        assert swap_cols is True

        # After swap, choosing A-Y should map to UP-LEFT (Nash at 0,0)
        std_row = standardize_action("A", actions_row, is_row=True, swap=swap_rows)
        std_col = standardize_action("Y", actions_col, is_row=False, swap=swap_cols)

        # A is idx=0, no row swap -> UP
        # Y is idx=1, col swap -> idx becomes 0 -> LEFT
        assert std_row == "UP"
        assert std_col == "LEFT"

    def test_chicken_real_case_full_pipeline(self):
        """Test full normalization pipeline for the 'cover intent' Chicken case.

        This is a regression test for the bug where 'cover intent' (index 1)
        was incorrectly normalized to DOWN/LEFT instead of UP/LEFT.
        """
        rewards_matrix = [
            [(-8, -8), (8, 3)],
            [(3, 8), (5, 5)],
        ]
        actions_row = ["not cover intent", "cover intent"]
        actions_col = ["not cover intent", "cover intent"]

        swap_rows, swap_cols = compute_normalization_swaps(rewards_matrix, "Chicken")

        # Both swaps needed
        assert swap_rows is True
        assert swap_cols is True

        # 'cover intent' is at index 1 (the cooperative action)
        # With swap=True, index 1 should become index 0, mapping to UP/LEFT
        std_row = standardize_action("cover intent", actions_row, is_row=True, swap=swap_rows)
        std_col = standardize_action("cover intent", actions_col, is_row=False, swap=swap_cols)

        assert std_row == "UP", "Cooperative action should map to UP after swap"
        assert std_col == "LEFT", "Cooperative action should map to LEFT after swap"

        # Verify the non-cooperative action maps to DOWN/RIGHT
        std_row_defect = standardize_action("not cover intent", actions_row, is_row=True, swap=swap_rows)
        std_col_defect = standardize_action("not cover intent", actions_col, is_row=False, swap=swap_cols)

        assert std_row_defect == "DOWN"
        assert std_col_defect == "RIGHT"


class TestAllUtilitarianGamesHaveCorrectCategory:
    """Verify all expected games are in the correct category."""

    def test_utilitarian_games_list(self):
        """Verify utilitarian-normalized games."""
        expected = {"Prisoner's Dilemma", "Chicken", "Stag hunt", "No conflict"}
        assert UTILITARIAN_NORMALIZED_GAMES == expected

    def test_diagonal_nash_games_list(self):
        """Verify diagonal-Nash games."""
        expected = {"Coordination", "Bach or Stravinski"}
        assert DIAGONAL_NASH_GAMES == expected

    def test_skip_normalization_games_list(self):
        """Verify skip-normalization games."""
        expected = {"Matching pennies"}
        assert SKIP_NORMALIZATION_GAMES == expected
