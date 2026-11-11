"""Tests for action probability grid functions."""

from __future__ import annotations

import numpy as np
import pytest

from eval.plots.data.schemas import EvalRunData, MultiEvalData, SampleRecord
from eval.plots.renderers.heatmap import (
    collect_model_game_actions,
    compute_probability_matrix,
)


class TestComputeProbabilityMatrix:
    """Tests for compute_probability_matrix function."""

    def test_basic_probability_calculation(self):
        """Test basic probability calculation with known values."""
        action_pairs = [
            ("UP", "LEFT"),
            ("UP", "LEFT"),
            ("UP", "LEFT"),
            ("UP", "RIGHT"),
            ("DOWN", "LEFT"),
            ("DOWN", "LEFT"),
            ("DOWN", "RIGHT"),
            ("DOWN", "RIGHT"),
            ("DOWN", "RIGHT"),
            ("UP", "LEFT"),
        ]

        matrix, total = compute_probability_matrix(action_pairs)

        # Expected counts: UP-LEFT=4, UP-RIGHT=1, DOWN-LEFT=2, DOWN-RIGHT=3
        assert total == 10
        assert abs(matrix[0, 0] - 0.40) < 0.001, f"UP-LEFT wrong: {matrix[0, 0]}"
        assert abs(matrix[0, 1] - 0.10) < 0.001, f"UP-RIGHT wrong: {matrix[0, 1]}"
        assert abs(matrix[1, 0] - 0.20) < 0.001, f"DOWN-LEFT wrong: {matrix[1, 0]}"
        assert abs(matrix[1, 1] - 0.30) < 0.001, f"DOWN-RIGHT wrong: {matrix[1, 1]}"

    def test_probabilities_sum_to_one(self):
        """Test that probabilities always sum to 1.0."""
        action_pairs = [
            ("UP", "LEFT"),
            ("UP", "RIGHT"),
            ("DOWN", "LEFT"),
            ("DOWN", "RIGHT"),
            ("UP", "LEFT"),
            ("DOWN", "RIGHT"),
        ]

        matrix, total = compute_probability_matrix(action_pairs)

        assert total == 6
        assert abs(matrix.sum() - 1.0) < 0.001, f"Sum not 1.0: {matrix.sum()}"

    def test_empty_input(self):
        """Test handling of empty input."""
        matrix, total = compute_probability_matrix([])

        assert total == 0
        assert matrix.shape == (2, 2)
        assert matrix.sum() == 0.0

    def test_single_action_pair(self):
        """Test with a single action pair."""
        action_pairs = [("UP", "LEFT")]

        matrix, total = compute_probability_matrix(action_pairs)

        assert total == 1
        assert matrix[0, 0] == 1.0
        assert matrix[0, 1] == 0.0
        assert matrix[1, 0] == 0.0
        assert matrix[1, 1] == 0.0

    def test_all_same_action(self):
        """Test when all actions are the same."""
        action_pairs = [("DOWN", "RIGHT")] * 100

        matrix, total = compute_probability_matrix(action_pairs)

        assert total == 100
        assert matrix[0, 0] == 0.0
        assert matrix[0, 1] == 0.0
        assert matrix[1, 0] == 0.0
        assert matrix[1, 1] == 1.0

    def test_uniform_distribution(self):
        """Test with uniform distribution (25% each)."""
        action_pairs = [
            ("UP", "LEFT"),
            ("UP", "RIGHT"),
            ("DOWN", "LEFT"),
            ("DOWN", "RIGHT"),
        ]

        matrix, total = compute_probability_matrix(action_pairs)

        assert total == 4
        assert abs(matrix[0, 0] - 0.25) < 0.001
        assert abs(matrix[0, 1] - 0.25) < 0.001
        assert abs(matrix[1, 0] - 0.25) < 0.001
        assert abs(matrix[1, 1] - 0.25) < 0.001

    def test_invalid_actions_ignored(self):
        """Test that invalid action values are ignored."""
        action_pairs = [
            ("UP", "LEFT"),
            ("UP", "LEFT"),
            ("INVALID", "LEFT"),  # Should be ignored
            ("UP", "INVALID"),  # Should be ignored
            ("INVALID", "INVALID"),  # Should be ignored
        ]

        matrix, total = compute_probability_matrix(action_pairs)

        # Only 2 valid pairs
        assert total == 2
        assert matrix[0, 0] == 1.0
        assert matrix.sum() == 1.0

    def test_matrix_layout(self):
        """Test that matrix layout matches expected positions.

        Expected layout:
            [0,0] = P(UP, LEFT)     [0,1] = P(UP, RIGHT)
            [1,0] = P(DOWN, LEFT)   [1,1] = P(DOWN, RIGHT)
        """
        # Test each position individually
        test_cases = [
            (("UP", "LEFT"), (0, 0)),
            (("UP", "RIGHT"), (0, 1)),
            (("DOWN", "LEFT"), (1, 0)),
            (("DOWN", "RIGHT"), (1, 1)),
        ]

        for action_pair, expected_pos in test_cases:
            matrix, total = compute_probability_matrix([action_pair])
            assert total == 1
            assert matrix[expected_pos] == 1.0, f"Wrong position for {action_pair}"
            # All other positions should be 0
            for i in range(2):
                for j in range(2):
                    if (i, j) != expected_pos:
                        assert matrix[i, j] == 0.0

    def test_large_sample_size(self):
        """Test with a large number of samples."""
        np.random.seed(42)
        n_samples = 10000

        # Generate random action pairs
        row_actions = np.random.choice(["UP", "DOWN"], size=n_samples)
        col_actions = np.random.choice(["LEFT", "RIGHT"], size=n_samples)
        action_pairs = list(zip(row_actions, col_actions))

        matrix, total = compute_probability_matrix(action_pairs)

        assert total == n_samples
        assert abs(matrix.sum() - 1.0) < 0.001
        # With random sampling, each cell should be roughly 0.25
        for i in range(2):
            for j in range(2):
                assert 0.20 < matrix[i, j] < 0.30, f"Cell [{i},{j}] out of expected range"


class TestCollectModelGameActions:
    """Tests for collect_model_game_actions function."""

    def _create_sample_record(
        self,
        sample_id: str,
        model_name: str,
        game_type: str,
        std_row: str,
        std_col: str,
    ) -> SampleRecord:
        """Helper to create a SampleRecord."""
        return SampleRecord(
            sample_id=sample_id,
            model_name=model_name,
            game_type=game_type,
            std_row=std_row,
            std_col=std_col,
            parse_success=True,
        )

    def _create_eval_run(
        self,
        model_name: str,
        samples: list[SampleRecord],
    ) -> EvalRunData:
        """Helper to create an EvalRunData."""
        return EvalRunData(
            log_id="test_log",
            log_path="/test/path",
            model_name=model_name,
            num_samples=len(samples),
            samples=samples,
        )

    def test_single_model_single_game(self):
        """Test collecting actions from a single model and game type."""
        samples = [
            self._create_sample_record("1", "model_a", "Chicken", "UP", "LEFT"),
            self._create_sample_record("2", "model_a", "Chicken", "DOWN", "RIGHT"),
        ]
        run = self._create_eval_run("model_a", samples)
        multi_data = MultiEvalData(runs=[run])

        result = collect_model_game_actions(multi_data)

        assert len(result) == 1
        assert ("model_a", "Chicken") in result
        assert result[("model_a", "Chicken")] == [("UP", "LEFT"), ("DOWN", "RIGHT")]

    def test_multiple_models(self):
        """Test collecting actions from multiple models."""
        samples_a = [
            self._create_sample_record("1", "model_a", "Chicken", "UP", "LEFT"),
        ]
        samples_b = [
            self._create_sample_record("2", "model_b", "Chicken", "DOWN", "RIGHT"),
        ]
        run_a = self._create_eval_run("model_a", samples_a)
        run_b = self._create_eval_run("model_b", samples_b)
        multi_data = MultiEvalData(runs=[run_a, run_b])

        result = collect_model_game_actions(multi_data)

        assert len(result) == 2
        assert ("model_a", "Chicken") in result
        assert ("model_b", "Chicken") in result

    def test_multiple_game_types(self):
        """Test collecting actions from multiple game types."""
        samples = [
            self._create_sample_record("1", "model_a", "Chicken", "UP", "LEFT"),
            self._create_sample_record("2", "model_a", "Stag hunt", "DOWN", "RIGHT"),
        ]
        run = self._create_eval_run("model_a", samples)
        multi_data = MultiEvalData(runs=[run])

        result = collect_model_game_actions(multi_data)

        assert len(result) == 2
        assert ("model_a", "Chicken") in result
        assert ("model_a", "Stag hunt") in result

    def test_empty_multi_data(self):
        """Test with empty MultiEvalData."""
        multi_data = MultiEvalData(runs=[])

        result = collect_model_game_actions(multi_data)

        assert result == {}

    def test_skips_invalid_model_names(self):
        """Test that invalid model names are skipped."""
        samples = [
            self._create_sample_record("1", "?", "Chicken", "UP", "LEFT"),
        ]
        run = self._create_eval_run("?", samples)
        multi_data = MultiEvalData(runs=[run])

        result = collect_model_game_actions(multi_data)

        assert result == {}

    def test_skips_samples_without_std_actions(self):
        """Test that samples without standardized actions are skipped."""
        sample_valid = self._create_sample_record("1", "model_a", "Chicken", "UP", "LEFT")
        sample_invalid = SampleRecord(
            sample_id="2",
            model_name="model_a",
            game_type="Chicken",
            std_row=None,  # Invalid
            std_col="LEFT",
            parse_success=True,
        )
        run = self._create_eval_run("model_a", [sample_valid, sample_invalid])
        multi_data = MultiEvalData(runs=[run])

        result = collect_model_game_actions(multi_data)

        assert len(result[("model_a", "Chicken")]) == 1

    def test_merges_multiple_runs_same_model(self):
        """Test that multiple runs for the same model are merged."""
        samples_1 = [
            self._create_sample_record("1", "model_a", "Chicken", "UP", "LEFT"),
        ]
        samples_2 = [
            self._create_sample_record("2", "model_a", "Chicken", "DOWN", "RIGHT"),
        ]
        run_1 = self._create_eval_run("model_a", samples_1)
        run_2 = self._create_eval_run("model_a", samples_2)
        multi_data = MultiEvalData(runs=[run_1, run_2])

        result = collect_model_game_actions(multi_data)

        assert len(result) == 1
        assert len(result[("model_a", "Chicken")]) == 2


class TestProbabilitySumInvariant:
    """Property-based tests for probability sum invariant."""

    @pytest.mark.parametrize("n_samples", [1, 10, 100, 1000])
    def test_probabilities_sum_to_one_various_sizes(self, n_samples):
        """Test that probabilities sum to 1.0 for various sample sizes."""
        np.random.seed(42)
        row_actions = np.random.choice(["UP", "DOWN"], size=n_samples)
        col_actions = np.random.choice(["LEFT", "RIGHT"], size=n_samples)
        action_pairs = list(zip(row_actions, col_actions))

        matrix, total = compute_probability_matrix(action_pairs)

        assert total == n_samples
        assert abs(matrix.sum() - 1.0) < 0.001, f"Sum is {matrix.sum()}, expected 1.0"

    @pytest.mark.parametrize("bias", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_probabilities_sum_to_one_biased_distributions(self, bias):
        """Test probability sum with various biased distributions."""
        np.random.seed(42)
        n_samples = 500

        # Generate biased samples
        action_pairs = []
        for _ in range(n_samples):
            row = "UP" if np.random.random() < bias else "DOWN"
            col = "LEFT" if np.random.random() < bias else "RIGHT"
            action_pairs.append((row, col))

        matrix, total = compute_probability_matrix(action_pairs)

        assert total == n_samples
        assert abs(matrix.sum() - 1.0) < 0.001, f"Sum is {matrix.sum()}, expected 1.0"
