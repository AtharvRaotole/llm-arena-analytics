"""
Unit tests for CostOptimizer class.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cost_optimizer import CostOptimizer
from database.db_manager import DatabaseManager


class TestCostOptimizer(unittest.TestCase):
    """Test cases for CostOptimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_db = Mock(spec=DatabaseManager)
        self.optimizer = CostOptimizer(db_manager=self.mock_db)

    def test_calculate_cost(self):
        """Test calculate_cost method."""
        # Mock model lookup
        self.mock_db.get_model_by_name.return_value = {'id': 1, 'name': 'GPT-4', 'provider': 'OpenAI'}
        
        # Mock pricing lookup
        self.mock_db.get_latest_pricing.return_value = [
            {
                'model_id': 1,
                'input_cost_per_token': 0.01,
                'output_cost_per_token': 0.03
            }
        ]

        cost = self.optimizer.calculate_cost('GPT-4', 1000, 500)
        
        # Expected: (1000/1000 * 0.01) + (500/1000 * 0.03) = 0.01 + 0.015 = 0.025
        self.assertAlmostEqual(cost, 0.025, places=6)

    def test_calculate_cost_model_not_found(self):
        """Test calculate_cost with non-existent model."""
        self.mock_db.get_model_by_name.return_value = None

        with self.assertRaises(ValueError) as context:
            self.optimizer.calculate_cost('NonExistent', 1000, 500)
        
        self.assertIn("not found", str(context.exception))

    def test_calculate_cost_no_pricing(self):
        """Test calculate_cost when pricing not available."""
        self.mock_db.get_model_by_name.return_value = {'id': 1, 'name': 'GPT-4'}
        self.mock_db.get_latest_pricing.return_value = []

        with self.assertRaises(ValueError) as context:
            self.optimizer.calculate_cost('GPT-4', 1000, 500)
        
        self.assertIn("Pricing not available", str(context.exception))

    def test_get_cheapest_model(self):
        """Test get_cheapest_model method."""
        # Mock rankings query
        self.mock_db.execute_query.return_value = [
            {'id': 1, 'name': 'GPT-4', 'provider': 'OpenAI', 'score': 1250, 'category': 'overall'},
            {'id': 2, 'name': 'Claude', 'provider': 'Anthropic', 'score': 1240, 'category': 'overall'},
        ]

        # Mock pricing
        self.mock_db.get_latest_pricing.return_value = [
            {'model_id': 1, 'input_cost_per_token': 0.03, 'output_cost_per_token': 0.06},
            {'model_id': 2, 'input_cost_per_token': 0.003, 'output_cost_per_token': 0.015},
        ]

        result = self.optimizer.get_cheapest_model(min_score=1200)

        self.assertIsNotNone(result)
        self.assertEqual(result['model_name'], 'Claude')  # Should be cheaper
        self.assertGreaterEqual(result['score'], 1200)

    def test_get_cheapest_model_no_results(self):
        """Test get_cheapest_model when no models meet criteria."""
        self.mock_db.execute_query.return_value = []

        with self.assertRaises(ValueError) as context:
            self.optimizer.get_cheapest_model(min_score=2000)
        
        self.assertIn("No models found", str(context.exception))

    def test_calculate_value_score(self):
        """Test calculate_value_score method."""
        # Mock model
        self.mock_db.get_model_by_name.return_value = {'id': 1, 'name': 'GPT-4'}

        # Mock history
        self.mock_db.get_arena_history.return_value = [
            {'elo_rating': 1250, 'recorded_at': '2024-01-01'}
        ]

        # Mock pricing
        self.mock_db.get_latest_pricing.return_value = [
            {'model_id': 1, 'input_cost_per_token': 0.01, 'output_cost_per_token': 0.03}
        ]

        # Mock all models for normalization
        self.mock_db.get_models.return_value = [
            {'id': 1, 'name': 'GPT-4'},
            {'id': 2, 'name': 'Claude'}
        ]

        # Mock history for all models
        def mock_history(model_id, days):
            if model_id == 1:
                return [{'elo_rating': 1250}]
            elif model_id == 2:
                return [{'elo_rating': 1240}]
            return []

        self.mock_db.get_arena_history.side_effect = mock_history

        # Mock pricing for all models
        self.mock_db.get_latest_pricing.return_value = [
            {'model_id': 1, 'input_cost_per_token': 0.01, 'output_cost_per_token': 0.03},
            {'model_id': 2, 'input_cost_per_token': 0.003, 'output_cost_per_token': 0.015},
        ]

        score = self.optimizer.calculate_value_score('GPT-4')

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)

    def test_compare_costs(self):
        """Test compare_costs method."""
        import pandas as pd

        # Mock model lookups
        def mock_get_model(name):
            models = {
                'GPT-4': {'id': 1, 'name': 'GPT-4'},
                'Claude': {'id': 2, 'name': 'Claude'}
            }
            return models.get(name)

        self.mock_db.get_model_by_name.side_effect = mock_get_model

        # Mock history
        def mock_history(model_id, days):
            histories = {
                1: [{'elo_rating': 1250}],
                2: [{'elo_rating': 1240}]
            }
            return histories.get(model_id, [])

        self.mock_db.get_arena_history.side_effect = mock_history

        # Mock pricing
        self.mock_db.get_latest_pricing.return_value = [
            {'model_id': 1, 'input_cost_per_token': 0.01, 'output_cost_per_token': 0.03},
            {'model_id': 2, 'input_cost_per_token': 0.003, 'output_cost_per_token': 0.015},
        ]

        # Mock calculate_cost
        def mock_calculate_cost(model_name, input_tokens, output_tokens):
            costs = {
                'GPT-4': (input_tokens / 1000.0 * 0.01) + (output_tokens / 1000.0 * 0.03),
                'Claude': (input_tokens / 1000.0 * 0.003) + (output_tokens / 1000.0 * 0.015)
            }
            return round(costs.get(model_name, 0), 6)

        with patch.object(self.optimizer, 'calculate_cost', side_effect=mock_calculate_cost):
            df = self.optimizer.compare_costs(['GPT-4', 'Claude'], monthly_tokens=1000000)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn('model', df.columns)
        self.assertIn('monthly_cost', df.columns)
        self.assertIn('savings_vs_best', df.columns)

        # Claude should be cheaper
        claude_row = df[df['model'] == 'Claude'].iloc[0]
        gpt4_row = df[df['model'] == 'GPT-4'].iloc[0]
        self.assertLess(claude_row['monthly_cost'], gpt4_row['monthly_cost'])

    def test_compare_costs_with_invalid_model(self):
        """Test compare_costs with invalid model."""
        import pandas as pd

        self.mock_db.get_model_by_name.return_value = None

        df = self.optimizer.compare_costs(['InvalidModel'], monthly_tokens=1000000)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertIsNone(df.iloc[0]['monthly_cost'])
        self.assertIsNotNone(df.iloc[0]['error'])


if __name__ == '__main__':
    unittest.main()

