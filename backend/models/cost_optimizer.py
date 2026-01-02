"""
Cost optimization models for LLM usage.

This module provides optimization algorithms to minimize costs while
maintaining performance requirements.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager


class CostOptimizer:
    """Optimizer for minimizing LLM usage costs."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        """
        Initialize the cost optimizer.

        Args:
            db_manager: Optional database manager instance. If None, creates a new one.
        """
        self.db_manager = db_manager or DatabaseManager()

    def calculate_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate total cost for a given model and token usage.

        Gets latest pricing from database and calculates total cost in USD.

        Args:
            model_name: Name of the model
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in USD, rounded to 6 decimals

        Raises:
            ValueError: If model not found or pricing not available
        """
        # Get model from database
        model = self.db_manager.get_model_by_name(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found in database")

        # Get latest pricing for the model
        pricing = self.db_manager.get_latest_pricing()
        model_pricing = None
        
        for price in pricing:
            if price.get('model_id') == model['id']:
                model_pricing = price
                break
        
        if not model_pricing:
            raise ValueError(f"Pricing not available for model '{model_name}'")

        input_price_per_1k = model_pricing.get('input_cost_per_token', 0.0)
        output_price_per_1k = model_pricing.get('output_cost_per_token', 0.0)

        if input_price_per_1k is None:
            input_price_per_1k = 0.0
        if output_price_per_1k is None:
            output_price_per_1k = 0.0

        # Calculate cost (prices are per 1K tokens)
        input_cost = (input_tokens / 1000.0) * input_price_per_1k
        output_cost = (output_tokens / 1000.0) * output_price_per_1k
        total_cost = input_cost + output_cost

        return round(total_cost, 6)

    def get_cheapest_model(
        self,
        min_score: float,
        task_type: str = "overall"
    ) -> Dict[str, Any]:
        """
        Find the cheapest model above a score threshold.

        Args:
            min_score: Minimum ELO score required
            task_type: Task type filter ("overall", "coding", "creative", "reasoning")

        Returns:
            Dictionary with model name, score, cost per 1M tokens, and provider

        Raises:
            ValueError: If no model meets the criteria
        """
        # Get latest arena rankings
        query = """
            SELECT DISTINCT ON (m.id)
                m.id,
                m.name,
                m.provider,
                ar.elo_rating as score,
                ar.category
            FROM arena_rankings ar
            JOIN models m ON ar.model_id = m.id
            WHERE ar.elo_rating >= %s
        """
        
        params = [min_score]
        
        if task_type != "overall":
            query += " AND ar.category = %s"
            params.append(task_type)
        
        query += """
            ORDER BY m.id, ar.recorded_at DESC
        """
        
        rankings = self.db_manager.execute_query(query, tuple(params))
        
        if not rankings:
            raise ValueError(f"No models found with score >= {min_score} for task type '{task_type}'")

        # Get latest pricing for all models
        pricing = self.db_manager.get_latest_pricing()
        pricing_dict = {p.get('model_id'): p for p in pricing}

        # Calculate cost per 1M tokens for each model
        cheapest = None
        cheapest_cost = float('inf')

        for ranking in rankings:
            model_id = ranking['id']
            model_name = ranking['name']
            score = ranking['score']

            if model_id not in pricing_dict:
                continue

            price_data = pricing_dict[model_id]
            input_price = price_data.get('input_cost_per_token', 0.0) or 0.0
            output_price = price_data.get('output_cost_per_token', 0.0) or 0.0

            # Cost per 1M tokens (assuming 50/50 input/output split)
            cost_per_1m = (input_price * 500) + (output_price * 500)

            if cost_per_1m < cheapest_cost:
                cheapest_cost = cost_per_1m
                cheapest = ranking  # Keep the full ranking dict
                cheapest_pricing = price_data

        if not cheapest:
            raise ValueError(f"No models with pricing data found for score >= {min_score}")

        # Get pricing for cheapest model
        cheapest_pricing = pricing_dict[cheapest['id']]
        input_price = cheapest_pricing.get('input_cost_per_token', 0.0) or 0.0
        output_price = cheapest_pricing.get('output_cost_per_token', 0.0) or 0.0

        return {
            'model': cheapest['name'],
            'model_name': cheapest['name'],  # Backward compatibility
            'provider': cheapest.get('provider'),
            'score': float(cheapest['score']),
            'cost_per_1m': round(cheapest_cost, 2),
            'cost_per_1m_tokens': round(cheapest_cost, 2),  # Backward compatibility
            'input_price_per_1k': input_price * 1000,
            'output_price_per_1k': output_price * 1000
        }

    def calculate_value_score(self, model_name: str) -> float:
        """
        Calculate value score for a model.

        Custom metric: score / (input_price + output_price)
        Higher = better value
        Returns normalized 0-100 score.

        Args:
            model_name: Name of the model

        Returns:
            Value score from 0-100

        Raises:
            ValueError: If model not found or data not available
        """
        # Get model
        model = self.db_manager.get_model_by_name(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found")

        # Get latest score
        history = self.db_manager.get_arena_history(model['id'], days=30)
        if not history:
            raise ValueError(f"No score data available for model '{model_name}'")

        latest_score = history[0].get('elo_rating', 0)
        if not latest_score:
            raise ValueError(f"No valid score for model '{model_name}'")

        # Get latest pricing
        pricing = self.db_manager.get_latest_pricing()
        model_pricing = None

        for price in pricing:
            if price.get('model_id') == model['id']:
                model_pricing = price
                break

        if not model_pricing:
            raise ValueError(f"Pricing not available for model '{model_name}'")

        input_price = model_pricing.get('input_cost_per_token', 0.0) or 0.0
        output_price = model_pricing.get('output_cost_per_token', 0.0) or 0.0

        # Calculate raw value score
        total_price = input_price + output_price
        if total_price == 0:
            # Free model - assign high value score
            raw_value = latest_score / 0.001  # Use small denominator
        else:
            raw_value = latest_score / total_price

        # Normalize to 0-100 scale
        # Get all models for normalization
        all_models = self.db_manager.get_models()
        all_scores = []
        all_prices = []

        for m in all_models:
            m_history = self.db_manager.get_arena_history(m['id'], days=30)
            if m_history:
                m_score = m_history[0].get('elo_rating', 0)
                m_pricing = next((p for p in pricing if p.get('model_id') == m['id']), None)
                if m_pricing:
                    m_input = m_pricing.get('input_cost_per_token', 0.0) or 0.0
                    m_output = m_pricing.get('output_cost_per_token', 0.0) or 0.0
                    m_total_price = m_input + m_output
                    if m_total_price > 0:
                        all_scores.append(m_score)
                        all_prices.append(m_total_price)

        if all_scores and all_prices:
            # Calculate value scores for all models
            all_values = [s / p for s, p in zip(all_scores, all_prices)]
            min_value = min(all_values)
            max_value = max(all_values)

            if max_value > min_value:
                # Normalize to 0-100
                normalized = ((raw_value - min_value) / (max_value - min_value)) * 100
            else:
                normalized = 50.0  # Default if all values are same
        else:
            # Fallback normalization
            normalized = min(100.0, max(0.0, raw_value / 10.0))

        return round(normalized, 2)

    def compare_costs(
        self,
        model_list: List[str],
        monthly_tokens: int,
        input_output_ratio: float = 0.5
    ) -> pd.DataFrame:
        """
        Compare total monthly cost for multiple models.

        Args:
            model_list: List of model names to compare
            monthly_tokens: Estimated tokens per month
            input_output_ratio: Ratio of input tokens (default 0.5 = 50/50 split)

        Returns:
            DataFrame with model, score, monthly_cost, savings_vs_best

        Raises:
            ValueError: If any model not found
        """
        results = []

        for model_name in model_list:
            try:
                # Get model
                model = self.db_manager.get_model_by_name(model_name)
                if not model:
                    results.append({
                        'model': model_name,
                        'score': None,
                        'monthly_cost': None,
                        'savings_vs_best': None,
                        'error': f"Model not found"
                    })
                    continue

                # Get latest score
                history = self.db_manager.get_arena_history(model['id'], days=30)
                score = history[0].get('elo_rating', 0) if history else None

                # Calculate monthly cost
                input_tokens = int(monthly_tokens * input_output_ratio)
                output_tokens = int(monthly_tokens * (1 - input_output_ratio))
                
                monthly_cost = self.calculate_cost(model_name, input_tokens, output_tokens)

                results.append({
                    'model': model_name,
                    'score': float(score) if score else None,
                    'monthly_cost': monthly_cost,
                    'savings_vs_best': None,  # Will calculate after
                    'error': None
                })

            except Exception as e:
                results.append({
                    'model': model_name,
                    'score': None,
                    'monthly_cost': None,
                    'savings_vs_best': None,
                    'error': str(e)
                })

        # Create DataFrame
        df = pd.DataFrame(results)

        # Calculate savings vs best (cheapest)
        valid_costs = df[df['monthly_cost'].notna()]
        if not valid_costs.empty:
            best_cost = valid_costs['monthly_cost'].min()
            df['savings_vs_best'] = df.apply(
                lambda row: round(row['monthly_cost'] - best_cost, 2) if pd.notna(row['monthly_cost']) else None,
                axis=1
            )

        # Sort by monthly cost
        df = df.sort_values('monthly_cost', na_position='last')

        return df
