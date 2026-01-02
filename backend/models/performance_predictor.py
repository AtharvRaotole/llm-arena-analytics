"""
Performance prediction models for LLMs.

This module provides machine learning models to predict LLM performance
based on various features and metrics.
"""

import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager

# Try to import advanced models (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")


class PerformancePredictor:
    """Machine learning model for predicting LLM performance metrics."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        """
        Initialize the performance predictor.

        Args:
            db_manager: Optional database manager instance
        """
        self.db_manager = db_manager or DatabaseManager()
        self.model = None
        self.model_type = None
        self.feature_names = None
        self.is_trained = False
        self.scaler = None
        self.task_type = 'regression'  # or 'classification'

    def load_historical_data(self, days: int = 90) -> pd.DataFrame:
        """
        Load historical arena data from database.

        Args:
            days: Number of days to look back

        Returns:
            DataFrame with historical data
        """
        query = """
            SELECT 
                ar.model_id,
                m.name as model_name,
                m.provider,
                ar.elo_rating as score,
                ar.win_rate,
                ar.total_battles,
                ar.category,
                ar.recorded_at as date
            FROM arena_rankings ar
            JOIN models m ON ar.model_id = m.id
            WHERE ar.recorded_at >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY ar.recorded_at ASC
        """
        
        results = self.db_manager.execute_query(query, (days,))
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def engineer_features(
        self,
        df: pd.DataFrame,
        task_prompt_length: int = 1000,
        task_complexity: int = 5,
        task_domain: str = 'general'
    ) -> pd.DataFrame:
        """
        Engineer features from historical data.

        Args:
            df: Historical data DataFrame
            task_prompt_length: Estimated prompt length in tokens
            task_complexity: Complexity score (1-10)
            task_domain: Domain (coding/creative/reasoning/general)

        Returns:
            DataFrame with engineered features
        """
        features_list = []
        
        for model_id in df['model_id'].unique():
            model_data = df[df['model_id'] == model_id].copy()
            model_data = model_data.sort_values('date')
            
            if len(model_data) == 0:
                continue
            
            # Get latest record
            latest = model_data.iloc[-1]
            
            # Model characteristics
            current_score = latest['score']
            provider = latest['provider'] or 'Unknown'
            
            # Calculate trends
            if len(model_data) >= 7:
                score_7d_ago = model_data.iloc[-7]['score'] if len(model_data) >= 7 else current_score
                trend_7d = current_score - score_7d_ago
            else:
                trend_7d = 0.0
            
            if len(model_data) >= 30:
                score_30d_ago = model_data.iloc[-30]['score'] if len(model_data) >= 30 else current_score
                trend_30d = current_score - score_30d_ago
            else:
                trend_30d = 0.0
            
            # Calculate volatility (std dev of scores)
            volatility = model_data['score'].std() if len(model_data) > 1 else 0.0
            
            # Win rate
            win_rate = latest.get('win_rate', 0.0) or 0.0
            
            # Total battles
            total_battles = latest.get('total_battles', 0) or 0
            
            # Get pricing for cost tier
            pricing = self.db_manager.get_latest_pricing()
            model_pricing = next(
                (p for p in pricing if p.get('model_id') == model_id),
                None
            )
            
            if model_pricing:
                input_price = model_pricing.get('input_cost_per_token', 0.0) or 0.0
                output_price = model_pricing.get('output_cost_per_token', 0.0) or 0.0
                total_price = input_price + output_price
                
                # Cost tier (cheap < 0.01, mid 0.01-0.05, expensive > 0.05)
                if total_price < 0.01:
                    cost_tier = 0  # cheap
                elif total_price < 0.05:
                    cost_tier = 1  # mid
                else:
                    cost_tier = 2  # expensive
            else:
                total_price = 0.0
                cost_tier = 0
            
            # Provider encoding (one-hot would be better, but simple for now)
            provider_map = {
                'OpenAI': 0,
                'Anthropic': 1,
                'Google': 2,
                'Meta': 3,
                'Mistral AI': 4
            }
            provider_encoded = provider_map.get(provider, 5)
            
            # Task characteristics (same for all models in this prediction)
            # Domain encoding
            domain_map = {
                'coding': 0,
                'creative': 1,
                'reasoning': 2,
                'general': 3
            }
            domain_encoded = domain_map.get(task_domain, 3)
            
            # Feature vector
            features = {
                'model_id': model_id,
                'model_name': latest['model_name'],
                # Model characteristics
                'current_score': current_score,
                'trend_7d': trend_7d,
                'trend_30d': trend_30d,
                'volatility': volatility,
                'win_rate': win_rate,
                'total_battles': total_battles,
                'cost_tier': cost_tier,
                'total_price': total_price,
                'provider_encoded': provider_encoded,
                # Task characteristics
                'task_prompt_length': task_prompt_length,
                'task_complexity': task_complexity,
                'task_domain': domain_encoded,
                # Target (will be set during training)
                'target_score': current_score,  # For regression
                'target_success': 1 if current_score >= 1200 else 0  # For classification
            }
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        task_variations: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare training data with task variations.

        Args:
            df: Historical data DataFrame
            task_variations: List of task configurations to generate synthetic data

        Returns:
            Tuple of (features_df, regression_target, classification_target)
        """
        if task_variations is None:
            # Generate default task variations
            task_variations = []
            for prompt_len in [500, 1000, 2000, 5000]:
                for complexity in [1, 3, 5, 7, 9]:
                    for domain in ['coding', 'creative', 'reasoning', 'general']:
                        task_variations.append({
                            'prompt_length': prompt_len,
                            'complexity': complexity,
                            'domain': domain
                        })
        
        all_features = []
        
        for task in task_variations:
            features_df = self.engineer_features(
                df,
                task_prompt_length=task['prompt_length'],
                task_complexity=task['complexity'],
                task_domain=task['domain']
            )
            all_features.append(features_df)
        
        combined_df = pd.concat(all_features, ignore_index=True)
        
        # Select feature columns (exclude targets and identifiers)
        feature_cols = [
            'current_score', 'trend_7d', 'trend_30d', 'volatility',
            'win_rate', 'total_battles', 'cost_tier', 'total_price',
            'provider_encoded', 'task_prompt_length', 'task_complexity', 'task_domain'
        ]
        
        X = combined_df[feature_cols].copy()
        y_regression = combined_df['target_score']
        y_classification = combined_df['target_success']
        
        self.feature_names = feature_cols
        
        return X, y_regression, y_classification

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'random_forest',
        task_type: str = 'regression',
        cv_folds: int = 5,
        tune_hyperparameters: bool = True
    ) -> Dict[str, float]:
        """
        Train the performance prediction model.

        Args:
            X: Feature matrix
            y: Target values
            model_type: Type of model ('random_forest', 'xgboost', 'lightgbm')
            task_type: 'regression' or 'classification'
            cv_folds: Number of cross-validation folds
            tune_hyperparameters: Whether to tune hyperparameters

        Returns:
            Dictionary containing training metrics
        """
        self.task_type = task_type
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize model
        if model_type == 'random_forest':
            if task_type == 'regression':
                base_model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                base_model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            if task_type == 'regression':
                base_model = xgb.XGBRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
            else:
                base_model = xgb.XGBClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            if task_type == 'regression':
                base_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
            else:
                base_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3]
                }
        else:
            # Fallback to random forest
            print(f"Warning: {model_type} not available, using Random Forest")
            model_type = 'random_forest'
            if task_type == 'regression':
                base_model = RandomForestRegressor(random_state=42)
            else:
                base_model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [100],
                'max_depth': [20],
                'min_samples_split': [5]
            }
        
        # Hyperparameter tuning
        if tune_hyperparameters:
            print(f"Tuning hyperparameters for {model_type}...")
            scoring = 'f1' if task_type == 'classification' else 'neg_mean_absolute_error'
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model = base_model
            self.model.fit(X_train, y_train)
        
        self.model_type = model_type
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        if task_type == 'regression':
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'mse': float(mse),
                'rmse': float(np.sqrt(mse)),
                'mae': float(mae),
                'r2_score': float(r2)
            }
        else:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
        
        # Cross-validation scores
        cv_scoring = 'f1' if task_type == 'classification' else 'neg_mean_absolute_error'
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring=cv_scoring)
        
        if task_type == 'classification':
            metrics['cv_f1_mean'] = float(cv_scores.mean())
            metrics['cv_f1_std'] = float(cv_scores.std())
        else:
            metrics['cv_mae_mean'] = float(-cv_scores.mean())
            metrics['cv_mae_std'] = float(cv_scores.std())
        
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict performance for given features.

        Args:
            X: Feature matrix

        Returns:
            Array of predicted values

        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances))
        else:
            return {}

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'task_type': self.task_type
            }, f)
        
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.model_type = data['model_type']
            self.feature_names = data['feature_names']
            self.task_type = data.get('task_type', 'regression')
            self.is_trained = True
        
        print(f"Model loaded from {filepath}")

    def plot_feature_importance(self, save_path: Optional[str] = None) -> None:
        """
        Plot feature importance.

        Args:
            save_path: Optional path to save the plot
        """
        importance = self.get_feature_importance()
        
        if not importance:
            print("No feature importance available")
            return
        
        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_importance)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    """Train and evaluate the performance predictor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train performance predictor model')
    parser.add_argument('--model-type', choices=['random_forest', 'xgboost', 'lightgbm'],
                        default='random_forest', help='Model type to train')
    parser.add_argument('--task-type', choices=['regression', 'classification'],
                        default='regression', help='Task type')
    parser.add_argument('--days', type=int, default=90, help='Days of historical data')
    parser.add_argument('--no-tune', action='store_true', help='Skip hyperparameter tuning')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Performance Predictor Training")
    print("=" * 60)
    print()
    
    # Initialize
    db = DatabaseManager()
    predictor = PerformancePredictor(db_manager=db)
    
    try:
        # Load data
        print(f"Loading historical data (last {args.days} days)...")
        historical_df = predictor.load_historical_data(days=args.days)
        
        if historical_df.empty:
            print("No historical data found. Please seed the database first.")
            sys.exit(1)
        
        print(f"Loaded {len(historical_df)} records")
        print()
        
        # Prepare training data
        print("Engineering features...")
        X, y_reg, y_clf = predictor.prepare_training_data(historical_df)
        print(f"Created {len(X)} training samples with {len(X.columns)} features")
        print()
        
        # Select target based on task type
        y = y_reg if args.task_type == 'regression' else y_clf
        
        # Train model
        print(f"Training {args.model_type} model ({args.task_type})...")
        metrics = predictor.train(
            X, y,
            model_type=args.model_type,
            task_type=args.task_type,
            tune_hyperparameters=not args.no_tune
        )
        
        print()
        print("Training Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        print()
        
        # Feature importance
        importance = predictor.get_feature_importance()
        print("Top 5 Most Important Features:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, imp in sorted_importance[:5]:
            print(f"  {feature}: {imp:.4f}")
        print()
        
        # Save model
        models_dir = Path(__file__).parent.parent.parent / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / "performance_predictor.pkl"
        predictor.save_model(str(model_path))
        
        # Save metrics
        metrics_path = models_dir / "metrics.json"
        metrics['model_type'] = args.model_type
        metrics['task_type'] = args.task_type
        metrics['training_date'] = datetime.now().isoformat()
        metrics['n_samples'] = len(X)
        metrics['n_features'] = len(X.columns)
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")
        
        # Plot feature importance
        plot_path = models_dir / "feature_importance.png"
        predictor.plot_feature_importance(str(plot_path))
        
        print()
        print("=" * 60)
        print("✅ Training completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()
