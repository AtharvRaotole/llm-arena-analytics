"""
Trend forecasting models for LLM market analysis.

This module provides time series forecasting capabilities to predict
future trends in LLM performance, pricing, and adoption.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager

# Try to import Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Install with: pip install prophet")

# Try to import ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("Warning: statsmodels not available. Install with: pip install statsmodels")


class TrendForecaster:
    """Forecasting model for LLM market trends."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        """
        Initialize the trend forecaster.

        Args:
            db_manager: Optional database manager instance
        """
        self.db_manager = db_manager or DatabaseManager()
        self.historical_data: Optional[pd.DataFrame] = None
        self.forecast_method = 'prophet' if PROPHET_AVAILABLE else 'arima'

    def load_historical_data(self, model_name: str, days: int = 180) -> pd.DataFrame:
        """
        Load historical score data for a model.

        Args:
            model_name: Name of the model
            days: Number of days to look back

        Returns:
            DataFrame with historical scores
        """
        model = self.db_manager.get_model_by_name(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found")

        history = self.db_manager.get_arena_history(model['id'], days=days)
        
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        df['recorded_at'] = pd.to_datetime(df['recorded_at'])
        df = df.sort_values('recorded_at')
        
        # Prepare for time series (Prophet format)
        df['ds'] = df['recorded_at'].dt.date
        df['y'] = df['elo_rating'].astype(float)
        
        return df[['ds', 'y', 'recorded_at']].copy()

    def forecast_score(
        self,
        model_name: str,
        days_ahead: int = 30,
        method: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Forecast future scores for a model.

        Args:
            model_name: Name of the model
            days_ahead: Number of days to forecast ahead
            method: Forecasting method ('prophet' or 'arima')

        Returns:
            DataFrame with date, predicted_score, lower_bound, upper_bound
        """
        method = method or self.forecast_method
        
        # Load historical data
        df = self.load_historical_data(model_name, days=180)
        
        if df.empty or len(df) < 7:
            raise ValueError(f"Insufficient historical data for {model_name}")
        
        if method == 'prophet' and PROPHET_AVAILABLE:
            return self._forecast_prophet(df, days_ahead)
        elif method == 'arima' and ARIMA_AVAILABLE:
            return self._forecast_arima(df, days_ahead)
        else:
            # Fallback: simple linear trend
            return self._forecast_linear(df, days_ahead)

    def _forecast_prophet(
        self,
        df: pd.DataFrame,
        days_ahead: int
    ) -> pd.DataFrame:
        """Forecast using Facebook Prophet."""
        # Prepare data for Prophet
        prophet_df = df[['ds', 'y']].copy()
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        # Initialize Prophet with hyperparameters
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,  # Lower = less flexible
            seasonality_prior_scale=10.0,
            interval_width=0.80  # 80% confidence interval
        )
        
        # Fit model
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=days_ahead)
        
        # Forecast
        forecast = model.predict(future)
        
        # Extract forecast period only
        forecast_period = forecast.tail(days_ahead).copy()
        
        result = pd.DataFrame({
            'date': forecast_period['ds'].dt.date,
            'predicted_score': forecast_period['yhat'].values,
            'lower_bound': forecast_period['yhat_lower'].values,
            'upper_bound': forecast_period['yhat_upper'].values
        })
        
        return result

    def _forecast_arima(
        self,
        df: pd.DataFrame,
        days_ahead: int
    ) -> pd.DataFrame:
        """Forecast using ARIMA."""
        # Prepare time series
        ts = df.set_index('ds')['y'].sort_index()
        
        # Check for stationarity
        adf_result = adfuller(ts.dropna())
        is_stationary = adf_result[1] < 0.05
        
        # Auto-select ARIMA parameters (simplified)
        # Try different combinations
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        # Fit best model
        model = ARIMA(ts, order=best_order)
        fitted = model.fit()
        
        # Forecast
        forecast = fitted.get_forecast(steps=days_ahead)
        forecast_ci = forecast.conf_int()
        
        # Create result dataframe
        future_dates = pd.date_range(start=ts.index[-1] + timedelta(days=1), periods=days_ahead, freq='D')
        
        result = pd.DataFrame({
            'date': future_dates.date,
            'predicted_score': forecast.predicted_mean.values,
            'lower_bound': forecast_ci.iloc[:, 0].values,
            'upper_bound': forecast_ci.iloc[:, 1].values
        })
        
        return result

    def _forecast_linear(
        self,
        df: pd.DataFrame,
        days_ahead: int
    ) -> pd.DataFrame:
        """Fallback: Simple linear trend forecast."""
        df_sorted = df.sort_values('ds')
        dates = pd.to_datetime(df_sorted['ds'])
        scores = df_sorted['y'].values
        
        # Linear regression
        x = np.arange(len(dates))
        coeffs = np.polyfit(x, scores, 1)
        trend = np.poly1d(coeffs)
        
        # Forecast
        future_x = np.arange(len(dates), len(dates) + days_ahead)
        future_dates = [dates.iloc[-1].date() + timedelta(days=i+1) for i in range(days_ahead)]
        predicted = trend(future_x)
        
        # Simple confidence interval (based on historical std)
        std_dev = np.std(scores)
        lower = predicted - 1.96 * std_dev
        upper = predicted + 1.96 * std_dev
        
        result = pd.DataFrame({
            'date': future_dates,
            'predicted_score': predicted,
            'lower_bound': lower,
            'upper_bound': upper
        })
        
        return result

    def detect_trend(
        self,
        model_name: str,
        window: int = 30
    ) -> Dict[str, Any]:
        """
        Detect trend direction for a model.

        Args:
            model_name: Name of the model
            window: Number of days to analyze

        Returns:
            Dictionary with trend direction and percentage change
        """
        df = self.load_historical_data(model_name, days=window)
        
        if df.empty or len(df) < 2:
            return {
                'trend': 'unknown',
                'percentage_change': 0.0,
                'message': 'Insufficient data'
            }
        
        df_sorted = df.sort_values('ds')
        first_score = df_sorted.iloc[0]['y']
        last_score = df_sorted.iloc[-1]['y']
        
        percentage_change = ((last_score - first_score) / first_score) * 100
        
        # Determine trend
        if percentage_change > 2.0:
            trend = 'rising'
        elif percentage_change < -2.0:
            trend = 'falling'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'percentage_change': round(percentage_change, 2),
            'start_score': float(first_score),
            'end_score': float(last_score),
            'window_days': window
        }

    def rank_forecast(
        self,
        days_ahead: int = 30
    ) -> pd.DataFrame:
        """
        Forecast future rankings for all models.

        Args:
            days_ahead: Number of days to forecast ahead

        Returns:
            DataFrame with model, current_rank, predicted_rank, rank_change
        """
        models = self.db_manager.get_models()
        
        forecasts = []
        
        for model in models:
            try:
                # Get current rank
                history = self.db_manager.get_arena_history(model['id'], days=1)
                if not history:
                    continue
                
                current_score = history[0].get('elo_rating', 0)
                current_rank = history[0].get('rank_position', 999)
                
                # Forecast future score
                forecast_df = self.forecast_score(model['name'], days_ahead=days_ahead)
                if forecast_df.empty:
                    continue
                
                predicted_score = forecast_df.iloc[-1]['predicted_score']
                
                forecasts.append({
                    'model': model['name'],
                    'provider': model.get('provider', 'Unknown'),
                    'current_score': current_score,
                    'current_rank': current_rank,
                    'predicted_score': predicted_score
                })
            except Exception as e:
                # Skip models with errors
                continue
        
        if not forecasts:
            return pd.DataFrame()
        
        forecast_df = pd.DataFrame(forecasts)
        
        # Calculate predicted ranks (based on predicted scores)
        forecast_df = forecast_df.sort_values('predicted_score', ascending=False)
        forecast_df['predicted_rank'] = range(1, len(forecast_df) + 1)
        
        # Calculate rank change
        forecast_df['rank_change'] = forecast_df['current_rank'] - forecast_df['predicted_rank']
        
        # Sort by predicted rank
        forecast_df = forecast_df.sort_values('predicted_rank')
        
        return forecast_df[['model', 'provider', 'current_rank', 'predicted_rank', 
                           'rank_change', 'current_score', 'predicted_score']]

    def anomaly_detection(
        self,
        model_name: str,
        threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in model scores.

        Args:
            model_name: Name of the model
            threshold: Standard deviations threshold for anomaly detection

        Returns:
            List of anomaly records with dates and scores
        """
        df = self.load_historical_data(model_name, days=180)
        
        if df.empty or len(df) < 10:
            return []
        
        df_sorted = df.sort_values('ds')
        scores = df_sorted['y'].values
        
        # Calculate rolling mean and std
        window = min(7, len(scores) // 3)
        rolling_mean = pd.Series(scores).rolling(window=window, center=True).mean()
        rolling_std = pd.Series(scores).rolling(window=window, center=True).std()
        
        # Detect anomalies (scores beyond threshold * std from mean)
        anomalies = []
        for i, (date, score, mean, std) in enumerate(zip(
            df_sorted['ds'],
            scores,
            rolling_mean,
            rolling_std
        )):
            if pd.isna(mean) or pd.isna(std) or std == 0:
                continue
            
            z_score = abs((score - mean) / std)
            
            if z_score > threshold:
                # Calculate change from previous score
                prev_score = scores[i-1] if i > 0 else score
                change = score - prev_score
                change_pct = (change / prev_score * 100) if prev_score > 0 else 0
                
                anomalies.append({
                    'date': date.date() if hasattr(date, 'date') else date,
                    'score': float(score),
                    'z_score': float(z_score),
                    'change': float(change),
                    'change_percentage': round(change_pct, 2),
                    'type': 'spike' if change > 0 else 'drop'
                })
        
        return sorted(anomalies, key=lambda x: x['z_score'], reverse=True)

    def backtest(
        self,
        model_name: str,
        test_days: int = 30
    ) -> Dict[str, float]:
        """
        Backtest forecast accuracy on historical data.

        Args:
            model_name: Name of the model
            test_days: Number of days to use for testing

        Returns:
            Dictionary with accuracy metrics
        """
        # Load data excluding last test_days
        df = self.load_historical_data(model_name, days=180)
        
        if df.empty or len(df) < test_days + 7:
            return {'error': 'Insufficient data for backtesting'}
        
        df_sorted = df.sort_values('ds')
        train_df = df_sorted.iloc[:-test_days].copy()
        test_df = df_sorted.iloc[-test_days:].copy()
        
        # Forecast on training data
        try:
            forecast_df = self._forecast_prophet(train_df, days_ahead=test_days) if PROPHET_AVAILABLE else \
                         self._forecast_arima(train_df, days_ahead=test_days) if ARIMA_AVAILABLE else \
                         self._forecast_linear(train_df, days_ahead=test_days)
        except Exception as e:
            return {'error': f'Forecast failed: {str(e)}'}
        
        # Align dates and compare
        test_scores = test_df['y'].values
        predicted_scores = forecast_df['predicted_score'].values[:len(test_scores)]
        
        if len(test_scores) != len(predicted_scores):
            min_len = min(len(test_scores), len(predicted_scores))
            test_scores = test_scores[:min_len]
            predicted_scores = predicted_scores[:min_len]
        
        # Calculate metrics
        mae = mean_absolute_error(test_scores, predicted_scores)
        mse = mean_squared_error(test_scores, predicted_scores)
        rmse = np.sqrt(mse)
        
        # Mean absolute percentage error
        mape = np.mean(np.abs((test_scores - predicted_scores) / test_scores)) * 100
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'test_days': len(test_scores)
        }

    def plot_forecast(
        self,
        model_name: str,
        days_ahead: int = 30,
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize forecast with historical data.

        Args:
            model_name: Name of the model
            days_ahead: Number of days to forecast
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        
        # Load historical data
        historical_df = self.load_historical_data(model_name, days=180)
        
        if historical_df.empty:
            print(f"No historical data for {model_name}")
            return
        
        # Generate forecast
        forecast_df = self.forecast_score(model_name, days_ahead=days_ahead)
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Historical data
        plt.plot(
            pd.to_datetime(historical_df['ds']),
            historical_df['y'],
            'b-',
            label='Historical',
            linewidth=2
        )
        
        # Forecast
        forecast_dates = pd.to_datetime(forecast_df['date'])
        plt.plot(
            forecast_dates,
            forecast_df['predicted_score'],
            'r--',
            label='Forecast',
            linewidth=2
        )
        
        # Confidence interval
        plt.fill_between(
            forecast_dates,
            forecast_df['lower_bound'],
            forecast_df['upper_bound'],
            alpha=0.3,
            color='red',
            label='80% Confidence Interval'
        )
        
        plt.xlabel('Date')
        plt.ylabel('Arena Score')
        plt.title(f'Score Forecast: {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    """Test the trend forecaster."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trend forecasting')
    parser.add_argument('--model', type=str, default='GPT-4 Turbo', help='Model name')
    parser.add_argument('--days', type=int, default=30, help='Days to forecast')
    parser.add_argument('--method', choices=['prophet', 'arima', 'linear'], 
                        default=None, help='Forecasting method')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Trend Forecaster Test")
    print("=" * 60)
    print()
    
    db = DatabaseManager()
    forecaster = TrendForecaster(db_manager=db)
    
    try:
        # Test 1: Forecast score
        print(f"[Test 1] Forecasting scores for {args.model} ({args.days} days ahead)...")
        forecast_df = forecaster.forecast_score(args.model, days_ahead=args.days, method=args.method)
        
        if not forecast_df.empty:
            print(f"✅ Generated forecast for {len(forecast_df)} days")
            print("\nSample forecast:")
            print(forecast_df.head(10).to_string(index=False))
        else:
            print("⚠️  No forecast generated")
        
        print()
        
        # Test 2: Detect trend
        print(f"[Test 2] Detecting trend for {args.model}...")
        trend = forecaster.detect_trend(args.model, window=30)
        print(f"✅ Trend: {trend['trend']}")
        print(f"   Change: {trend['percentage_change']:.2f}%")
        print()
        
        # Test 3: Rank forecast
        print(f"[Test 3] Forecasting rankings ({args.days} days ahead)...")
        rank_forecast = forecaster.rank_forecast(days_ahead=args.days)
        
        if not rank_forecast.empty:
            print(f"✅ Forecasted rankings for {len(rank_forecast)} models")
            print("\nTop 10 predicted rankings:")
            print(rank_forecast.head(10).to_string(index=False))
        else:
            print("⚠️  No rank forecast generated")
        
        print()
        
        # Test 4: Anomaly detection
        print(f"[Test 4] Detecting anomalies for {args.model}...")
        anomalies = forecaster.anomaly_detection(args.model)
        
        if anomalies:
            print(f"✅ Found {len(anomalies)} anomalies")
            print("\nTop 5 anomalies:")
            for i, anomaly in enumerate(anomalies[:5], 1):
                print(f"  {i}. {anomaly['date']}: Score {anomaly['score']:.1f} "
                      f"({anomaly['type']}, {anomaly['change_percentage']:+.1f}%)")
        else:
            print("✅ No anomalies detected")
        
        print()
        
        # Test 5: Backtest
        print(f"[Test 5] Backtesting forecast accuracy...")
        backtest_results = forecaster.backtest(args.model, test_days=30)
        
        if 'error' not in backtest_results:
            print(f"✅ Backtest results:")
            print(f"   MAE: {backtest_results['mae']:.2f}")
            print(f"   RMSE: {backtest_results['rmse']:.2f}")
            print(f"   MAPE: {backtest_results['mape']:.2f}%")
        else:
            print(f"⚠️  {backtest_results['error']}")
        
        print()
        print("=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
