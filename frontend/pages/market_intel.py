"""
Market intelligence page for Streamlit frontend.

This module provides trend analysis, predictions, and market insights.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

# Add backend to path for imports
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from database.db_manager import DatabaseManager
from models.trend_forecaster import TrendForecaster

# Configure page
st.set_page_config(
    page_title="Market Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL (can be configured)
API_BASE_URL = "http://localhost:8000"


@st.cache_resource
def get_db_manager():
    """Get database manager instance (cached)."""
    return DatabaseManager()


@st.cache_resource
def get_trend_forecaster():
    """Get trend forecaster instance (cached)."""
    db = get_db_manager()
    return TrendForecaster(db_manager=db)


@st.cache_data(ttl=300)
def fetch_rank_forecast(days_ahead: int = 30) -> pd.DataFrame:
    """Fetch rank forecast from API or directly."""
    try:
        # Try API first
        response = requests.get(f"{API_BASE_URL}/forecast/rank", params={"days_ahead": days_ahead}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
    except:
        pass
    
    # Fallback: direct database access
    forecaster = get_trend_forecaster()
    return forecaster.rank_forecast(days_ahead=days_ahead)


@st.cache_data(ttl=300)
def fetch_model_forecast(model_name: str, days_ahead: int = 30) -> Dict[str, Any]:
    """Fetch model forecast from API or directly."""
    try:
        response = requests.get(
            f"{API_BASE_URL}/forecast/model/{model_name}",
            params={"days_ahead": days_ahead},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Fallback: direct access
    forecaster = get_trend_forecaster()
    forecast_df = forecaster.forecast_score(model_name, days_ahead=days_ahead)
    trend = forecaster.detect_trend(model_name, window=30)
    
    return {
        'model_name': model_name,
        'forecast': forecast_df.to_dict('records'),
        'trend': trend,
        'days_ahead': days_ahead
    }


@st.cache_data(ttl=300)
def fetch_anomalies(model_name: str) -> List[Dict[str, Any]]:
    """Fetch anomalies for a model."""
    try:
        response = requests.get(f"{API_BASE_URL}/forecast/anomalies/{model_name}", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Fallback
    forecaster = get_trend_forecaster()
    return forecaster.anomaly_detection(model_name)


def get_rank_change_indicator(change: int) -> str:
    """Get indicator for rank change."""
    if change > 0:
        return f"↑{change}"
    elif change < 0:
        return f"↓{abs(change)}"
    else:
        return "→"


def show_market_intel_page() -> None:
    """Display the Market Intelligence page."""
    st.header("🔮 Market Intelligence")
    st.markdown("Predictions, trends, and insights for LLM market")
    
    forecaster = get_trend_forecaster()
    
    # Section 1: Future Rankings Prediction
    st.markdown("---")
    st.subheader("🔮 Future Rankings Prediction")
    st.markdown("Predicted rank changes for the next 30 days")
    
    forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30, key="forecast_days")
    
    with st.spinner("Forecasting rankings..."):
        rank_forecast_df = fetch_rank_forecast(days_ahead=forecast_days)
    
    if not rank_forecast_df.empty:
        # Prepare display dataframe
        display_df = rank_forecast_df.copy()
        display_df['change_indicator'] = display_df['rank_change'].apply(get_rank_change_indicator)
        
        # Color coding function
        def color_rank_change(val):
            if val > 0:
                return 'background-color: #90EE90'  # Light green
            elif val < 0:
                return 'background-color: #FFB6C1'  # Light pink
            else:
                return 'background-color: #F0F0F0'  # Light gray
        
        # Display table
        styled_df = display_df.style.applymap(
            color_rank_change,
            subset=['rank_change']
        )
        
        st.dataframe(
            styled_df[['model', 'provider', 'current_rank', 'predicted_rank', 
                      'change_indicator', 'current_score', 'predicted_score']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'model': 'Model',
                'provider': 'Provider',
                'current_rank': 'Current Rank',
                'predicted_rank': 'Predicted Rank',
                'change_indicator': 'Change',
                'current_score': st.column_config.NumberColumn('Current Score', format="%.1f"),
                'predicted_score': st.column_config.NumberColumn('Predicted Score', format="%.1f")
            }
        )
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            moving_up = len(display_df[display_df['rank_change'] > 0])
            st.metric("Moving Up", moving_up)
        with col2:
            moving_down = len(display_df[display_df['rank_change'] < 0])
            st.metric("Moving Down", moving_down)
        with col3:
            stable = len(display_df[display_df['rank_change'] == 0])
            st.metric("Stable", stable)
        with col4:
            avg_change = display_df['rank_change'].mean()
            st.metric("Avg Change", f"{avg_change:+.1f}")
    else:
        st.warning("No forecast data available. Please ensure historical data is populated.")
    
    # Section 2: Trend Analysis
    st.markdown("---")
    st.subheader("📊 Trend Analysis")
    st.markdown("Historical trends and forecasts for top models")
    
    # Get top models
    db = get_db_manager()
    models = db.get_models()
    
    if models:
        # Select models to display
        model_names = [m['name'] for m in models[:10]]  # Top 10
        selected_models = st.multiselect(
            "Select Models to Analyze",
            model_names,
            default=model_names[:5] if len(model_names) >= 5 else model_names
        )
        
        if selected_models:
            # Create grid of charts
            cols_per_row = 2
            rows = (len(selected_models) + cols_per_row - 1) // cols_per_row
            
            for i, model_name in enumerate(selected_models):
                row = i // cols_per_row
                col = i % cols_per_row
                
                with st.container():
                    if col == 0:
                        chart_cols = st.columns(cols_per_row)
                    
                    with chart_cols[col]:
                        try:
                            # Get forecast data
                            forecast_data = fetch_model_forecast(model_name, days_ahead=30)
                            
                            if forecast_data and 'forecast' in forecast_data:
                                # Load historical data
                                historical_df = forecaster.load_historical_data(model_name, days=90)
                                
                                if not historical_df.empty:
                                    # Create sparkline chart
                                    fig = go.Figure()
                                    
                                    # Historical data
                                    fig.add_trace(go.Scatter(
                                        x=pd.to_datetime(historical_df['ds']),
                                        y=historical_df['y'],
                                        mode='lines',
                                        name='Historical',
                                        line=dict(color='blue', width=2)
                                    ))
                                    
                                    # Forecast
                                    forecast_df = pd.DataFrame(forecast_data['forecast'])
                                    forecast_df = pd.DataFrame(forecast_data['forecast'])
                                    forecast_dates = pd.to_datetime(forecast_df['date'])
                                    
                                    fig.add_trace(go.Scatter(
                                        x=forecast_dates,
                                        y=forecast_df['predicted_score'],
                                        mode='lines',
                                        name='Forecast',
                                        line=dict(color='red', width=2, dash='dash')
                                    ))
                                    
                                    # Confidence interval
                                    fig.add_trace(go.Scatter(
                                        x=forecast_dates,
                                        y=forecast_df['upper_bound'],
                                        mode='lines',
                                        name='Upper Bound',
                                        line=dict(width=0),
                                        showlegend=False
                                    ))
                                    
                                    fig.add_trace(go.Scatter(
                                        x=forecast_dates,
                                        y=forecast_df['lower_bound'],
                                        mode='lines',
                                        name='Confidence Interval',
                                        fill='tonexty',
                                        fillcolor='rgba(255,0,0,0.2)',
                                        line=dict(width=0),
                                        showlegend=False
                                    ))
                                    
                                    # Trend indicator
                                    trend = forecast_data.get('trend', {})
                                    trend_text = trend.get('trend', 'unknown')
                                    trend_pct = trend.get('percentage_change', 0)
                                    
                                    fig.update_layout(
                                        title=f"{model_name}<br><sub>{trend_text.title()} ({trend_pct:+.1f}%)</sub>",
                                        height=250,
                                        showlegend=False,
                                        margin=dict(l=0, r=0, t=40, b=0),
                                        xaxis=dict(showgrid=False),
                                        yaxis=dict(showgrid=True)
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True, use_container_height=True)
                        except Exception as e:
                            st.error(f"Error loading {model_name}: {e}")
    
    # Section 3: Market Events
    st.markdown("---")
    st.subheader("⚠️ Market Events")
    st.markdown("Detected anomalies and significant changes")
    
    # Get all models and their anomalies
    all_anomalies = []
    if models:
        for model in models[:10]:  # Check top 10 models
            try:
                anomalies = fetch_anomalies(model['name'])
                for anomaly in anomalies:
                    anomaly['model'] = model['name']
                    anomaly['provider'] = model.get('provider', 'Unknown')
                    all_anomalies.append(anomaly)
            except:
                continue
    
    if all_anomalies:
        # Sort by date (most recent first)
        all_anomalies.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        # Display timeline
        for anomaly in all_anomalies[:20]:  # Show top 20
            with st.expander(
                f"**{anomaly['date']}** - {anomaly['model']} - "
                f"{anomaly['type'].title()} ({anomaly['change_percentage']:+.1f}%)"
            ):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Score", f"{anomaly['score']:.1f}")
                    st.metric("Change", f"{anomaly['change']:+.1f}")
                with col2:
                    st.metric("Z-Score", f"{anomaly['z_score']:.2f}")
                    st.metric("Change %", f"{anomaly['change_percentage']:+.1f}%")
                
                st.info(f"**Type:** {anomaly['type'].title()} | **Provider:** {anomaly.get('provider', 'Unknown')}")
    else:
        st.info("No significant market events detected in the last 90 days.")
    
    # Section 4: Insights & Recommendations
    st.markdown("---")
    st.subheader("💡 Insights & Recommendations")
    
    # Generate insights
    insights = []
    recommendations = []
    
    if not rank_forecast_df.empty:
        # Analyze rank changes
        top_movers_up = rank_forecast_df.nlargest(3, 'rank_change')
        top_movers_down = rank_forecast_df.nsmallest(3, 'rank_change')
        
        for _, row in top_movers_up.iterrows():
            if row['rank_change'] > 0:
                insights.append(
                    f"**{row['model']}** showing strong upward trend "
                    f"(predicted to move up {int(row['rank_change'])} ranks)"
                )
        
        for _, row in top_movers_down.iterrows():
            if row['rank_change'] < 0:
                insights.append(
                    f"**{row['model']}** losing ground "
                    f"(predicted to drop {int(abs(row['rank_change']))} ranks)"
                )
        
        # Trend insights
        if 'selected_models' in locals() and selected_models:
            for model_name in selected_models[:5]:
            try:
                forecast_data = fetch_model_forecast(model_name, days_ahead=30)
                trend = forecast_data.get('trend', {})
                if trend.get('trend') == 'rising' and trend.get('percentage_change', 0) > 5:
                    insights.append(
                        f"**{model_name}** showing strong upward trend "
                        f"(+{trend['percentage_change']:.1f}% over 30 days)"
                    )
            except:
                continue
    
    # Display insights
    if insights:
        st.markdown("### Key Insights")
        for insight in insights[:5]:
            st.markdown(f"- {insight}")
    else:
        st.info("Generate insights by analyzing forecast data above.")
    
    # Recommendations
    st.markdown("### Recommendations")
    
    if not rank_forecast_df.empty:
        # Find best value models (high score, low cost)
        try:
            from models.cost_optimizer import CostOptimizer
            optimizer = CostOptimizer(db_manager=db)
            
            recommendations_list = []
            for _, row in rank_forecast_df.head(5).iterrows():
                try:
                    value_score = optimizer.calculate_value_score(row['model'])
                    if value_score > 70:
                        recommendations_list.append(
                            f"**{row['model']}** offers excellent value "
                            f"(Value Score: {value_score:.1f}/100) - consider for cost-sensitive projects"
                        )
                except:
                    continue
            
            if recommendations_list:
                for rec in recommendations_list[:3]:
                    st.markdown(f"- {rec}")
            else:
                st.info("Run cost analysis to generate recommendations.")
        except:
            st.info("Cost optimizer not available for recommendations.")
    else:
        st.info("Generate recommendations by analyzing forecast and cost data.")


if __name__ == "__main__":
    show_market_intel_page()
