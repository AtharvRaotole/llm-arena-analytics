"""
Cost Intelligence page for Streamlit frontend.

This module provides cost analysis and optimization tools.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Add backend to path for imports
backend_path = Path(__file__).parent / "backend"
if not backend_path.exists():
    backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from database.db_manager import DatabaseManager
from models.cost_optimizer import CostOptimizer

# Configure page
st.set_page_config(
    page_title="Cost Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_db_manager():
    """Get database manager instance (cached)."""
    return DatabaseManager()


@st.cache_resource
def get_cost_optimizer():
    """Get cost optimizer instance (cached)."""
    db = get_db_manager()
    return CostOptimizer(db_manager=db)


@st.cache_data(ttl=300)
def get_all_models_with_pricing() -> pd.DataFrame:
    """Get all models with pricing data."""
    db = get_db_manager()
    optimizer = get_cost_optimizer()
    
    try:
        models = db.get_models()
        results = []
        
        for model in models:
            try:
                # Get value score
                value_score = optimizer.calculate_value_score(model['name'])
                
                # Get latest pricing
                pricing = db.get_latest_pricing()
                model_pricing = next(
                    (p for p in pricing if p.get('model_id') == model['id']),
                    None
                )
                
                if model_pricing:
                    input_price = model_pricing.get('input_cost_per_token', 0.0) or 0.0
                    output_price = model_pricing.get('output_cost_per_token', 0.0) or 0.0
                    cost_per_1m = (input_price * 500) + (output_price * 500)
                else:
                    cost_per_1m = None
                    input_price = None
                    output_price = None
                
                # Get latest score
                history = db.get_arena_history(model['id'], days=30)
                score = history[0].get('elo_rating', None) if history else None
                
                results.append({
                    'Model': model['name'],
                    'Provider': model.get('provider', 'Unknown'),
                    'Score': float(score) if score else None,
                    'Cost per 1M Tokens': cost_per_1m,
                    'Value Score': value_score,
                    'Input Price/1K': input_price,
                    'Output Price/1K': output_price
                })
            except Exception as e:
                # Skip models with errors
                continue
        
        df = pd.DataFrame(results)
        df = df[df['Value Score'].notna()].sort_values('Value Score', ascending=False)
        return df
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_pricing_history(model_name: str, days: int = 90) -> pd.DataFrame:
    """Get pricing history for a model."""
    db = get_db_manager()
    
    try:
        model = db.get_model_by_name(model_name)
        if not model:
            return pd.DataFrame()
        
        query = """
            SELECT 
                effective_date as date,
                input_cost_per_token,
                output_cost_per_token
            FROM pricing_data
            WHERE model_id = %s
                AND effective_date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY effective_date ASC
        """
        
        results = db.execute_query(query, (model['id'], days))
        
        if results:
            df = pd.DataFrame(results)
            df['date'] = pd.to_datetime(df['date'])
            df['total_cost_per_1k'] = (
                (df['input_cost_per_token'].fillna(0) + df['output_cost_per_token'].fillna(0))
            )
            return df
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()


def show_cost_intelligence_page() -> None:
    """Display the Cost Intelligence page."""
    st.header("💰 Cost Intelligence Center")
    st.markdown("Optimize your LLM costs while maintaining performance")
    
    optimizer = get_cost_optimizer()
    
    # Section 1: Interactive Calculator
    st.markdown("---")
    st.subheader("📊 Interactive Cost Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        monthly_input = st.slider(
            "Monthly Input Tokens",
            min_value=0,
            max_value=100_000_000,
            value=10_000_000,
            step=1_000_000,
            format="%d"
        )
        
        monthly_output = st.slider(
            "Monthly Output Tokens",
            min_value=0,
            max_value=100_000_000,
            value=5_000_000,
            step=1_000_000,
            format="%d"
        )
    
    with col2:
        min_score = st.slider(
            "Minimum Acceptable Score",
            min_value=1000,
            max_value=1300,
            value=1200,
            step=10
        )
        
        st.info(f"""
        **Usage Summary:**
        - Input: {monthly_input:,} tokens/month
        - Output: {monthly_output:,} tokens/month
        - Total: {monthly_input + monthly_output:,} tokens/month
        """)
    
    if st.button("🔍 Calculate Best Value", use_container_width=True, type="primary"):
        with st.spinner("Calculating best value models..."):
            try:
                # Get cheapest model above threshold
                best = optimizer.get_cheapest_model(min_score=min_score)
                
                # Calculate monthly cost for best model
                best_monthly_cost = optimizer.calculate_cost(
                    best['model_name'],
                    monthly_input,
                    monthly_output
                )
                
                # Compare with top models
                all_models_df = get_all_models_with_pricing()
                top_models = all_models_df.head(5)['Model'].tolist()
                
                comparison_data = []
                for model_name in top_models:
                    try:
                        cost = optimizer.calculate_cost(model_name, monthly_input, monthly_output)
                        history = get_db_manager().get_arena_history(
                            get_db_manager().get_model_by_name(model_name)['id'],
                            days=30
                        )
                        score = history[0].get('elo_rating', 0) if history else 0
                        comparison_data.append({
                            'Model': model_name,
                            'Monthly Cost': cost,
                            'Score': score
                        })
                    except:
                        continue
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Display results
                st.success(f"✅ **Recommended Model: {best['model_name']}**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Monthly Cost", f"${best_monthly_cost:,.2f}")
                with col2:
                    st.metric("Score", f"{best['score']:.1f}")
                with col3:
                    # Calculate savings vs GPT-4
                    try:
                        gpt4_cost = optimizer.calculate_cost("GPT-4", monthly_input, monthly_output)
                        savings = gpt4_cost - best_monthly_cost
                        st.metric("Savings vs GPT-4", f"${savings:,.2f}")
                    except:
                        st.metric("Savings vs GPT-4", "N/A")
                
                # Rationale
                st.info(f"""
                **Why {best['model_name']}?**
                - Meets your minimum score requirement ({best['score']:.1f} >= {min_score})
                - Lowest cost per 1M tokens: ${best['cost_per_1m_tokens']:.2f}
                - Best value for your usage pattern
                """)
                
                # Comparison chart
                if not comparison_df.empty:
                    st.subheader("Top 5 Models Comparison")
                    fig = px.bar(
                        comparison_df,
                        x='Model',
                        y='Monthly Cost',
                        color='Score',
                        color_continuous_scale='RdYlGn',
                        title="Monthly Cost Comparison",
                        labels={'Monthly Cost': 'Monthly Cost ($)', 'Score': 'Arena Score'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error calculating best value: {e}")
    
    # Section 2: Value Score Leaderboard
    st.markdown("---")
    st.subheader("🏆 Value Score Leaderboard")
    st.markdown("Models ranked by value score (performance per dollar)")
    
    value_df = get_all_models_with_pricing()
    
    if not value_df.empty:
        # Prepare display dataframe
        display_df = value_df[['Model', 'Provider', 'Score', 'Cost per 1M Tokens', 'Value Score']].copy()
        display_df = display_df.sort_values('Value Score', ascending=False)
        
        # Format for display
        display_df['Score'] = display_df['Score'].round(1)
        display_df['Cost per 1M Tokens'] = display_df['Cost per 1M Tokens'].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
        )
        display_df['Value Score'] = display_df['Value Score'].round(2)
        
        # Style with color gradient
        def color_value_score(val):
            """Color code value score."""
            if pd.isna(val):
                return 'background-color: #f0f0f0'
            # Normalize to 0-1 for color mapping
            normalized = val / 100.0
            # Green (high) to red (low)
            if normalized > 0.7:
                return 'background-color: #90EE90'  # Light green
            elif normalized > 0.5:
                return 'background-color: #FFD700'  # Gold
            elif normalized > 0.3:
                return 'background-color: #FFA500'  # Orange
            else:
                return 'background-color: #FFB6C1'  # Light pink
        
        styled_df = display_df.style.applymap(
            color_value_score,
            subset=['Value Score']
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No value score data available. Please ensure models and pricing data are in the database.")
    
    # Section 3: Price History
    st.markdown("---")
    st.subheader("📈 Price History")
    st.markdown("Track pricing changes over the last 90 days")
    
    # Model selector for price history
    all_models = get_db_manager().get_models()
    model_names = [m['name'] for m in all_models]
    
    if model_names:
        selected_model = st.selectbox(
            "Select Model to View Price History",
            model_names,
            index=0 if len(model_names) > 0 else None
        )
        
        if selected_model:
            price_history = get_pricing_history(selected_model, days=90)
            
            if not price_history.empty:
                # Create line chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=price_history['date'],
                    y=price_history['total_cost_per_1k'],
                    mode='lines+markers',
                    name='Total Cost per 1K Tokens',
                    line=dict(width=2, color='#3B82F6'),
                    marker=dict(size=6)
                ))
                
                # Detect major price drops (more than 20% decrease)
                if len(price_history) > 1:
                    price_history_sorted = price_history.sort_values('date')
                    prev_price = None
                    for idx, row in price_history_sorted.iterrows():
                        current_price = row['total_cost_per_1k']
                        if prev_price is not None and prev_price > 0:
                            change_pct = ((current_price - prev_price) / prev_price) * 100
                            if change_pct < -20:  # More than 20% drop
                                fig.add_annotation(
                                    x=row['date'],
                                    y=current_price,
                                    text=f"Price Drop<br>{change_pct:.1f}%",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowcolor='green',
                                    bgcolor='rgba(144, 238, 144, 0.8)',
                                    bordercolor='green',
                                    borderwidth=1
                                )
                        prev_price = current_price
                
                fig.update_layout(
                    title=f"Price History: {selected_model}",
                    xaxis_title="Date",
                    yaxis_title="Cost per 1K Tokens ($)",
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate total savings
                if len(price_history) > 1:
                    first_price = price_history.iloc[0]['total_cost_per_1k']
                    last_price = price_history.iloc[-1]['total_cost_per_1k']
                    if first_price > 0:
                        savings_pct = ((first_price - last_price) / first_price) * 100
                        st.metric(
                            "Total Price Change",
                            f"{savings_pct:+.1f}%",
                            delta=f"${first_price - last_price:.4f} per 1K tokens"
                        )
            else:
                st.info(f"No price history available for {selected_model} in the last 90 days.")
    else:
        st.warning("No models available. Please seed the database with model data.")


if __name__ == "__main__":
    show_cost_intelligence_page()
