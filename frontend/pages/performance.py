"""
Performance analysis page for Streamlit frontend.

This module provides performance metrics and analysis tools with professional styling.
"""

import streamlit as st
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta, date
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Add backend to path for imports
backend_paths = [
    Path("/app/backend"),
    Path(__file__).parent / "backend",
    Path(__file__).parent.parent / "backend",
]

backend_path = None
for path in backend_paths:
    if path.exists() and (path / "database").exists():
        backend_path = path
        break

if backend_path:
    sys.path.insert(0, str(backend_path))
    try:
        from database.db_manager import DatabaseManager
    except ImportError:
        DatabaseManager = None
else:
    DatabaseManager = None

# Configure page
st.set_page_config(
    page_title="Performance Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_manager():
    """Get database manager instance."""
    if DatabaseManager:
        return DatabaseManager()
    return None


def get_models() -> List[Dict]:
    """Get list of models."""
    db = get_db_manager()
    if not db:
        return []
    try:
        return db.get_models()
    except:
        return []


def get_model_history(model_id: int, days: int = 90) -> pd.DataFrame:
    """Get historical performance data for a model."""
    db = get_db_manager()
    if not db:
        return pd.DataFrame()
    
    try:
        history = db.get_arena_history(model_id, days=days)
        if history:
            df = pd.DataFrame(history)
            # Normalize column names
            if 'elo_rating' in df.columns:
                df = df.rename(columns={'elo_rating': 'Score', 'recorded_at': 'Date'})
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()


def show_performance_page() -> None:
    """Display the Performance Analysis page with professional styling."""
    # Professional header
    st.markdown("""
    <div style='padding: 2rem 0; border-bottom: 2px solid rgba(102, 126, 234, 0.3); margin-bottom: 2rem;'>
        <h1 style='margin: 0; font-size: 2.8rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            📈 Performance Analysis
        </h1>
        <p style='margin-top: 0.5rem; color: #b0b0b0; font-size: 1.1rem;'>Deep dive into model performance metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get models
    models = get_models()
    
    if not models:
        st.warning("⚠️ No models available. Please populate the database.")
        return
    
    # Model selection
    with st.sidebar:
        st.markdown("### 🎯 Model Selection")
        model_names = [m.get('name', 'Unknown') for m in models]
        selected_model_name = st.selectbox("Select Model", model_names)
        
        selected_model = next((m for m in models if m.get('name') == selected_model_name), None)
        
        if selected_model:
            st.markdown("---")
            st.markdown("### 📊 Model Info")
            st.markdown(f"**Provider:** {selected_model.get('provider', 'Unknown')}")
            st.markdown(f"**Model ID:** {selected_model.get('id', 'N/A')}")
    
    if not selected_model:
        return
    
    model_id = selected_model.get('id')
    
    # Date range
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📅 Date Range")
        end_date = st.date_input("End Date", value=date.today(), max_value=date.today())
        start_date = st.date_input("Start Date", value=end_date - timedelta(days=30), max_value=end_date)
    
    # Fetch historical data
    days = (end_date - start_date).days
    with st.spinner("🔄 Loading performance data..."):
        df = get_model_history(model_id, days=days)
    
    if df.empty:
        st.warning(f"⚠️ No performance data available for {selected_model_name} in the selected date range.")
        return
    
    # Performance metrics section
    st.markdown(f"### 📊 Performance Metrics: {selected_model_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if 'Score' in df.columns and not df.empty:
        current_score = df['Score'].iloc[-1] if len(df) > 0 else 0
        start_score = df['Score'].iloc[0] if len(df) > 0 else 0
        change = current_score - start_score
        volatility = df['Score'].std() if len(df) > 1 else 0
        
        with col1:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%); 
                        padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.3);'>
                <div style='font-size: 0.9rem; color: #b0b0b0; margin-bottom: 0.5rem;'>Current Score</div>
                <div style='font-size: 2rem; font-weight: 700; color: #ffffff;'>{current_score:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            change_color = '#22c55e' if change >= 0 else '#ef4444'
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(37, 99, 235, 0.2) 100%); 
                        padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.3);'>
                <div style='font-size: 0.9rem; color: #b0b0b0; margin-bottom: 0.5rem;'>Change</div>
                <div style='font-size: 2rem; font-weight: 700; color: {change_color};'>{change:+.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(22, 163, 74, 0.2) 100%); 
                        padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(34, 197, 94, 0.3);'>
                <div style='font-size: 0.9rem; color: #b0b0b0; margin-bottom: 0.5rem;'>Volatility</div>
                <div style='font-size: 2rem; font-weight: 700; color: #ffffff;'>{volatility:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            trend = "📈 Upward" if change > 0.1 else "📉 Downward" if change < -0.1 else "➡️ Stable"
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(234, 179, 8, 0.2) 0%, rgba(202, 138, 4, 0.2) 100%); 
                        padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(234, 179, 8, 0.3);'>
                <div style='font-size: 0.9rem; color: #b0b0b0; margin-bottom: 0.5rem;'>Trend</div>
                <div style='font-size: 1.5rem; font-weight: 700; color: #ffffff;'>{trend}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Performance chart
    st.markdown("### 📈 Score Evolution")
    if 'Score' in df.columns and 'Date' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Score'],
            mode='lines+markers',
            name=selected_model_name,
            line=dict(width=3, color='#667eea'),
            marker=dict(size=6, color='#667eea'),
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        fig.update_layout(
            title={
                'text': f'{selected_model_name} Performance Over Time',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Inter', 'color': '#ffffff'}
            },
            xaxis=dict(
                title='Date',
                titlefont=dict(size=14, family='Inter', color='#b0b0b0'),
                tickfont=dict(size=12, family='Inter', color='#b0b0b0'),
                gridcolor='rgba(102, 126, 234, 0.2)'
            ),
            yaxis=dict(
                title='Arena Score',
                titlefont=dict(size=14, family='Inter', color='#b0b0b0'),
                tickfont=dict(size=12, family='Inter', color='#b0b0b0'),
                gridcolor='rgba(102, 126, 234, 0.2)'
            ),
            template='plotly_dark',
            height=500,
            plot_bgcolor='rgba(26, 26, 46, 0.5)',
            paper_bgcolor='rgba(15, 15, 35, 0.8)',
            font=dict(family='Inter', color='#ffffff'),
            margin=dict(l=60, r=30, t=80, b=60)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
    
    # Model comparison section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔄 Model Comparison")
    
    comparison_models = st.multiselect(
        "Select models to compare",
        model_names,
        default=[selected_model_name] + [m.get('name') for m in models[:2] if m.get('name') != selected_model_name],
        max_selections=5
    )
    
    if comparison_models and len(comparison_models) > 1:
        comparison_data = []
        for model_name in comparison_models:
            model = next((m for m in models if m.get('name') == model_name), None)
            if model:
                hist = get_model_history(model.get('id'), days=days)
                if not hist.empty and 'Score' in hist.columns:
                    latest_score = hist['Score'].iloc[-1] if len(hist) > 0 else 0
                    comparison_data.append({
                        'Model': model_name,
                        'Current Score': latest_score,
                        'Provider': model.get('provider', 'Unknown')
                    })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            
            fig = px.bar(
                comp_df,
                x='Model',
                y='Current Score',
                color='Provider',
                title='Model Performance Comparison',
                color_discrete_map={
                    'OpenAI': '#10B981',
                    'Anthropic': '#F59E0B',
                    'Google': '#3B82F6',
                    'Meta': '#8B5CF6',
                    'Mistral AI': '#EC4899'
                }
            )
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(26, 26, 46, 0.5)',
                paper_bgcolor='rgba(15, 15, 35, 0.8)',
                font=dict(family='Inter', color='#ffffff'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    show_performance_page()
