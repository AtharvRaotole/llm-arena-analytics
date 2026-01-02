"""
Streamlit frontend application for LLM Arena Analytics.

This is the main entry point for the Streamlit dashboard showing
the current LLM Arena leaderboard.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, date, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Add backend to path for imports
# Try multiple possible paths
backend_paths = [
    Path("/app/backend"),  # Absolute path in Docker (mounted volume)
    Path(__file__).parent / "backend",  # /app/backend (relative)
    Path(__file__).parent.parent / "backend",  # ../backend (local dev)
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
    except ImportError as e:
        st.error(f"Failed to import DatabaseManager: {e}")
        DatabaseManager = None
else:
    st.error("Backend directory not found. Please check Docker volume mounts.")
    DatabaseManager = None


# Configure page
st.set_page_config(
    page_title="LLM Arena Analytics",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "LLM Arena Analytics - Real-time intelligence platform for LLM model performance"
    }
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root styling - Clean dark theme */
    .main {
        background: #0a0e27;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }
    
    h1 {
        font-size: 2.5rem;
        color: #ffffff !important;
        margin-bottom: 1rem;
    }
    
    /* Text */
    p, div, span, label {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0 !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #111827;
    }
    
    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1f2937;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background: transparent;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #b0b0b0 !important;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        background: #374151 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #4b5563 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(55, 65, 81, 0.4);
    }
    
    .stButton > button:focus {
        background: #374151 !important;
        box-shadow: none !important;
    }
    
    /* Dataframes */
    .dataframe {
        background: #1a1a2e;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Selectbox and inputs */
    .stSelectbox label, .stSlider label {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        color: #ffffff !important;
    }
    
    /* Cards and containers */
    .element-container {
        background: rgba(17, 24, 39, 0.8);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(31, 41, 55, 0.5);
    }
    
    /* Code blocks */
    code {
        font-family: 'JetBrains Mono', monospace;
        background: #0f0f23;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        color: #a78bfa;
    }
    
    /* Info boxes */
    .stInfo {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
    }
    
    .stSuccess {
        background: rgba(34, 197, 94, 0.1);
        border-left: 4px solid #22c55e;
        border-radius: 4px;
    }
    
    .stWarning {
        background: rgba(234, 179, 8, 0.1);
        border-left: 4px solid #eab308;
        border-radius: 4px;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        border-radius: 4px;
    }
    
    /* Tabs - Remove all blue colors with !important */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(17, 24, 39, 0.8) !important;
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid rgba(31, 41, 55, 0.5) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #9ca3af !important;
        font-weight: 500;
        background: transparent !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: #374151 !important;
    }
    
    /* Override Streamlit's default blue tab styling */
    .stTabs [data-baseweb="tab"]:focus,
    .stTabs [data-baseweb="tab"]:hover {
        background: #374151 !important;
        color: #ffffff !important;
    }
    
    /* Remove blue underline/border */
    .stTabs [aria-selected="true"]::after,
    .stTabs [aria-selected="true"]::before {
        background: #374151 !important;
        border-color: #374151 !important;
    }
    
    /* Remove any blue border-bottom */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #374151 !important;
    }
    
    /* Override any inline styles Streamlit might add */
    .stTabs [data-baseweb="tab"][style*="blue"],
    .stTabs [data-baseweb="tab"][style*="rgb"] {
        background: #374151 !important;
    }
    
    /* Override Streamlit's default blue tab styling */
    .stTabs [data-baseweb="tab"]:focus,
    .stTabs [data-baseweb="tab"]:hover {
        background: #374151 !important;
        color: #ffffff !important;
    }
    
    /* Remove blue underline/border */
    .stTabs [aria-selected="true"]::after,
    .stTabs [aria-selected="true"]::before {
        background: #374151 !important;
        border-color: #374151 !important;
    }
    
    /* Remove any blue border-bottom */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #374151 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #111827;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #374151;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #4b5563;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None
if 'leaderboard_data' not in st.session_state:
    st.session_state.leaderboard_data = None


@st.cache_resource
def get_db_manager():
    """Get database manager instance (cached)."""
    return DatabaseManager()


def get_leaderboard_data(limit: int = 20) -> pd.DataFrame:
    """
    Fetch leaderboard data from database.

    Args:
        limit: Maximum number of models to return

    Returns:
        DataFrame with leaderboard data
    """
    db = get_db_manager()
    
    try:
        # Get latest arena rankings - use DISTINCT ON to get only the most recent ranking per model
        query = """
            SELECT DISTINCT ON (m.id)
                ar.rank_position as rank,
                m.name as model,
                COALESCE(m.provider, 'Unknown') as provider,
                ar.elo_rating as score,
                ar.win_rate as win_rate,
                ar.total_battles as total_battles,
                ar.recorded_at as last_updated
            FROM arena_rankings ar
            JOIN models m ON ar.model_id = m.id
            ORDER BY m.id, ar.recorded_at DESC
            LIMIT %s
        """
        
        results = db.execute_query(query, (limit,))
        
        if results:
            df = pd.DataFrame(results)
            # Normalize column names (PostgreSQL returns lowercase, convert to title case)
            # Map lowercase to proper case
            column_mapping = {
                'rank': 'Rank',
                'model': 'Model',
                'provider': 'Provider',
                'score': 'Score',
                'win_rate': 'Win_Rate',
                'total_battles': 'Total_Battles',
                'last_updated': 'Last_Updated'
            }
            # Only rename columns that exist
            rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
            if rename_dict:
                df = df.rename(columns=rename_dict)
            
            # Ensure critical columns exist (fallback to lowercase if needed)
            if 'Model' not in df.columns:
                if 'model' in df.columns:
                    df['Model'] = df['model']
                else:
                    st.error("Missing Model column in data")
                    return pd.DataFrame()
            
            # Remove duplicates - keep only the most recent entry per model
            if 'Model' in df.columns:
                df = df.sort_values('Last_Updated', ascending=False).drop_duplicates(subset=['Model'], keep='first')
                # Re-sort by rank
                df = df.sort_values('Rank', ascending=True).reset_index(drop=True)
            
            # Format columns
            if 'Score' in df.columns:
                df['Score'] = df['Score'].round(1)
            if 'Win_Rate' in df.columns:
                df['Win_Rate'] = (df['Win_Rate'] * 100).round(1) if df['Win_Rate'].dtype != 'object' else df['Win_Rate']
            if 'Total_Battles' in df.columns:
                df['Total_Battles'] = df['Total_Battles'].astype(int, errors='ignore')
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def get_total_models() -> int:
    """Get total number of models in database."""
    db = get_db_manager()
    try:
        query = "SELECT COUNT(*) as count FROM models"
        results = db.execute_query(query)
        return results[0]['count'] if results else 0
    except Exception:
        return 0


def get_last_updated() -> datetime:
    """Get timestamp of most recent arena ranking."""
    db = get_db_manager()
    try:
        query = """
            SELECT MAX(recorded_at) as last_update
            FROM arena_rankings
        """
        results = db.execute_query(query)
        if results and results[0].get('last_update'):
            return results[0]['last_update']
    except Exception:
        pass
    return None


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_available_models() -> pd.DataFrame:
    """Get list of available models for selection."""
    db = get_db_manager()
    try:
        query = """
            SELECT DISTINCT m.id, m.name, m.provider
            FROM models m
            JOIN arena_rankings ar ON m.id = ar.model_id
            ORDER BY m.name
        """
        results = db.execute_query(query)
        if results:
            return pd.DataFrame(results)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_time_series_data(
    model_ids: List[int],
    start_date: date,
    end_date: date
) -> pd.DataFrame:
    """
    Fetch time series data for selected models.

    Args:
        model_ids: List of model IDs
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with time series data
    """
    if not model_ids:
        return pd.DataFrame()
    
    db = get_db_manager()
    try:
        # Create placeholders for IN clause
        placeholders = ','.join(['%s'] * len(model_ids))
        query = f"""
            SELECT 
                m.name as model,
                m.provider as provider,
                ar.elo_rating as score,
                ar.recorded_at as date
            FROM arena_rankings ar
            JOIN models m ON ar.model_id = m.id
            WHERE ar.model_id IN ({placeholders})
                AND DATE(ar.recorded_at) >= %s
                AND DATE(ar.recorded_at) <= %s
            ORDER BY ar.recorded_at ASC, m.name ASC
        """
        
        params = tuple(model_ids) + (start_date.isoformat(), end_date.isoformat())
        results = db.execute_query(query, params)
        
        if results:
            df = pd.DataFrame(results)
            # Normalize column names to title case
            column_mapping = {
                'model': 'Model',
                'provider': 'Provider',
                'score': 'Score',
                'date': 'Date'
            }
            rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
            if rename_dict:
                df = df.rename(columns=rename_dict)
            # Convert Date to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching time series data: {e}")
        return pd.DataFrame()


def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics for each model.

    Args:
        df: DataFrame with time series data

    Returns:
        DataFrame with statistics
    """
    if df.empty:
        return pd.DataFrame()
    
    stats_list = []
    
    # Handle case-insensitive column access
    model_col = 'Model' if 'Model' in df.columns else ('model' if 'model' in df.columns else None)
    score_col = 'Score' if 'Score' in df.columns else ('score' if 'score' in df.columns else None)
    date_col = 'Date' if 'Date' in df.columns else ('date' if 'date' in df.columns else None)
    
    if not model_col or not score_col:
        return pd.DataFrame()
    
    for model in df[model_col].unique():
        model_data = df[df[model_col] == model].copy()
        if date_col:
            model_data = model_data.sort_values(date_col)
        
        if len(model_data) == 0:
            continue
        
        current_score = model_data[score_col].iloc[-1]
        start_score = model_data[score_col].iloc[0]
        change = current_score - start_score
        volatility = model_data[score_col].std()
        
        # Calculate trend
        if len(model_data) > 1:
            # Simple linear regression slope
            x = np.arange(len(model_data))
            y = model_data[score_col].values
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.1:
                trend = "📈 Upward"
            elif slope < -0.1:
                trend = "📉 Downward"
            else:
                trend = "➡️ Stable"
        else:
            trend = "➡️ Stable"
        
        stats_list.append({
            'Model': model,
            'Current Score': round(current_score, 1),
            'Change': round(change, 1),
            'Volatility': round(volatility, 2),
            'Trend': trend
        })
    
    return pd.DataFrame(stats_list)


def show_performance_trends() -> None:
    """Display performance trends page."""
    st.header("📈 Performance Trends")
    st.markdown("Compare model performance over time")
    
    # Get available models
    available_models_df = get_available_models()
    
    if available_models_df.empty:
        st.warning("⚠️ No models available. Please seed historical data first.")
        return
    
    # Create model selection dictionary
    model_options = {
        f"{row['name']} ({row['provider']})": row['id']
        for _, row in available_models_df.iterrows()
    }
    
    # Sidebar controls
    with st.sidebar:
        st.header("📊 Chart Controls")
        
        # Model selection (multiselect, max 5)
        selected_model_names = st.multiselect(
            "Select Models (max 5)",
            options=list(model_options.keys()),
            default=list(model_options.keys())[:3] if len(model_options) >= 3 else list(model_options.keys()),
            max_selections=5
        )
        
        # Date range picker
        default_end = date.today()
        default_start = default_end - timedelta(days=30)
        
        date_range = st.date_input(
            "Date Range",
            value=(default_start, default_end),
            min_value=date(2020, 1, 1),
            max_value=date.today()
        )
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = default_start
            end_date = default_end
    
    # Get selected model IDs
    selected_model_ids = [model_options[name] for name in selected_model_names if name in model_options]
    
    if not selected_model_ids:
        st.info("👈 Please select at least one model from the sidebar to view trends.")
        return
    
    if start_date > end_date:
        st.error("Start date must be before end date.")
        return
    
    # Fetch time series data
    with st.spinner("Loading time series data..."):
        df = get_time_series_data(selected_model_ids, start_date, end_date)
    
    if df.empty:
        st.warning("⚠️ No data available for the selected models and date range.")
        return
    
    # Create line chart
    st.subheader("Score Evolution Over Time")
    
    # Color palette for models
    colors = px.colors.qualitative.Set3
    if len(selected_model_names) > len(colors):
        colors = colors * (len(selected_model_names) // len(colors) + 1)
    
    fig = go.Figure()
    
    # Handle case-insensitive column access
    model_col = 'Model' if 'Model' in df.columns else ('model' if 'model' in df.columns else None)
    score_col = 'Score' if 'Score' in df.columns else ('score' if 'score' in df.columns else None)
    date_col = 'Date' if 'Date' in df.columns else ('date' if 'date' in df.columns else None)
    
    if not model_col or not score_col or not date_col:
        st.error("Missing required columns in data")
        return go.Figure()
    
    for i, model in enumerate(df[model_col].unique()):
        model_data = df[df[model_col] == model].copy()
        model_data = model_data.sort_values(date_col)
        
        fig.add_trace(go.Scatter(
            x=model_data[date_col],
            y=model_data[score_col],
            mode='lines+markers',
            name=model,
            line=dict(width=3, color=colors[i % len(colors)]),
            marker=dict(size=6, color=colors[i % len(colors)], line=dict(width=1, color='white')),
            hovertemplate=f'<b style="font-size: 14px;">{model}</b><br>' +
                         '<span style="color: #b0b0b0;">Date:</span> %{x|%Y-%m-%d}<br>' +
                         '<span style="color: #b0b0b0;">Score:</span> <b>%{y:.1f}</b><extra></extra>',
            fill='tonexty' if i > 0 else None
        ))
    
    # Add annotations for major events
    annotations = [
        {'date': '2024-11-06', 'text': 'GPT-4 Turbo Released', 'y': 1250},
        {'date': '2024-10-15', 'text': 'Claude 3.5 Sonnet Update', 'y': 1248},
    ]
    
    for ann in annotations:
        try:
            ann_date = pd.to_datetime(ann['date'])
            if start_date <= ann_date.date() <= end_date:
                fig.add_annotation(
                    x=ann_date,
                    y=ann['y'],
                    text=ann['text'],
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='gray',
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='gray',
                    borderwidth=1
                )
        except:
            pass
    
    fig.update_layout(
        title="Arena Score Evolution",
        xaxis_title="Date",
        yaxis_title="Arena Score (ELO Rating)",
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics table
    st.markdown("---")
    st.subheader("Performance Statistics")
    
    stats_df = calculate_statistics(df)
    
    if not stats_df.empty:
        # Display statistics
        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "Current Score": st.column_config.NumberColumn("Current Score", format="%.1f"),
                "Change": st.column_config.NumberColumn("Change", format="%.1f"),
                "Volatility": st.column_config.NumberColumn("Volatility", format="%.2f"),
                "Trend": st.column_config.TextColumn("Trend", width="small")
            }
        )
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_current = stats_df['Current Score'].mean()
            st.metric("Avg Current Score", f"{avg_current:.1f}")
        with col2:
            avg_change = stats_df['Change'].mean()
            st.metric("Avg Change", f"{avg_change:+.1f}")
        with col3:
            avg_volatility = stats_df['Volatility'].mean()
            st.metric("Avg Volatility", f"{avg_volatility:.2f}")
    else:
        st.info("No statistics available for the selected period.")


def color_provider(val: str) -> str:
    """
    Return color based on provider.

    Args:
        val: Provider name

    Returns:
        CSS color string
    """
    provider_colors = {
        'OpenAI': '#10B981',  # Green
        'Anthropic': '#F59E0B',  # Orange/Amber
        'Google': '#3B82F6',  # Blue
        'Meta': '#6366f1',  # Indigo
        'Mistral AI': '#EC4899',  # Pink
        'Unknown': '#6B7280',  # Gray
    }
    return f"background-color: {provider_colors.get(val, '#6B7280')}; color: white"


def show_leaderboard() -> None:
    """Display leaderboard page."""
    st.header("Current Leaderboard")
    
    # Fetch data
    with st.spinner("Loading leaderboard data..."):
        df = get_leaderboard_data(limit=20)
    
    if df.empty:
        st.warning("⚠️ No leaderboard data available. Please run the scraper to populate the database.")
        st.info("""
        **To populate data:**
        1. Ensure PostgreSQL is running
        2. Run the scraper: `cd backend/scrapers && python scraper_pipeline.py`
        3. Refresh this page
        """)
    else:
        # Display stats with professional styling
        st.markdown("### 📊 Overview Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div style='background: #1f2937; padding: 1.5rem; border-radius: 12px; border: 1px solid #374151;'>
                <div style='font-size: 0.9rem; color: #9ca3af; margin-bottom: 0.5rem; font-weight: 500;'>Models Tracked</div>
                <div style='font-size: 2rem; font-weight: 700; color: #ffffff;'>{}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if 'Score' in df.columns:
                avg_score = df['Score'].mean()
                st.markdown("""
                <div style='background: #1f2937; padding: 1.5rem; border-radius: 12px; border: 1px solid #374151;'>
                    <div style='font-size: 0.9rem; color: #9ca3af; margin-bottom: 0.5rem; font-weight: 500;'>Average Score</div>
                    <div style='font-size: 2rem; font-weight: 700; color: #ffffff;'>{:.1f}</div>
                </div>
                """.format(avg_score), unsafe_allow_html=True)
        
        with col3:
            if 'Provider' in df.columns:
                unique_providers = df['Provider'].nunique()
                st.markdown("""
                <div style='background: #1f2937; padding: 1.5rem; border-radius: 12px; border: 1px solid #374151;'>
                    <div style='font-size: 0.9rem; color: #9ca3af; margin-bottom: 0.5rem; font-weight: 500;'>Providers</div>
                    <div style='font-size: 2rem; font-weight: 700; color: #ffffff;'>{}</div>
                </div>
                """.format(unique_providers), unsafe_allow_html=True)
        
        with col4:
            if 'Total_Battles' in df.columns:
                total_battles = df['Total_Battles'].sum() if df['Total_Battles'].notna().any() else 0
                st.markdown("""
                <div style='background: #1f2937; padding: 1.5rem; border-radius: 12px; border: 1px solid #374151;'>
                    <div style='font-size: 0.9rem; color: #9ca3af; margin-bottom: 0.5rem; font-weight: 500;'>Total Battles</div>
                    <div style='font-size: 2rem; font-weight: 700; color: #ffffff;'>{:,}</div>
                </div>
                """.format(int(total_battles)), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display leaderboard table
        st.markdown("### 🎯 Top Models Leaderboard")
        
        # Prepare display dataframe
        display_df = df.copy()
        
        # Select columns to display
        display_columns = ['Rank', 'Model', 'Provider', 'Score']
        if 'Win_Rate' in display_df.columns:
            display_columns.append('Win_Rate')
        if 'Total_Battles' in display_df.columns:
            display_columns.append('Total_Battles')
        
        # Filter to available columns
        display_columns = [col for col in display_columns if col in display_df.columns]
        display_df = display_df[display_columns]
        
        # Rename columns for display
        display_df = display_df.rename(columns={
            'Win_Rate': 'Win Rate (%)',
            'Total_Battles': 'Total Battles'
        })
        
        # Add CSS to completely remove blue colors from table
        st.markdown("""
        <style>
            /* Override Streamlit's default blue selection/hover colors */
            .dataframe tbody tr:hover {
                background: #374151 !important;
            }
            .dataframe tbody tr:hover td {
                background: transparent !important;
                color: #e5e7eb !important;
            }
            /* Remove any blue borders, backgrounds, or highlights */
            .dataframe {
                border: 1px solid #374151 !important;
            }
            .dataframe thead th {
                background: #1f2937 !important;
                border-bottom: 2px solid #374151 !important;
                color: #ffffff !important;
            }
            .dataframe tbody tr {
                background: #111827 !important;
            }
            .dataframe tbody tr:nth-child(even) {
                background: #1f2937 !important;
            }
            .dataframe tbody td {
                color: #e5e7eb !important;
            }
            /* Remove any blue selection */
            .dataframe tbody tr:focus,
            .dataframe tbody tr:focus-within {
                background: #374151 !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Clean, readable dataframe - no row coloring for better readability
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=600
        )
        
        # Additional info
        with st.expander("📈 View Full Data"):
            st.dataframe(df, use_container_width=True, height=400)
        
        # Provider breakdown
        if 'Provider' in df.columns:
            st.markdown("---")
            st.subheader("Provider Breakdown")
            provider_counts = df['Provider'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(provider_counts)
            with col2:
                st.dataframe(
                    provider_counts.reset_index().rename(columns={'index': 'Provider', 'Provider': 'Count'}),
                    use_container_width=True,
                    hide_index=True
                )


def main() -> None:
    """Main dashboard function."""
    # Clean title with minimal spacing
    st.markdown("""
    <div style='text-align: center; padding: 0; margin: 0 0 1rem 0;'>
        <h1 style='font-size: 2.5rem; font-weight: 700; margin: 0; padding: 0; color: #ffffff;'>
            🏆 LLM Arena Analytics
        </h1>
        <p style='font-size: 0.9rem; color: #9ca3af; margin: 0.25rem 0 0 0; padding: 0; font-weight: 400;'>
            Real-time intelligence platform for LLM model performance, costs, and trends
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2 = st.tabs(["🏆 Leaderboard", "📈 Performance Trends"])
    
    # Sidebar (shared across tabs)
    with st.sidebar:
        st.header("📊 Dashboard Info")
        
        # Last updated
        last_updated = get_last_updated()
        if last_updated:
            if isinstance(last_updated, str):
                try:
                    last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                except:
                    pass
            if isinstance(last_updated, datetime):
                st.metric(
                    "Last Updated",
                    last_updated.strftime("%Y-%m-%d %H:%M")
                )
            else:
                st.metric("Last Updated", "N/A")
        else:
            st.metric("Last Updated", "No data")
        
        # Total models
        total_models = get_total_models()
        st.metric("Total Models Tracked", total_models)
        
        st.markdown("---")
        
        # Data source
        st.subheader("📡 Data Source")
        st.markdown("""
        **Chatbot Arena Leaderboard**
        
        Data from [LMSYS Chatbot Arena](https://chat.lmsys.org)
        
        Rankings updated daily via automated scraping.
        """)
        
        st.markdown("---")
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    # Tab content
    with tab1:
        show_leaderboard()
    
    with tab2:
        show_performance_trends()


# Navigation setup - using sidebar for Streamlit 1.28.1 compatibility
if __name__ == "__main__":
    # Clean sidebar navigation
    st.sidebar.markdown("""
    <div style='padding: 1.5rem 0; border-bottom: 2px solid #374151; margin-bottom: 1.5rem;'>
        <h2 style='margin: 0; color: #ffffff; font-size: 1.5rem; font-weight: 600;'>🧭 Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Select Page",
        ["🏆 Leaderboard", "📊 Arena Rankings", "📈 Performance", "💰 Cost Intelligence", "🔮 Market Intelligence"],
        key="page_selector",
        label_visibility="collapsed"
    )
    
    if page == "🏆 Leaderboard":
        main()
    elif page == "📊 Arena Rankings":
        import pages.arena as arena_page
        arena_page.show_arena_page()
    elif page == "📈 Performance":
        import pages.performance as perf_page
        perf_page.show_performance_page()
    elif page == "💰 Cost Intelligence":
        import pages.cost_intelligence as cost_page
        if hasattr(cost_page, 'show_cost_intelligence_page'):
            cost_page.show_cost_intelligence_page()
        else:
            cost_page.main()
    elif page == "🔮 Market Intelligence":
        import pages.market_intel as market_page
        if hasattr(market_page, 'show_market_intel_page'):
            market_page.show_market_intel_page()
        else:
            market_page.main()
else:
    # When imported as a page, run main
    main()
