"""
Streamlit frontend application for LLM Arena Analytics.

This is the main entry point for the Streamlit dashboard showing
the current LLM Arena leaderboard.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Add backend to path for imports
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from database.db_manager import DatabaseManager


# Configure page
st.set_page_config(
    page_title="LLM Arena Analytics",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        # Get latest arena rankings
        query = """
            SELECT 
                ar.rank_position as Rank,
                m.name as Model,
                COALESCE(m.provider, 'Unknown') as Provider,
                ar.elo_rating as Score,
                ar.win_rate as Win_Rate,
                ar.total_battles as Total_Battles,
                ar.recorded_at as Last_Updated
            FROM arena_rankings ar
            JOIN models m ON ar.model_id = m.id
            WHERE ar.recorded_at >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY ar.rank_position ASC
            LIMIT %s
        """
        
        results = db.execute_query(query, (limit,))
        
        if not results:
            # Fallback: get any recent rankings
            query = """
                SELECT DISTINCT ON (m.id)
                    ar.rank_position as Rank,
                    m.name as Model,
                    COALESCE(m.provider, 'Unknown') as Provider,
                    ar.elo_rating as Score,
                    ar.win_rate as Win_Rate,
                    ar.total_battles as Total_Battles,
                    ar.recorded_at as Last_Updated
                FROM arena_rankings ar
                JOIN models m ON ar.model_id = m.id
                ORDER BY m.id, ar.recorded_at DESC
                LIMIT %s
            """
            results = db.execute_query(query, (limit,))
        
        if results:
            df = pd.DataFrame(results)
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
                m.name as Model,
                m.provider as Provider,
                ar.elo_rating as Score,
                ar.recorded_at as Date
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
    
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model].copy()
        model_data = model_data.sort_values('Date')
        
        if len(model_data) == 0:
            continue
        
        current_score = model_data['Score'].iloc[-1]
        start_score = model_data['Score'].iloc[0]
        change = current_score - start_score
        volatility = model_data['Score'].std()
        
        # Calculate trend
        if len(model_data) > 1:
            # Simple linear regression slope
            x = np.arange(len(model_data))
            y = model_data['Score'].values
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
    
    for i, model in enumerate(df['Model'].unique()):
        model_data = df[df['Model'] == model].copy()
        model_data = model_data.sort_values('Date')
        
        fig.add_trace(go.Scatter(
            x=model_data['Date'],
            y=model_data['Score'],
            mode='lines+markers',
            name=model,
            line=dict(width=2),
            marker=dict(size=4),
            hovertemplate=f'<b>{model}</b><br>' +
                         'Date: %{x}<br>' +
                         'Score: %{y:.1f}<extra></extra>'
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
        'Meta': '#8B5CF6',  # Purple
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
        # Display stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Models Shown", len(df))
        with col2:
            if 'Score' in df.columns:
                avg_score = df['Score'].mean()
                st.metric("Avg Score", f"{avg_score:.1f}")
        with col3:
            if 'Provider' in df.columns:
                unique_providers = df['Provider'].nunique()
                st.metric("Providers", unique_providers)
        with col4:
            if 'Total_Battles' in df.columns:
                total_battles = df['Total_Battles'].sum()
                st.metric("Total Battles", f"{total_battles:,}")
        
        st.markdown("---")
        
        # Display leaderboard table
        st.subheader("Top 20 Models")
        
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
        
        # Style the dataframe
        def style_row(row):
            """Style each row based on provider."""
            provider = row.get('Provider', 'Unknown')
            colors = {
                'OpenAI': '#E0F2FE',  # Light blue
                'Anthropic': '#FEF3C7',  # Light orange
                'Google': '#DBEAFE',  # Light blue
                'Meta': '#EDE9FE',  # Light purple
                'Mistral AI': '#FCE7F3',  # Light pink
            }
            bg_color = colors.get(provider, '#F3F4F6')  # Light gray default
            return [f'background-color: {bg_color}'] * len(row)
        
        # Apply styling
        styled_df = display_df.style.apply(style_row, axis=1)
        
        # Display styled dataframe
        st.dataframe(
            styled_df,
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
    # Title
    st.title("🏆 LLM Arena Analytics")
    st.markdown("---")
    
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


# Navigation setup
if __name__ == "__main__":
    pg = st.navigation([
        st.Page("app.py", title="Arena", icon="🏆"),
        st.Page("pages/cost_intelligence.py", title="Cost Intelligence", icon="💰"),
        st.Page("pages/market_intel.py", title="Market Intelligence", icon="🔮"),
    ])
    pg.run()
else:
    # When imported as a page, run main
    main()
