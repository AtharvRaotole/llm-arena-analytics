"""
Arena rankings page for Streamlit frontend.

This module displays Chatbot Arena leaderboards and rankings with professional styling.
"""

import streamlit as st
import sys
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    page_title="Arena Rankings",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; color: #ffffff !important; }
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_manager():
    """Get database manager instance."""
    if DatabaseManager:
        return DatabaseManager()
    return None


def get_arena_data(limit: int = 50) -> pd.DataFrame:
    """Fetch arena rankings data."""
    db = get_db_manager()
    if not db:
        return pd.DataFrame()
    
    try:
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
            # Normalize column names
            column_mapping = {
                'rank': 'Rank',
                'model': 'Model',
                'provider': 'Provider',
                'score': 'Score',
                'win_rate': 'Win_Rate',
                'total_battles': 'Total_Battles',
                'last_updated': 'Last_Updated'
            }
            rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
            if rename_dict:
                df = df.rename(columns=rename_dict)
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def show_arena_page() -> None:
    """Display the Arena rankings page with professional styling."""
    # Professional header
    st.markdown("""
    <div style='padding: 2rem 0; border-bottom: 2px solid rgba(102, 126, 234, 0.3); margin-bottom: 2rem;'>
        <h1 style='margin: 0; font-size: 2.8rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🏆 Chatbot Arena Rankings
        </h1>
        <p style='margin-top: 0.5rem; color: #b0b0b0; font-size: 1.1rem;'>Real-time leaderboard from LMSYS Chatbot Arena</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filters in sidebar
    with st.sidebar:
        st.markdown("### 🔍 Filters")
        provider_filter = st.selectbox(
            "Provider",
            ["All", "OpenAI", "Anthropic", "Google", "Meta", "Mistral AI"],
            key="arena_provider_filter"
        )
        sort_by = st.selectbox(
            "Sort By",
            ["Rank", "Score", "Model Name"],
            key="arena_sort"
        )
        limit = st.slider("Number of Results", 10, 100, 50, key="arena_limit")
    
    # Fetch data
    with st.spinner("🔄 Loading arena data..."):
        df = get_arena_data(limit=limit)
    
    if df.empty:
        st.warning("⚠️ No arena data available. Please populate the database.")
        return
    
    # Apply filters
    if provider_filter != "All" and 'Provider' in df.columns:
        df = df[df['Provider'] == provider_filter]
    
    # Sort
    if sort_by == "Score" and 'Score' in df.columns:
        df = df.sort_values('Score', ascending=False)
    elif sort_by == "Model Name" and 'Model' in df.columns:
        df = df.sort_values('Model')
    elif 'Rank' in df.columns:
        df = df.sort_values('Rank')
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size: 0.9rem; color: #b0b0b0; margin-bottom: 0.5rem;'>Models Shown</div>
            <div style='font-size: 2rem; font-weight: 700; color: #ffffff;'>{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if 'Score' in df.columns:
            avg_score = df['Score'].mean()
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 0.9rem; color: #b0b0b0; margin-bottom: 0.5rem;'>Average Score</div>
                <div style='font-size: 2rem; font-weight: 700; color: #ffffff;'>{avg_score:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'Provider' in df.columns:
            unique_providers = df['Provider'].nunique()
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 0.9rem; color: #b0b0b0; margin-bottom: 0.5rem;'>Providers</div>
                <div style='font-size: 2rem; font-weight: 700; color: #ffffff;'>{unique_providers}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'Total_Battles' in df.columns:
            total_battles = df['Total_Battles'].sum() if df['Total_Battles'].notna().any() else 0
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size: 0.9rem; color: #b0b0b0; margin-bottom: 0.5rem;'>Total Battles</div>
                <div style='font-size: 2rem; font-weight: 700; color: #ffffff;'>{int(total_battles):,}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display leaderboard table
    st.markdown("### 📊 Leaderboard")
    display_cols = ['Rank', 'Model', 'Provider', 'Score']
    if 'Win_Rate' in df.columns:
        display_cols.append('Win_Rate')
    if 'Total_Battles' in df.columns:
        display_cols.append('Total_Battles')
    
    display_cols = [col for col in display_cols if col in df.columns]
    display_df = df[display_cols].copy()
    
    # Format columns
    if 'Score' in display_df.columns:
        display_df['Score'] = display_df['Score'].round(1)
    if 'Win_Rate' in display_df.columns:
        display_df['Win_Rate'] = (display_df['Win_Rate'] * 100).round(1) if display_df['Win_Rate'].dtype != 'object' else display_df['Win_Rate']
        display_df = display_df.rename(columns={'Win_Rate': 'Win Rate (%)'})
    
    st.dataframe(
        display_df.style.background_gradient(subset=['Score'], cmap='RdYlGn'),
        use_container_width=True,
        hide_index=True,
        height=600
    )
    
    # Visualizations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Score' in df.columns and 'Model' in df.columns:
            top_10 = df.head(10)
            fig = px.bar(
                top_10,
                x='Model',
                y='Score',
                color='Provider',
                title='Top 10 Models by Score',
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
    
    with col2:
        if 'Total_Battles' in df.columns and 'Win_Rate' in df.columns and 'Model' in df.columns:
            scatter_df = df[df['Total_Battles'].notna() & df['Win_Rate'].notna()].copy()
            if not scatter_df.empty:
                fig = px.scatter(
                    scatter_df,
                    x='Total_Battles',
                    y='Win_Rate',
                    size='Score',
                    color='Provider',
                    hover_name='Model',
                    title='Win Rate vs Total Battles',
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
            else:
                st.info("Win rate and battle data not available for visualization.")


if __name__ == "__main__":
    show_arena_page()
