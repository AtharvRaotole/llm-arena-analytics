"""
Arena rankings page for Streamlit frontend.

This module displays Chatbot Arena leaderboards and rankings.
"""

import streamlit as st
from typing import List, Dict, Optional
import pandas as pd
import plotly.express as px


def show_arena_page() -> None:
    """Display the Arena rankings page."""
    st.header("🏆 Chatbot Arena Rankings")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        provider_filter = st.selectbox(
            "Filter by Provider",
            ["All", "OpenAI", "Anthropic", "Google", "Meta", "Mistral AI"]
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Elo Rating", "Win Rate", "Total Battles", "Rank Position"]
        )
    with col3:
        limit = st.slider("Number of results", 10, 100, 50)
    
    # Placeholder data
    sample_data = {
        "Model": ["GPT-4", "Claude 3 Opus", "GPT-4 Turbo", "Claude 3 Sonnet", "Gemini Pro"],
        "Provider": ["OpenAI", "Anthropic", "OpenAI", "Anthropic", "Google"],
        "Elo Rating": [1250, 1240, 1230, 1220, 1210],
        "Win Rate": [0.65, 0.63, 0.61, 0.59, 0.57],
        "Total Battles": [5000, 4800, 4500, 4200, 4000],
        "Rank": [1, 2, 3, 4, 5]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Apply filters
    if provider_filter != "All":
        df = df[df["Provider"] == provider_filter]
    
    # Display table
    st.subheader("Leaderboard")
    st.dataframe(df, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            df.head(10),
            x="Model",
            y="Elo Rating",
            color="Provider",
            title="Top 10 Models by Elo Rating"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df,
            x="Total Battles",
            y="Win Rate",
            size="Elo Rating",
            color="Provider",
            hover_name="Model",
            title="Win Rate vs Total Battles"
        )
        st.plotly_chart(fig, use_container_width=True)

