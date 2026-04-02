"""
segmentation_app.py — Streamlit UI for Customer Segmentation (Student Version)
===============================================================================
A web interface for running customer segmentation, viewing clusters,
and generating AI-powered customer personas.

Run with:  streamlit run segmentation_app.py

Students: Complete the TODO sections to build the full app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Make sure we can import from the current directory
sys.path.insert(0, os.path.dirname(__file__))

# NOTE: Import from solution/ to use completed engine, or from current dir to test yours
# from solution.segmentation_engine import SegmentationEngine
from segmentation_engine import SegmentationEngine


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("👥 Customer Segmentation Dashboard")
st.caption("Segment customers using RFM analysis and K-means clustering")


# ============================================================
# SIDEBAR: Configuration
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Gemini API Key (optional)",
        type="password",
        help="Needed for AI persona generation only",
    )

    data_path = st.text_input(
        "Data file path",
        value="data/online_retail_sample.csv",
    )

    n_clusters = st.slider(
        "Number of clusters (K)",
        min_value=2, max_value=8, value=4,
        help="Try the elbow method to find the best K"
    )

    run_button = st.button("🚀 Run Segmentation", type="primary", use_container_width=True)

    st.divider()


# ============================================================
# SESSION STATE
# ============================================================

# TODO 11: Initialize session state
# Create session state variables for the engine instance and a flag
# indicating whether segmentation has been run.
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'run_complete' not in st.session_state:
    st.session_state.run_complete = False



# ============================================================
# RUN SEGMENTATION
# ============================================================

# TODO 12: Handle the Run button
# When run_button is clicked:
#   1. Create a SegmentationEngine with the data path and optional API key
#   2. Call engine.run_segmentation(n_clusters=n_clusters)
#   3. Store the engine in session state
#   4. Show a success toast/message
# Wrap in a try/except to handle errors gracefully.
if run_button:
    try:
        with st.spinner("Running segmentation pipeline..."):
            engine = SegmentationEngine(filepath=data_path, api_key=api_key)
            engine.run_segmentation(n_clusters=n_clusters)
            st.session_state.engine = engine
            st.session_state.run_complete = True
        st.success("Segmentation completed successfully!")
    except Exception as e:
        st.error(f"An error occurred: {e}")


# ============================================================
# MAIN AREA: Results Display
# ============================================================

# TODO 13: Data Overview Tab
# If segmentation has been run, create tabs and show:
#   - Basic metrics (total customers, total transactions, date range)
#     using st.columns and st.metric
#   - RFM distribution histograms (Recency, Frequency, Monetary)
#     using matplotlib figures displayed with st.pyplot

# TODO 14: Cluster Results Tab
# Show:
#   - Cluster summary table (mean R, F, M per cluster + customer count)
#     using engine.get_cluster_summary() and st.dataframe
#   - PCA 2D scatter plot of clusters
#   - Bar chart comparing cluster profiles (mean RFM per cluster)
#   - Bar chart showing customer count per cluster

# TODO 15: Persona Generation Tab
# If API key is provided:
#   - Show a "Generate Personas" button
#   - When clicked, call engine.generate_personas()
#   - Display each persona in an st.expander
# If no API key, show a message about needing one.

if st.session_state.run_complete:
    engine = st.session_state.engine
    tab1, tab2, tab3 = st.tabs(["Data Overview", "Cluster Results", "AI Personas"])

    with tab1:
        st.header("Data Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(engine.rfm_df))
        col2.metric("Total Transactions", len(engine.clean_df))
        min_date = engine.clean_df['InvoiceDate'].min().strftime('%Y-%m-%d')
        max_date = engine.clean_df['InvoiceDate'].max().strftime('%Y-%m-%d')
        col3.metric("Date Range", f"{min_date} to {max_date}")

        st.subheader("RFM Distributions")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].hist(engine.rfm_df['Recency'], bins=30, color='skyblue', edgecolor='black')
        axes[0].set_title('Recency (Days)')
        axes[1].hist(engine.rfm_df['Frequency'], bins=30, color='lightgreen', edgecolor='black')
        axes[1].set_title('Frequency (Orders)')
        axes[2].hist(engine.rfm_df['Monetary'], bins=30, color='salmon', edgecolor='black')
        axes[2].set_title('Monetary (Spend)')
        st.pyplot(fig)

    with tab2:
        st.header("Cluster Results")
        summary_df = engine.get_cluster_summary()
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("Cluster Profiles (Mean RFM)")
        st.bar_chart(summary_df[['Recency_Mean', 'Frequency_Mean', 'Monetary_Mean']])

        st.subheader("Customer Count per Cluster")
        st.bar_chart(summary_df[['Count']])

        st.subheader("2D PCA Projection")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        components = pca.fit_transform(engine.scaled_data)
        fig_pca, ax_pca = plt.subplots(figsize=(8, 6))
        scatter = ax_pca.scatter(components[:, 0], components[:, 1], c=engine.rfm_df['Cluster'], cmap='Set2', alpha=0.6)
        ax_pca.set_xlabel('Principal Component 1')
        ax_pca.set_ylabel('Principal Component 2')
        ax_pca.legend(*scatter.legend_elements(), title='Cluster')
        st.pyplot(fig_pca)

    with tab3:
        st.header("AI-Generated Personas")
        if not engine.api_key:
            st.warning("Please enter your Gemini API Key in the sidebar to generate personas.")
        else:
            if st.button("Generate Personas", type="primary"):
                with st.spinner("Generating personas with Gemini... (this may take a minute)"):
                    personas = engine.generate_personas()
                    if personas:
                        for cluster_id, persona_text in personas.items():
                            with st.expander(f"Cluster {cluster_id} Persona", expanded=True):
                                st.write(persona_text)
                    else:
                        st.error("Failed to generate personas. Check your API key and network connection.")
else:
    st.info("👈 Configure settings and click 'Run Segmentation' to begin.")


# ============================================================
# OPTIONAL EXTENSIONS
# ============================================================

# OPTIONAL TODO A: Add data download
# Let users download the RFM table with cluster labels as CSV.

# OPTIONAL TODO B: Add cluster comparison
# Let users select two clusters and show a side-by-side comparison.

# OPTIONAL TODO C: Add a "Try Different K" feature
# Let users re-run with a different K without reloading all data.
