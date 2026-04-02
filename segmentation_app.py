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
pass  # Replace with your implementation


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
pass  # Replace with your implementation


# ============================================================
# MAIN AREA: Results Display
# ============================================================

# TODO 13: Data Overview Tab
# If segmentation has been run, create tabs and show:
#   - Basic metrics (total customers, total transactions, date range)
#     using st.columns and st.metric
#   - RFM distribution histograms (Recency, Frequency, Monetary)
#     using matplotlib figures displayed with st.pyplot
pass  # Replace with your implementation


# TODO 14: Cluster Results Tab
# Show:
#   - Cluster summary table (mean R, F, M per cluster + customer count)
#     using engine.get_cluster_summary() and st.dataframe
#   - PCA 2D scatter plot of clusters
#   - Bar chart comparing cluster profiles (mean RFM per cluster)
#   - Bar chart showing customer count per cluster
pass  # Replace with your implementation


# TODO 15: Persona Generation Tab
# If API key is provided:
#   - Show a "Generate Personas" button
#   - When clicked, call engine.generate_personas()
#   - Display each persona in an st.expander
# If no API key, show a message about needing one.
pass  # Replace with your implementation


# ============================================================
# OPTIONAL EXTENSIONS
# ============================================================

# OPTIONAL TODO A: Add data download
# Let users download the RFM table with cluster labels as CSV.

# OPTIONAL TODO B: Add cluster comparison
# Let users select two clusters and show a side-by-side comparison.

# OPTIONAL TODO C: Add a "Try Different K" feature
# Let users re-run with a different K without reloading all data.
