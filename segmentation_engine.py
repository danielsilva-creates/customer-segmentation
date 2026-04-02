"""
segmentation_engine.py — Customer Segmentation Engine (Student Version with TODOs)
==================================================================================
Segments customers based on purchasing behavior using RFM analysis and K-means
clustering, then uses Google Gemini to generate persona descriptions.

Usage:
    from segmentation_engine import SegmentationEngine
    engine = SegmentationEngine('data/online_retail_sample.csv')
    engine.run_segmentation(n_clusters=4)
    engine.generate_personas()

API Key:
    Set GOOGLE_API_KEY in a .env file (see .env.example) or pass api_key= directly.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — that's fine, just use env vars directly


# ============================================================
# PART A: Data Loading & Cleaning
# ============================================================

def load_and_clean(filepath):
    """
    Load the Online Retail CSV and clean it for analysis.

    Steps:
    - Load the CSV
    - Drop rows with missing CustomerID
    - Remove rows where Quantity <= 0 (returns/adjustments)
    - Remove rows where UnitPrice <= 0
    - Add a TotalPrice column (Quantity * UnitPrice)
    - Parse InvoiceDate as datetime

    Parameters
    ----------
    filepath : str

    Returns
    -------
    pd.DataFrame
        Cleaned transaction data with TotalPrice column.
    """
    # TODO 1: Load the CSV file and perform all cleaning steps
    # Drop rows missing CustomerID, filter out non-positive Quantity/UnitPrice,
    # add TotalPrice, and parse dates.

    df = pd.read_csv(filepath)
    df = df.dropna(subset=['CustomerID'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df


# ============================================================
# PART B: RFM Feature Engineering
# ============================================================

def build_rfm(df, reference_date=None):
    """
    Create RFM (Recency, Frequency, Monetary) features per customer.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction data (must have CustomerID, InvoiceDate,
        InvoiceNo, TotalPrice columns).
    reference_date : str or pd.Timestamp, optional
        Date to measure recency from. Defaults to max(InvoiceDate) + 1 day.

    Returns
    -------
    pd.DataFrame
        One row per customer with columns: CustomerID, Recency, Frequency, Monetary
    """
    # TODO 2: Calculate the reference date if not provided
    if reference_date is None:
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)


    # TODO 3: Group by CustomerID and compute:
    #   - Recency: days between reference_date and the customer's last purchase
    #   - Frequency: number of unique invoices
    #   - Monetary: total spending (sum of TotalPrice)
    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()

    return rfm




# ============================================================
# PART C: Feature Scaling
# ============================================================

def scale_features(rfm_df):
    """
    Scale RFM features using StandardScaler.

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Must have columns: Recency, Frequency, Monetary

    Returns
    -------
    tuple (np.ndarray, StandardScaler)
        Scaled feature array and the fitted scaler object.
    """
    # TODO 4: Create a StandardScaler, fit it on the three RFM columns,
    # and return the scaled data along with the scaler.
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    return scaled_data, scaler


# ============================================================
# PART D: K-Means Clustering
# ============================================================

def find_optimal_k(scaled_data, k_range=range(2, 9)):
    """
    Run K-means for multiple K values and return inertia + silhouette scores.

    Parameters
    ----------
    scaled_data : np.ndarray
    k_range : range

    Returns
    -------
    pd.DataFrame
        Columns: K, Inertia, Silhouette
    """
    # TODO 5: Loop through each K, run KMeans, and record:
    #   - inertia (km.inertia_)
    #   - silhouette score (silhouette_score(scaled_data, km.labels_))
    # Return as a DataFrame.
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled_data)
        inertia = km.inertia_
        silhouette = silhouette_score(scaled_data, labels)
        results.append({'K': k, 'Inertia': inertia, 'Silhouette': silhouette})
    return pd.DataFrame(results)


def run_kmeans(scaled_data, n_clusters):
    """
    Run K-means with a specific K and return the fitted model.

    Parameters
    ----------
    scaled_data : np.ndarray
    n_clusters : int

    Returns
    -------
    KMeans
        Fitted KMeans model.
    """
    # TODO 6: Create and fit a KMeans model. Use random_state=42, n_init=10.
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(scaled_data)
    return km


# ============================================================
# PART E: Visualization Helpers
# ============================================================

def plot_elbow_and_silhouette(results_df):
    """
    Plot the elbow curve and silhouette scores side by side.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have columns: K, Inertia, Silhouette
    """
    # TODO 7: Create a 1×2 subplot figure.
    #   Left: K vs Inertia (elbow plot)
    #   Right: K vs Silhouette Score
    # Label axes and add titles.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(results_df['K'], results_df['Inertia'], marker='o')
    axes[0].set_title('Elbow Method')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia')
    axes[1].plot(results_df['K'], results_df['Silhouette'], marker='o')
    axes[1].set_title('Silhouette Score')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    plt.tight_layout()
    plt.show




def plot_clusters_2d(scaled_data, labels, rfm_df):
    """
    Use PCA to reduce to 2D and plot clusters as a scatter plot.

    Parameters
    ----------
    scaled_data : np.ndarray
    labels : np.ndarray
        Cluster labels from KMeans.
    rfm_df : pd.DataFrame
        Original RFM data (for hover info).
    """
    # TODO 8: Use PCA(n_components=2) to project the scaled data to 2D.
    # Create a scatter plot colored by cluster label.
    # Add axis labels showing variance explained, a legend, and a title.
    pass  # Replace with your implementation


# ============================================================
# PART F: AI-Powered Persona Generator
# ============================================================

def generate_persona(cluster_stats, cluster_id, client, model='gemini-2.0-flash'):
    """
    Use Gemini to generate a marketing persona for a customer cluster.

    Parameters
    ----------
    cluster_stats : pd.Series
        Mean RFM values and size for one cluster.
    cluster_id : int
    client : genai.Client
    model : str

    Returns
    -------
    str
        Persona description from the AI.
    """
    # TODO 9: Build a prompt that gives the AI the cluster statistics
    # and asks it to generate a creative customer persona with:
    #   - A catchy persona name
    #   - A 2-3 sentence behavioral description
    #   - 2-3 marketing recommendations for this segment
    pass  # Replace with your implementation


# ============================================================
# PART G: The Complete Segmentation Engine
# ============================================================

class SegmentationEngine:
    """
    End-to-end customer segmentation pipeline.

    Parameters
    ----------
    filepath : str
        Path to the Online Retail CSV.
    api_key : str
        Google Gemini API key.
    model : str
        Gemini model name.
    """

    def __init__(self, filepath, api_key=None, model='gemini-2.0-flash'):
        self.filepath = filepath
        self.model = model
        self.api_key = api_key or os.environ.get('GOOGLE_API_KEY')

        # These get populated as the pipeline runs
        self.raw_df = None
        self.clean_df = None
        self.rfm_df = None
        self.scaled_data = None
        self.scaler = None
        self.kmeans_model = None
        self.k_results = None
        self.personas = {}

        # Initialize Gemini client if key provided
        self.client = None
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize Gemini client: {e}")

    def run_segmentation(self, n_clusters=None):
        """
        Run the full segmentation pipeline.

        Steps:
        1. Load and clean the data
        2. Build RFM features
        3. Scale features
        4. If n_clusters is None, find optimal K using elbow/silhouette methods
        5. Run K-means clustering
        6. Add cluster labels to RFM dataframe
        7. Print summary statistics

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters for K-means. If None, optimal K is found first.
        """
        # TODO 10: Implement the run_segmentation method
        # This should:
        #   1. Load and clean the data (store in self.clean_df)
        #   2. Build RFM features (store in self.rfm_df)
        #   3. Scale features (store in self.scaled_data, self.scaler)
        #   4. If n_clusters is None, run find_optimal_k first (store in self.k_results)
        #   5. Run K-means (store in self.kmeans_model)
        #   6. Add cluster labels to self.rfm_df
        #   7. Print a summary of results
        pass  # Replace with your implementation

    def generate_personas(self):
        """Generate AI personas for each cluster."""
        if self.rfm_df is None or 'Cluster' not in self.rfm_df.columns:
            print("Run segmentation first!")
            return {}

        if self.client is None:
            print("Gemini API key required for persona generation.")
            return {}

        cluster_summary = self.rfm_df.groupby('Cluster').agg(
            Recency_Mean=('Recency', 'mean'),
            Frequency_Mean=('Frequency', 'mean'),
            Monetary_Mean=('Monetary', 'mean'),
            Count=('Recency', 'count')
        ).round(1)

        print("\nGenerating personas...\n")
        for cluster_id, stats in cluster_summary.iterrows():
            persona = generate_persona(stats, cluster_id, self.client, self.model)
            self.personas[cluster_id] = persona
            print(f"--- Cluster {cluster_id} ({stats['Count']:.0f} customers) ---")
            print(persona)
            print()

        return self.personas

    def get_cluster_summary(self):
        """Return a summary DataFrame of cluster statistics."""
        if self.rfm_df is None or 'Cluster' not in self.rfm_df.columns:
            print("Run segmentation first!")
            return None
        return self.rfm_df.groupby('Cluster').agg(
            Count=('Recency', 'count'),
            Recency_Mean=('Recency', 'mean'),
            Frequency_Mean=('Frequency', 'mean'),
            Monetary_Mean=('Monetary', 'mean'),
            Monetary_Total=('Monetary', 'sum')
        ).round(1)

    def plot_results(self):
        """Plot elbow/silhouette curves and 2D cluster scatter."""
        if self.k_results is not None:
            plot_elbow_and_silhouette(self.k_results)
        if self.scaled_data is not None and self.kmeans_model is not None:
            plot_clusters_2d(self.scaled_data, self.kmeans_model.labels_, self.rfm_df)
