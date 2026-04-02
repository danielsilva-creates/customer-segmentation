# Session 16: Project 3 — Customer Segmentation with AI-Generated Personas

## Overview

Segment online retail customers into distinct groups using **RFM analysis** (Recency, Frequency, Monetary) and **K-means clustering**, then use **Google Gemini** to automatically generate marketing personas for each segment — all wrapped in a Streamlit dashboard.

### What is RFM?

**RFM** stands for **Recency**, **Frequency**, and **Monetary** — three numerical features that summarize a customer's purchasing behavior:

- **Recency (R):** How many days since the customer's last purchase. Lower = more recently active.
- **Frequency (F):** How many separate orders the customer has placed. Higher = more loyal.
- **Monetary (M):** How much the customer has spent in total. Higher = more valuable.

RFM analysis has been an industry-standard technique in marketing, CRM, and e-commerce analytics since the 1990s. Companies like Amazon, Shopify merchants, banks, and subscription services use it daily to segment their customer base and tailor marketing strategies to each group. It works because these three numbers — when combined — capture the behavioral differences that matter most for business decisions: who's engaged, who's loyal, and who's valuable.

---

## Getting Started

### 1. Create a GitHub repository and set up locally

**Step 1: Create the repository on GitHub**

1. Go to [github.com](https://github.com) and sign in
2. Click the **+** icon in the top-right corner → **New repository**
3. Enter repository name: `customer-segmentation`
4. Leave it **Public** (or Private if you prefer)
5. **Do NOT** check "Add a README file" (we already have one locally)
6. **Do NOT** add a `.gitignore` or license (we have our own)
7. Click **Create repository**

**Step 2: Initialize Git locally and push**

```bash
# Navigate to your project folder
cd /path/to/your/session_16

# Initialize Git tracking
git init

# Add all files to staging
git add .

# Create the first commit
git commit -m "Initial commit: project starter code and data"

# Add the GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/customer-segmentation.git

# Push to GitHub
git push -u origin main
```


### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your API key

```bash
cp .env.example .env
```

Open `.env` and replace `your-gemini-api-key-here` with your actual key from [Google AI Studio](https://aistudio.google.com/apikey). The `.env` file is in `.gitignore` so your key stays private.

### 4. Open the notebook

```bash
jupyter notebook Session_16_Customer_Segmentation_Project.ipynb
```

### 5. Work through Parts 1-7

| Part | What You Do |
|------|-------------|
| **Part 1** | Load the dataset, explore and understand the data |
| **Part 2** | Complete TODOs 1-3 in `segmentation_engine.py` (cleaning + RFM) |
| **Part 3** | Complete TODOs 4-6 (scaling + K-means) |
| **Part 4** | Complete TODOs 7-8 (visualization) |
| **Part 5** | Complete TODOs 9-10 (personas + engine class) |
| **Part 6** | Build the Streamlit dashboard (TODOs 11-15 in `segmentation_app.py`) |
| **Part 7** | Reflection & next steps |

### 6. Run the Streamlit app

```bash
streamlit run segmentation_app.py
```

---

## GitHub Workflow

This project is designed to be worked on in a Git repository. Here's a recommended workflow:

### Initial setup

```bash
# Create the repo and push the starter code
git add .
git commit -m "Initial commit: project starter code and data"
git push -u origin main
```

### As you work through each Part

```bash
# After completing Part 2 (cleaning + RFM):
git add segmentation_engine.py
git commit -m "Implement data cleaning and RFM feature engineering"

# After completing Part 3 (scaling + clustering):
git add segmentation_engine.py
git commit -m "Add feature scaling and K-means clustering"

# After completing Part 4 (visualization):
git add segmentation_engine.py
git commit -m "Add elbow/silhouette plots and PCA cluster visualization"

# After completing Part 5 (personas + engine class):
git add segmentation_engine.py
git commit -m "Add Gemini persona generation and SegmentationEngine class"

# After completing Part 6 (Streamlit app):
git add segmentation_app.py
git commit -m "Build Streamlit dashboard with overview, clusters, and personas tabs"

# Push all your work
git push
```

### Branching (optional but good practice)

```bash
# Work on a feature branch
git checkout -b feature/rfm-engine
# ... do your work ...
git add segmentation_engine.py
git commit -m "Implement RFM segmentation pipeline"
git checkout main
git merge feature/rfm-engine
git push
```

---

## Project Structure

```
session_16/
├── README.md                                       ← You are here
├── requirements.txt                                ← pip install -r requirements.txt
├── .env.example                                    ← Copy to .env, add your API key
├── .gitignore                                      ← Keeps .env and caches out of git
│
├── Session_16_Customer_Segmentation_Project.ipynb  ← Main project notebook
│
├── data/
│   └── online_retail_sample.csv                    ← 8,000 transactions, ~800 customers
│
├── segmentation_engine.py                          ← YOUR WORK: 10 TODOs to complete
├── segmentation_app.py                             ← YOUR WORK: 5 TODOs for the dashboard
│
└── solution/                                       ← Reference solutions (try first!)
    ├── segmentation_engine.py                        Complete engine
    └── segmentation_app.py                           Complete Streamlit app
```

---

## The Dataset

A sample of online retail transactions (~8,000 rows) modeled after the [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail):

| Column | Description |
|--------|-------------|
| `InvoiceNo` | Unique invoice identifier |
| `StockCode` | Product code |
| `Description` | Product name |
| `Quantity` | Number of items (negative = return) |
| `InvoiceDate` | Transaction date |
| `UnitPrice` | Price per item |
| `CustomerID` | Customer identifier (~5% missing) |
| `Country` | Customer country |

---

## What You're Building

### The Segmentation Pipeline

```
Raw transactions  →  Clean  →  RFM features  →  Scale  →  K-means  →  Clusters
                                                                          ↓
                                                           Gemini  →  Personas
```

### Why RFM works in practice

- **Recency** predicts engagement — response rates to marketing campaigns drop sharply with time since last purchase. Direct mail marketers discovered this decades ago.
- **Frequency** captures loyalty economics — acquiring a new customer costs 5-7x more than retaining an existing one, so identifying repeat buyers is critical.
- **Monetary** surfaces the Pareto effect — roughly 20% of customers typically drive 80% of revenue.

Real analytics teams use RFM as a starting point, often extending it with average order value, product category diversity, return rate, and inter-purchase time. In this project, you'll implement the core RFM approach and see how K-means discovers natural customer segments from these features.

---

## TODO Summary

### segmentation_engine.py (10 TODOs)

| TODO | Function | What to Implement |
|------|----------|-------------------|
| 1 | `load_and_clean()` | Load CSV, drop NaN, filter, add TotalPrice |
| 2-3 | `build_rfm()` | Reference date calculation + RFM aggregation |
| 4 | `scale_features()` | StandardScaler on RFM columns |
| 5 | `find_optimal_k()` | Loop K values, record inertia + silhouette |
| 6 | `run_kmeans()` | Fit KMeans model |
| 7 | `plot_elbow_and_silhouette()` | 1×2 subplot for K selection |
| 8 | `plot_clusters_2d()` | PCA projection + scatter plot |
| 9 | `generate_persona()` | Gemini prompt for cluster persona |
| 10 | `SegmentationEngine.run_segmentation()` | Wire everything together |

### segmentation_app.py (5 TODOs)

| TODO | What to Implement |
|------|-------------------|
| 11 | Session state initialization |
| 12 | Run button handler |
| 13 | Data overview tab (metrics + RFM histograms) |
| 14 | Cluster results tab (summary + PCA + profiles) |
| 15 | Persona generation tab |

---

## Tips

- **Work in order**: TODOs 1-10 build on each other
- **Test as you go**: The notebook has test cells after each section
- **Commit often**: Save your progress with meaningful commit messages
- **Use the solution**: If stuck for 15+ minutes, peek at `solution/`
- **Restart kernel**: After editing `.py` files, restart Jupyter or use `importlib.reload()`

---

## Prerequisites

- **Session 15**: ML concepts (supervised vs unsupervised, K-means, scaling, PCA)
- **Sessions 11-13**: Python and SQL fundamentals
- **Git/GitHub**: Basic commands (init, add, commit, push)
