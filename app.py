"""
full_pipeline_marketing_campaign.py

End-to-end: Load marketing_campaign.csv -> build RFM (from dataset columns) ->
log-transform -> scale -> PCA (viz) -> choose k (elbow + silhouette) ->
KMeans clustering -> cluster profiling -> save outputs + plots.

Assumes file at: /mnt/data/marketing_campaign.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import sys

sns.set(style="whitegrid")

# ---------------------------
# CONFIG
# ---------------------------
INPUT_PATH = "/mnt/data/marketing_campaign.csv"
OUTPUT_DIR = "./output_rfm"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# ---------------------------
# 1) LOAD DATA
# ---------------------------
def load_data(path=INPUT_PATH):
    # marketing_campaign.csv typically uses semicolon delimiter
    if not os.path.exists(path):
        print(f"ERROR: File not found at {path}")
        sys.exit(1)
    try:
        df = pd.read_csv(path, sep=';')
    except Exception:
        df = pd.read_csv(path, sep=',', engine='python')
    df.columns = df.columns.str.strip()
    print(f"Loaded data with shape: {df.shape}")
    return df

# ---------------------------
# 2) PREPROCESS & BUILD RFM
# ---------------------------
def build_rfm_from_marketing(df):
    """
    This dataset (marketing_campaign) already contains:
      - 'Recency' (days since last purchase)
      - multiple 'Num*' columns (counts of purchases by channel)
      - multiple 'Mnt*' columns (amount spent on categories)
    We'll create:
      - Recency: use existing 'Recency' if present else compute
      - Frequency: sum of NumWebPurchases + NumCatalogPurchases + NumStorePurchases + NumDealsPurchases
      - Monetary: sum of all Mnt* columns (mntwines, mntmeatproducts, mntfruits, mntfishproducts, mntsweetproducts, mntgoldprods)
    """
    dfc = df.copy()
    # Normalize column names to lowercase for robustness
    dfc.columns = [c.lower() for c in dfc.columns]
    
    # Identify recency
    if 'recency' in dfc.columns:
        r = 'recency'
    else:
        # As fallback, try to compute recency from dt_customer or last purchase date if available
        if 'dt_customer' in dfc.columns:
            # dt_customer -> use as "customer since" (not last purchase). So fallback to NaN.
            r = None
        else:
            r = None
    
    # Frequency: sum of purchase counts columns (choose available ones)
    freq_cols = [c for c in ['numdealspurchases','numwebpurchases','numcatalogpurchases','numstorepurchases'] if c in dfc.columns]
    if not freq_cols:
        # fallback: try other numeric columns as frequency proxy
        numeric_cols = dfc.select_dtypes(include=[np.number]).columns.tolist()
        freq_cols = numeric_cols[:1]  # very rough fallback
    dfc['frequency'] = dfc[freq_cols].sum(axis=1)
    
    # Monetary: sum of all mnt* columns present
    mnt_cols = [c for c in dfc.columns if c.startswith('mnt')]
    if not mnt_cols:
        # fallback: try to use 'income' as proxy (not ideal)
        if 'income' in dfc.columns:
            dfc['monetary'] = pd.to_numeric(dfc['income'], errors='coerce').fillna(0)
        else:
            # If nothing, set zeros
            dfc['monetary'] = 0.0
    else:
        # ensure numeric then sum
        for c in mnt_cols:
            dfc[c] = pd.to_numeric(dfc[c], errors='coerce').fillna(0)
        dfc['monetary'] = dfc[mnt_cols].sum(axis=1)
    
    # Recency: use existing if present, else try to use 'recency' column if variant in name
    if r == 'recency':
        dfc['recency_rfm'] = pd.to_numeric(dfc['recency'], errors='coerce').fillna(dfc['recency'].median())
    else:
        # If 'recency' not present, try 'lastpurchase' or compute from dates if possible
        if 'lastdate' in dfc.columns:
            try:
                dfc['lastdate'] = pd.to_datetime(dfc['lastdate'], errors='coerce')
                snapshot = dfc['lastdate'].max() + timedelta(days=1)
                dfc['recency_rfm'] = (snapshot - dfc['lastdate']).dt.days.fillna(dfc['lastdate'].median())
            except Exception:
                dfc['recency_rfm'] = dfc['frequency'].apply(lambda x: np.nan)  # fallback
        else:
            # final fallback: use 'recency' name variants
            rec_cols = [c for c in dfc.columns if 'recency' in c]
            if rec_cols:
                dfc['recency_rfm'] = pd.to_numeric(dfc[rec_cols[0]], errors='coerce').fillna(dfc[rec_cols[0]].median())
            else:
                # if nothing, set recency to median value of frequency (not ideal)
                dfc['recency_rfm'] = dfc['frequency'].apply(lambda x: np.nan)
    
    # Build final RFM table
    # If dataset has unique ID column like 'id' or 'customerid', keep it
    id_col = None
    for candidate in ['id','customerid','customer_id','custid','cust_id']:
        if candidate in dfc.columns:
            id_col = candidate
            break
    if id_col is None:
        # create an index id
        dfc = dfc.reset_index().rename(columns={'index':'customer_index'})
        id_col = 'customer_index'
    
    rfm = dfc[[id_col, 'recency_rfm', 'frequency', 'monetary']].copy()
    rfm = rfm.rename(columns={id_col: 'CustomerID', 'recency_rfm':'Recency', 'frequency':'Frequency', 'monetary':'Monetary'})
    
    # Ensure numeric and fill_NA
    rfm['Recency'] = pd.to_numeric(rfm['Recency'], errors='coerce')
    rfm['Frequency'] = pd.to_numeric(rfm['Frequency'], errors='coerce')
    rfm['Monetary'] = pd.to_numeric(rfm['Monetary'], errors='coerce')
    
    # Fill missing Recency with a large value (meaning very old/no purchase)
    median_rec = int(rfm['Recency'].median(skipna=True)) if not rfm['Recency'].isna().all() else 999
    rfm['Recency'] = rfm['Recency'].fillna(median_rec)
    # Fill other NaNs with 0 or median
    rfm['Frequency'] = rfm['Frequency'].fillna(0)
    rfm['Monetary'] = rfm['Monetary'].fillna(0)
    
    # Drop rows with zero frequency and zero monetary optionally (if you want only purchasers)
    # rfm = rfm[~((rfm['Frequency']==0) & (rfm['Monetary']==0))]
    
    print(f"Built RFM with {len(rfm)} customers. (Recency col used from dataset if present)")
    return rfm, dfc

# ---------------------------
# 3) TRANSFORM & SCALE
# ---------------------------
def transform_scale(rfm):
    # log1p to reduce skewness
    rfm['Recency_log'] = np.log1p(rfm['Recency'])
    rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
    rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
    
    features = ['Recency_log','Frequency_log','Monetary_log']
    X = rfm[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, features, scaler

# ---------------------------
# 4) PCA (for viz)
# ---------------------------
def compute_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    return coords, pca

# ---------------------------
# 5) choose k (elbow + silhouette)
# ---------------------------
def evaluate_k(X_scaled, kmin=2, kmax=10):
    inertias = []
    silhouettes = []
    Ks = list(range(kmin, min(kmax, max(2, X_scaled.shape[0]-1)) + 1))
    for k in Ks:
        if k >= X_scaled.shape[0]:
            inertias.append(np.nan)
            silhouettes.append(np.nan)
            continue
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        try:
            silhouettes.append(silhouette_score(X_scaled, labels))
        except Exception:
            silhouettes.append(np.nan)
    return Ks, inertias, silhouettes

# ---------------------------
# 6) final kmeans & profile
# ---------------------------
def run_kmeans(X_scaled, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    centers = km.cluster_centers_
    return labels, centers, km

def profile_clusters(rfm_df):
    # rfm_df expected to have columns: CustomerID, Recency, Frequency, Monetary, Cluster
    profile = rfm_df.groupby('Cluster').agg({
        'Recency':['mean','median'],
        'Frequency':['mean','median'],
        'Monetary':['mean','median'],
        'CustomerID':'count'
    })
    # flatten
    profile.columns = ['_'.join(col).strip() for col in profile.columns.values]
    profile = profile.reset_index().rename(columns={'CustomerID_count':'Count'})
    return profile

# ---------------------------
# 7) plotting helpers
# ---------------------------
def plot_elbow_silhouette(Ks, inertias, silhouettes, outdir=OUTPUT_DIR):
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    axes[0].plot(Ks, inertias, marker='o')
    axes[0].set_title("Elbow Plot (Inertia)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[1].plot(Ks, silhouettes, marker='o', color='orange')
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")
    plt.tight_layout()
    fname = os.path.join(outdir, "elbow_silhouette.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved elbow+silhouette plot -> {fname}")

def plot_pca_clusters(coords, labels, outdir=OUTPUT_DIR):
    dfp = pd.DataFrame(coords, columns=['PC1','PC2'])
    dfp['Cluster'] = labels.astype(str)
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=dfp, x='PC1', y='PC2', hue='Cluster', palette='tab10', s=50, alpha=0.8)
    plt.title("PCA projection colored by KMeans cluster")
    fname = os.path.join(outdir, "pca_clusters.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved PCA cluster plot -> {fname}")

def plot_rfm_boxplots(rfm_df, outdir=OUTPUT_DIR):
    plt.figure(figsize=(14,4))
    plt.subplot(1,3,1)
    sns.boxplot(x='Cluster', y='Recency', data=rfm_df)
    plt.title("Recency by Cluster")
    plt.subplot(1,3,2)
    sns.boxplot(x='Cluster', y='Frequency', data=rfm_df)
    plt.title("Frequency by Cluster")
    plt.subplot(1,3,3)
    sns.boxplot(x='Cluster', y='Monetary', data=rfm_df)
    plt.title("Monetary by Cluster")
    fname = os.path.join(outdir, "rfm_boxplots.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved RFM boxplots -> {fname}")

# ---------------------------
# MAIN pipeline
# ---------------------------
def main():
    df = load_data(INPUT_PATH)
    rfm, df_proc = build_rfm_from_marketing(df)
    
    # Transform + scale
    X_scaled, feat_names, scaler = transform_scale(rfm)
    
    # PCA coords for visualization
    coords, pca = compute_pca(X_scaled, n_components=2)
    rfm['PC1'] = coords[:,0]
    rfm['PC2'] = coords[:,1]
    
    # Evaluate Ks
    Ks, inertias, silhouettes = evaluate_k(X_scaled, kmin=2, kmax=10)
    plot_elbow_silhouette(Ks, inertias, silhouettes)
    
    # Choose k:
    # Heuristic: pick k that maximizes silhouette (ignoring NaNs)
    valid_idx = [i for i, s in enumerate(silhouettes) if not np.isnan(s)]
    if valid_idx:
        best_idx = valid_idx[np.argmax([silhouettes[i] for i in valid_idx])]
        best_k = Ks[best_idx]
    else:
        best_k = 4
    print(f"Chosen k (heuristic by silhouette) = {best_k}")
    
    # Run kmeans
    labels, centers, km_model = run_kmeans(X_scaled, n_clusters=best_k)
    rfm['Cluster'] = labels
    
    # Profile clusters
    profile = profile_clusters(rfm)
    profile.to_csv(os.path.join(OUTPUT_DIR, "cluster_profile.csv"), index=False)
    print(f"Saved cluster profile -> {os.path.join(OUTPUT_DIR, 'cluster_profile.csv')}")
    
    # Save labeled dataset (merge original useful columns)
    # Merge CustomerID back to original df_proc if needed - rfm has CustomerID already
    labeled = rfm.copy()
    labeled.to_csv(os.path.join(OUTPUT_DIR, "rfm_labeled_customers.csv"), index=False)
    print(f"Saved labeled customers -> {os.path.join(OUTPUT_DIR, 'rfm_labeled_customers.csv')}")
    
    # Plots
    plot_pca_clusters(coords, labels)
    plot_rfm_boxplots(rfm)
    
    # Print summary to console
    print("\nCluster counts:")
    print(rfm['Cluster'].value_counts().sort_index())
    print("\nCluster profile (first few rows):")
    print(profile.head())
    
    print("\nAll outputs saved to folder:", OUTPUT_DIR)
    print("Files produced:")
    for f in os.listdir(OUTPUT_DIR):
        print(" -", f)

if __name__ == "__main__":
    main()
