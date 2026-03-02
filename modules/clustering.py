"""
Module 2: Spending Behavior Clustering (KMeans)
- Identify spending personalities: Conservative, Balanced, Impulsive, Subscription-heavy
- Features: % income spent, category distribution, frequency, avg transaction value
- Elbow method for optimal K
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


CLUSTER_PROFILES = {
    0: {
        "name": "Conservative Saver",
        "emoji": "🏦",
        "color": "#10b981",
        "description": "Low spending, high savings. Prioritizes essentials over luxuries.",
        "advice": "Great discipline! Consider investing your savings in mutual funds or SIPs."
    },
    1: {
        "name": "Balanced Spender",
        "emoji": "⚖️",
        "color": "#3b82f6",
        "description": "Well-distributed spending across categories with moderate savings.",
        "advice": "You're doing well! Small optimizations in Food & Entertainment could boost savings further."
    },
    2: {
        "name": "Impulsive Spender",
        "emoji": "🛍️",
        "color": "#f59e0b",
        "description": "High frequency of transactions, elevated spending in Entertainment & Others.",
        "advice": "Try the 24-hour rule before purchases. Setting monthly category budgets can help significantly."
    },
    3: {
        "name": "Subscription Heavy",
        "emoji": "📱",
        "color": "#8b5cf6",
        "description": "Multiple active subscriptions consuming significant monthly budget.",
        "advice": "Audit your subscriptions! Canceling 2-3 unused ones could save ₹500-2000/month."
    }
}


def extract_clustering_features(df: pd.DataFrame, monthly_income: float = 75000) -> pd.DataFrame:
    """Extract features per user/month for clustering"""
    features = {}
    
    # Monthly aggregations
    monthly = df.groupby('month').agg(
        total_spent=('amount', 'sum'),
        transaction_count=('amount', 'count'),
        avg_transaction=('amount', 'mean'),
        max_transaction=('amount', 'max')
    ).reset_index()
    
    # Category distribution
    cat_monthly = df.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0)
    cat_monthly = cat_monthly.div(cat_monthly.sum(axis=1), axis=0)  # normalize to %
    
    # Combine
    feature_df = monthly.merge(cat_monthly, on='month', how='left')
    feature_df['pct_income_spent'] = feature_df['total_spent'] / monthly_income
    feature_df['savings'] = monthly_income - feature_df['total_spent']
    feature_df['savings_rate'] = feature_df['savings'] / monthly_income
    
    return feature_df


def run_clustering(feature_df: pd.DataFrame, n_clusters: int = 4):
    """Run KMeans clustering with elbow analysis"""
    
    numeric_cols = ['pct_income_spent', 'transaction_count', 'avg_transaction',
                    'max_transaction', 'savings_rate']
    
    # Add category columns if they exist
    category_cols = ['Food', 'Transport', 'Subscriptions', 'Entertainment', 
                     'Utilities', 'Rent', 'Others']
    available_cats = [c for c in category_cols if c in feature_df.columns]
    
    feature_cols = numeric_cols + available_cats
    feature_cols = [c for c in feature_cols if c in feature_df.columns]
    
    X = feature_df[feature_cols].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Elbow method
    inertias = []
    K_range = range(2, 8)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    
    # Final model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return {
        'clusters': clusters,
        'feature_df': feature_df,
        'X_pca': X_pca,
        'inertias': list(inertias),
        'K_range': list(K_range),
        'scaler': scaler,
        'kmeans': kmeans,
        'feature_cols': feature_cols
    }


def get_cluster_profile(cluster_id: int) -> dict:
    """Get profile for a cluster"""
    return CLUSTER_PROFILES.get(cluster_id % len(CLUSTER_PROFILES), CLUSTER_PROFILES[1])


def get_dominant_cluster(cluster_results: dict) -> dict:
    """Determine the user's dominant spending personality"""
    clusters = cluster_results['clusters']
    # Most common cluster
    from collections import Counter
    dominant = Counter(clusters).most_common(1)[0][0]
    return get_cluster_profile(dominant)


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data.generate_data import generate_transactions
    
    df = generate_transactions()
    df['month'] = df['date'].dt.to_period('M')
    
    feature_df = extract_clustering_features(df)
    results = run_clustering(feature_df)
    
    print("📊 Clustering Results:")
    feature_df['cluster'] = results['clusters']
    
    for c in range(4):
        profile = get_cluster_profile(c)
        months = feature_df[feature_df['cluster'] == c]
        print(f"\n{profile['emoji']} Cluster {c}: {profile['name']}")
        print(f"   Months: {len(months)}")
        if len(months) > 0:
            print(f"   Avg Spent: ₹{months['total_spent'].mean():,.0f}")
            print(f"   Savings Rate: {months['savings_rate'].mean():.1%}")
    
    profile = get_dominant_cluster(results)
    print(f"\n🎯 Dominant Profile: {profile['emoji']} {profile['name']}")
    print(f"   {profile['advice']}")
