"""
Module 3: Anomaly Detection (Isolation Forest)
- Detects unusual spending: sudden high transactions, rare categories, frequency spikes
- No labeled data needed
- Outputs human-readable anomaly explanations
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def engineer_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features relevant to anomaly detection"""
    df = df.copy()
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    
    # Per-category rolling stats
    cat_stats = df.groupby('category')['amount'].agg(['mean', 'std']).reset_index()
    cat_stats.columns = ['category', 'cat_mean', 'cat_std']
    df = df.merge(cat_stats, on='category', how='left')
    
    # Z-score within category
    df['cat_std'] = df['cat_std'].fillna(1).replace(0, 1)
    df['amount_zscore'] = (df['amount'] - df['cat_mean']) / df['cat_std']
    
    # Daily transaction count (frequency feature)
    daily_counts = df.groupby('date')['amount'].count().reset_index()
    daily_counts.columns = ['date', 'daily_txn_count']
    df = df.merge(daily_counts, on='date', how='left')
    
    # Monthly total spend
    monthly_spend = df.groupby('month')['amount'].sum().reset_index()
    monthly_spend.columns = ['month', 'monthly_total']
    df = df.merge(monthly_spend, on='month', how='left')
    
    # Amount relative to personal average
    personal_avg = df['amount'].mean()
    personal_std = df['amount'].std() if df['amount'].std() > 0 else 1
    df['amount_vs_personal'] = (df['amount'] - personal_avg) / personal_std
    
    # Log amount (captures scale better)
    df['log_amount'] = np.log1p(df['amount'])
    
    return df


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """Run Isolation Forest anomaly detection"""
    df = engineer_anomaly_features(df)
    
    feature_cols = [
        'amount', 'log_amount', 'amount_zscore', 
        'amount_vs_personal', 'daily_txn_count'
    ]
    
    X = df[feature_cols].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
        max_samples='auto'
    )
    
    df['anomaly_score'] = iso_forest.fit_predict(X_scaled)
    df['anomaly_confidence'] = -iso_forest.score_samples(X_scaled)  # higher = more anomalous
    df['is_predicted_anomaly'] = df['anomaly_score'] == -1
    
    return df, iso_forest, scaler


def explain_anomaly(row: pd.Series, df: pd.DataFrame) -> str:
    """Generate human-readable explanation for an anomaly"""
    reasons = []
    
    cat_mean = df[df['category'] == row['category']]['amount'].mean()
    
    if row['amount'] > cat_mean * 2.5:
        multiple = round(row['amount'] / cat_mean, 1)
        reasons.append(f"Amount is {multiple}x your typical {row['category']} spend")
    
    if row['amount_zscore'] > 2.5:
        reasons.append(f"Transaction amount is statistically unusual (z-score: {row['amount_zscore']:.1f})")
    
    if row['daily_txn_count'] > df['daily_txn_count'].quantile(0.95):
        reasons.append(f"High number of transactions on this day ({int(row['daily_txn_count'])} transactions)")
    
    if row['amount'] > df['amount'].quantile(0.95):
        reasons.append(f"Top 5% most expensive transaction (₹{row['amount']:,.0f})")
    
    if not reasons:
        reasons.append("Unusual spending pattern detected by AI model")
    
    return "; ".join(reasons)


def get_anomaly_summary(df_with_anomalies: pd.DataFrame) -> list:
    """Get formatted anomaly summary for dashboard"""
    anomalies = df_with_anomalies[df_with_anomalies['is_predicted_anomaly']].copy()
    anomalies = anomalies.sort_values('anomaly_confidence', ascending=False)
    
    summary = []
    for _, row in anomalies.head(10).iterrows():
        explanation = explain_anomaly(row, df_with_anomalies)
        summary.append({
            'date': str(row['date'].date()) if hasattr(row['date'], 'date') else str(row['date'])[:10],
            'description': row['description'],
            'amount': row['amount'],
            'category': row['category'],
            'confidence': round(row['anomaly_confidence'] * 100, 1),
            'explanation': explanation
        })
    
    return summary


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data.generate_data import generate_transactions
    
    df = generate_transactions()
    df_anomalies, model, scaler = detect_anomalies(df)
    
    n_anomalies = df_anomalies['is_predicted_anomaly'].sum()
    print(f"⚠️  Detected {n_anomalies} anomalies out of {len(df)} transactions ({n_anomalies/len(df):.1%})")
    
    summary = get_anomaly_summary(df_anomalies)
    print("\n🚨 Top Anomalies:")
    for a in summary[:5]:
        print(f"\n  📅 {a['date']} | {a['description']}")
        print(f"     Amount: ₹{a['amount']:,.0f} | Category: {a['category']}")
        print(f"     ⚠️  {a['explanation']}")
