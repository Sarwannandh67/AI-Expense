"""
AI Expense Intelligence System — Streamlit Dashboard
Full product-grade dashboard with all 4 ML modules integrated
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import io
import warnings
warnings.filterwarnings("ignore")

# Add module paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Expense Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    
    .main { background: #0a0e1a; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1f35 0%, #12162a 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    
    .anomaly-card {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .health-score {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
    }
    
    .insight-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1f35;
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading & Caching ──────────────────────────────────────────────────
@st.cache_data
def load_and_process_data(uploaded_file, income: float = 75000):
    from data.generate_data import generate_transactions
    from modules.categorizer import build_categorizer, add_predicted_categories
    from modules.anomaly_detector import detect_anomalies, get_anomaly_summary
    from modules.clustering import extract_clustering_features, run_clustering, get_dominant_cluster
    from modules.forecaster import prepare_time_series, forecast_with_prophet, compute_financial_health_score

    # ── 1. Load Data ──
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = generate_transactions(monthly_income=income)

    # ── 2. Basic Validation ──
    required_cols = ['date', 'description', 'amount']
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert types safely
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    df = df.dropna(subset=['date', 'amount'])

    # ── 3. Module 1: Categorization ──
    cat_results = build_categorizer(df)
    df = add_predicted_categories(df, cat_results['model'])

    # ── 4. Module 2: Clustering ──
    feature_df = extract_clustering_features(df, income)
    cluster_results = run_clustering(feature_df)
    feature_df['cluster'] = cluster_results['clusters']
    dominant_profile = get_dominant_cluster(cluster_results)

    # ── 5. Module 3: Anomaly Detection ──
    df_anomalies, _, _ = detect_anomalies(df)
    anomaly_summary = get_anomaly_summary(df_anomalies)

    # ── 6. Module 4: Forecasting ──
    monthly_df = prepare_time_series(df, income)
    forecast_results = forecast_with_prophet(monthly_df, monthly_income=income)

    # ── 7. Financial Health ──
    health_score = compute_financial_health_score(df, income)

    return {
        'df': df,
        'df_anomalies': df_anomalies,
        'cat_results': cat_results,
        'feature_df': feature_df,
        'cluster_results': cluster_results,
        'dominant_profile': dominant_profile,
        'anomaly_summary': anomaly_summary,
        'monthly_df': monthly_df,
        'forecast_results': forecast_results,
        'health_score': health_score
    }


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💰 AI Expense Intelligence")
    st.markdown("---")
    
    monthly_income = st.slider(
        "Monthly Income (₹)",
        min_value=30000,
        max_value=200000,
        value=75000,
        step=5000,
        format="₹%d"
    )
    
    st.markdown("---")
    st.markdown("**Upload Your CSV**")
    uploaded_file = st.file_uploader(
        "Upload transactions.csv",
        type=['csv'],
        help="CSV with: date, description, amount columns"
    )
    
    st.markdown("---")
    
    # Navigation
    if "page" not in st.session_state:
        st.session_state.page = "📊 Dashboard"

    if st.button("📊 Dashboard", use_container_width=True):
        st.session_state.page = "📊 Dashboard"

    if st.button("🧠 AI Insights", use_container_width=True):
        st.session_state.page = "🧠 AI Insights"

    if st.button("🔮 Forecast", use_container_width=True):
        st.session_state.page = "🔮 Forecast"

    if st.button("📋 Transactions", use_container_width=True):
        st.session_state.page = "📋 Transactions"

    page = st.session_state.page

# ─── Load Data ───────────────────────────────────────────────────────────────
try:
    with st.spinner("🤖 Running AI models..."):
        data = load_and_process_data(uploaded_file, monthly_income)
except Exception as e:
    st.error(f"❌ Error processing file: {e}")
    st.stop()

df = data['df']
df_anomalies = data['df_anomalies']
monthly_df = data['monthly_df']
forecast_results = data['forecast_results']
health = data['health_score']
anomalies = data['anomaly_summary']
profile = data['dominant_profile']
cat_results = data['cat_results']


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("# 📊 Expense Dashboard")
    st.markdown(f"*Analyzing {len(df):,} transactions across {df['date'].dt.to_period('M').nunique()} months*")

    st.markdown("### 📋 Data Health Report")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with col3:
        st.metric("Date Range",
                  f"{df['date'].min().date()} → {df['date'].max().date()}")

    # ── KPI Row ──
    total_spent = df['amount'].sum()
    n_months = df['date'].dt.to_period('M').nunique()
    total_savings = monthly_income * n_months - total_spent
    avg_monthly = total_spent / df['date'].dt.to_period('M').nunique()
    n_anomalies = df_anomalies['is_predicted_anomaly'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Spent (12mo)", f"₹{total_spent/100000:.2f}L", 
                  delta=f"-₹{avg_monthly:,.0f}/mo avg")
    with col2:
        st.metric("Total Savings", f"₹{max(0, total_savings)/100000:.2f}L",
                  delta=f"{health['savings_rate']}% rate")
    with col3:
        st.metric("NLP Model Accuracy", f"{cat_results['accuracy']:.1%}",
                  delta="NLP Categorizer")
    with col4:
        st.metric("Anomalies Detected", f"{n_anomalies}",
                  delta=f"{n_anomalies/len(df):.1%} of txns", delta_color="inverse")
    
    st.markdown("---")
    
    # ── Charts Row 1 ──
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        # Monthly trend
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=monthly_df['ds'].astype(str).str[:7],
            y=monthly_df['y'],
            name='Expenses',
            marker_color='#6366f1',
            opacity=0.8
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_df['ds'].astype(str).str[:7],
            y=monthly_df['savings'].clip(lower=0),
            name='Savings',
            line=dict(color='#10b981', width=2.5),
            mode='lines+markers'
        ))
        fig_trend.add_hline(y=monthly_income, line_dash="dot", 
                            line_color="#f59e0b", annotation_text="Income")
        fig_trend.update_layout(
            title="Monthly Expense vs Savings",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            legend=dict(orientation='h', y=1.1),
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Category pie
        cat_spend = df.groupby('category')['amount'].sum().reset_index()
        
        COLORS = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#3b82f6', '#ef4444']
        
        fig_pie = px.pie(
            cat_spend, values='amount', names='category',
            color_discrete_sequence=COLORS,
            hole=0.5
        )
        fig_pie.update_layout(
            title="Spending by Category",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(t=50, b=20, l=20, r=20),
            showlegend=True,
            legend=dict(font=dict(size=11))
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # ── Charts Row 2 ──
    col1, col2 = st.columns(2)
    
    with col1:
        # Category bar
        cat_monthly = df.copy()
        cat_monthly['month_str'] = df['date'].dt.to_period('M').astype(str)
        cat_monthly_agg = cat_monthly.groupby(['month_str', 'category'])['amount'].sum().reset_index()
        
        fig_bar = px.bar(
            cat_monthly_agg, x='month_str', y='amount', color='category',
            color_discrete_sequence=COLORS,
            title="Category Breakdown by Month"
        )
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=320,
            margin=dict(t=50, b=20, l=20, r=20),
            xaxis_tickangle=-45,
            showlegend=True
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Top transactions
        top_txns = df.nlargest(10, 'amount')[['date', 'description', 'category', 'amount']].copy()
        top_txns['date'] = top_txns['date'].dt.strftime('%Y-%m-%d')
        top_txns['amount'] = top_txns['amount'].apply(lambda x: f"₹{x:,.0f}")
        
        st.markdown("**🔝 Top 10 Transactions**")
        st.dataframe(
            top_txns.reset_index(drop=True),
            use_container_width=True,
            height=290
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2: AI INSIGHTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🧠 AI Insights":
    st.markdown("# 🧠 AI Insights")
    
    tab1, tab2, tab3 = st.tabs(["🏥 Financial Health", "⚠️ Anomalies", "👤 Spending Profile"])
    
    # ── Tab 1: Health Score ──
    with tab1:
        col1, col2 = st.columns([0.4, 0.6])
        
        with col1:
            color = health['color']
            st.markdown(f"""
            <div style="text-align:center; padding: 40px; background: linear-gradient(135deg, #1a1f35, #12162a); 
                        border-radius: 20px; border: 2px solid {color}40;">
                <div style="font-size: 1rem; color: #94a3b8; margin-bottom: 8px;">FINANCIAL HEALTH SCORE</div>
                <div style="font-size: 5rem; font-weight: 800; color: {color}; line-height: 1;">
                    {health['score']:.0f}
                </div>
                <div style="font-size: 2rem; color: {color}; font-weight: 700;">Grade: {health['grade']}</div>
                <div style="color: #cbd5e1; margin-top: 16px; font-size: 0.95rem;">{health['message']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Health breakdown radar/bar
            breakdown = health['breakdown']
            max_scores = {
                'Savings Rate': 40, 'Consistency': 20, 
                'Subscription Control': 20, 'Anomaly Control': 20
            }
            
            fig_health = go.Figure(go.Bar(
                x=list(breakdown.keys()),
                y=list(breakdown.values()),
                marker_color=[color] * len(breakdown),
                text=[f"{v}/{max_scores[k]}" for k, v in breakdown.items()],
                textposition='outside'
            ))
            fig_health.update_layout(
                title="Health Score Breakdown",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                yaxis_title="Points Earned",
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_health, use_container_width=True)
        
        # Actionable tips
        st.markdown("### 💡 Personalized Recommendations")
        
        tips = []
        if health['savings_rate'] < 20:
            tips.append(("🚨 Critical", "Your savings rate is below 20%. Target at least ₹{:,.0f}/month in savings.".format(monthly_income * 0.2)))
        if df[df['category'] == 'Subscriptions']['amount'].sum() / df['amount'].sum() > 0.1:
            tips.append(("⚡ Quick Win", "Subscriptions are >10% of spend. Audit and cancel 2-3 unused ones to save ₹500-1500/month."))
        if df[df['category'] == 'Food']['amount'].sum() / df['amount'].sum() > 0.25:
            tips.append(("🍔 Food Budget", "Food spend is high. Cooking at home 3x/week could save ₹2000-4000/month."))
        tips.append(("📈 Invest", f"Consider SIP of ₹{monthly_income * 0.15:,.0f}/month in index funds for long-term wealth building."))
        
        for severity, tip in tips:
            st.markdown(f"""
            <div class="insight-card">
                <b>{severity}</b>: {tip}
            </div>
            """, unsafe_allow_html=True)
    
    # ── Tab 2: Anomalies ──
    with tab2:
        st.markdown(f"### ⚠️ {len(anomalies)} Anomalous Transactions Detected")
        
        if anomalies:
            # Anomaly scatter plot
            fig_scatter = px.scatter(
                df_anomalies,
                x='date', y='amount',
                color='is_predicted_anomaly',
                color_discrete_map={True: '#ef4444', False: '#6366f1'},
                size='amount',
                size_max=25,
                hover_data=['description', 'category'],
                title="Transaction Timeline — Anomalies Highlighted in Red"
            )
            fig_scatter.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown("### 🚨 Anomaly Details")
            for a in anomalies[:8]:
                st.markdown(f"""
                <div class="anomaly-card">
                    <div style="display: flex; justify-content: space-between;">
                        <b>📅 {a['date']} — {a['description']}</b>
                        <span style="color: #ef4444; font-weight: 700;">₹{a['amount']:,.0f}</span>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 6px;">
                        Category: {a['category']} | ⚠️ {a['explanation']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No significant anomalies detected in your spending!")
    
    # ── Tab 3: Spending Profile ──
    with tab3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a1f35, #12162a); 
                    border-radius: 20px; padding: 30px; text-align: center;
                    border: 1px solid {profile['color']}40; margin-bottom: 24px;">
            <div style="font-size: 4rem;">{profile['emoji']}</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {profile['color']}; margin: 8px 0;">
                {profile['name']}
            </div>
            <div style="color: #cbd5e1;">{profile['description']}</div>
            <div style="color: {profile['color']}; margin-top: 16px; font-weight: 500;">
                💡 {profile['advice']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Category distribution heatmap
        cat_monthly = df.copy()
        cat_monthly['month_str'] = df['date'].dt.to_period('M').astype(str)
        pivot = cat_monthly.groupby(['month_str', 'category'])['amount'].sum().unstack(fill_value=0)
        
        fig_heat = px.imshow(
            pivot.T,
            color_continuous_scale='Purples',
            aspect='auto',
            title="Spending Heatmap by Category & Month"
        )
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            height=350,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
        # NLP model performance
        st.markdown("### 🤖 NLP Categorizer Performance")
        col1, col2, col3 = st.columns(3)
        report = cat_results['classification_report']
        
        for i, (cat, col) in enumerate(zip(['Food', 'Transport', 'Rent'], [col1, col2, col3])):
            if cat in report:
                with col:
                    f1 = report[cat]['f1-score']
                    st.metric(f"F1: {cat}", f"{f1:.2%}")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3: FORECAST
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecast":
    st.markdown("# 🔮 Savings Forecast")
    
    predictions = forecast_results['predictions']
    method = forecast_results['method']
    
    st.markdown(f"*Powered by **{method}** — Predicting next 3 months*")
    
    # Prediction KPIs
    col1, col2, col3 = st.columns(3)
    for i, (_, row) in enumerate(predictions.iterrows()):
        with [col1, col2, col3][i]:
            month_str = str(row['month'])[:7]
            savings = row['predicted_savings']
            expense = row['predicted_expense']
            savings_rate = savings / monthly_income * 100
            
            color = "#10b981" if savings_rate > 20 else "#f59e0b" if savings_rate > 0 else "#ef4444"
            st.markdown(f"""
            <div class="metric-card" style="border-color: {color}40;">
                <div style="font-size: 1.1rem; font-weight: 600; color: #e2e8f0;">{month_str}</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: {color}; margin: 8px 0;">
                    ₹{savings:,.0f}
                </div>
                <div style="color: #94a3b8; font-size: 0.8rem;">Predicted Savings</div>
                <div style="color: #64748b; font-size: 0.75rem; margin-top: 4px;">
                    Expense: ₹{expense:,.0f} ({savings_rate:.0f}% saved)
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Combined forecast chart
    fig_forecast = go.Figure()
    
    # Historical
    hist = forecast_results['historical']
    fig_forecast.add_trace(go.Scatter(
        x=hist['ds'].astype(str).str[:7],
        y=hist['y'],
        name='Historical Expense',
        line=dict(color='#6366f1', width=2),
        mode='lines+markers'
    ))
    fig_forecast.add_trace(go.Scatter(
        x=hist['ds'].astype(str).str[:7],
        y=hist['savings'].clip(lower=0),
        name='Historical Savings',
        line=dict(color='#10b981', width=2),
        mode='lines+markers'
    ))
    
    # Future predictions
    fig_forecast.add_trace(go.Scatter(
        x=predictions['month'].astype(str).str[:7],
        y=predictions['predicted_expense'],
        name='Predicted Expense',
        line=dict(color='#f59e0b', width=2.5, dash='dash'),
        mode='lines+markers',
        marker=dict(size=10, symbol='diamond')
    ))
    fig_forecast.add_trace(go.Scatter(
        x=predictions['month'].astype(str).str[:7],
        y=predictions['predicted_savings'].clip(lower=0),
        name='Predicted Savings',
        line=dict(color='#34d399', width=2.5, dash='dash'),
        mode='lines+markers',
        marker=dict(size=10, symbol='diamond')
    ))
    
    # Confidence band (if available)
    if 'lower_bound' in predictions.columns:
        fig_forecast.add_trace(go.Scatter(
            x=pd.concat([predictions['month'], predictions['month'][::-1]]).astype(str).str[:7],
            y=pd.concat([predictions['upper_bound'], predictions['lower_bound'][::-1]]),
            fill='toself',
            fillcolor='rgba(243, 156, 18, 0.1)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Confidence Range'
        ))
    
    # Income line
    fig_forecast.add_hline(y=monthly_income, line_dash="dot",
                           line_color="#94a3b8", annotation_text=f"Income ₹{monthly_income:,.0f}")
    
    fig_forecast.update_layout(
        title="Expense & Savings Forecast (12 Historical + 3 Predicted Months)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        xaxis_tickangle=-45,
        legend=dict(orientation='h', y=1.15),
        margin=dict(t=80, b=40, l=20, r=20)
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Savings trend bar
    fig_savings = go.Figure(go.Bar(
        x=hist['ds'].astype(str).str[:7],
        y=hist['savings'].clip(lower=0),
        name='Actual Savings',
        marker_color='#10b981'
    ))
    
    fig_savings.add_trace(go.Bar(
        x=predictions['month'].astype(str).str[:7],
        y=predictions['predicted_savings'].clip(lower=0),
        name='Predicted Savings',
        marker_color='#34d399',
        opacity=0.7
    ))
    
    fig_savings.add_hline(
        y=monthly_income * 0.2,
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text="20% Savings Goal"
    )
    
    fig_savings.update_layout(
        title="Monthly Savings — Actual vs Forecast",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=320,
        margin=dict(t=50, b=40, l=20, r=20),
        barmode='group'
    )
    st.plotly_chart(fig_savings, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4: TRANSACTIONS
# ════════════════════════════════════════════════════════════════════════════
elif page == "📋 Transactions":
    st.markdown("# 📋 Transaction Explorer")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        cat_filter = st.multiselect(
            "Filter by Category",
            options=sorted(df['category'].unique()),
            default=[]
        )
    with col2:
        min_amount = st.number_input("Min Amount (₹)", value=0, min_value=0)
    with col3:
        show_anomalies = st.checkbox("Show Anomalies Only", value=False)
    
    display_df = df_anomalies.copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    
    if cat_filter:
        display_df = display_df[display_df['category'].isin(cat_filter)]
    if min_amount > 0:
        display_df = display_df[display_df['amount'] >= min_amount]
    if show_anomalies:
        display_df = display_df[display_df['is_predicted_anomaly'] == True]
    
    display_cols = ['date', 'description', 'category', 'ai_category', 'amount', 
                    'confidence', 'is_predicted_anomaly']
    available_cols = [c for c in display_cols if c in display_df.columns]
    
    st.dataframe(
        display_df[available_cols].sort_values('amount', ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=500
    )
    
    st.markdown(f"*Showing {len(display_df):,} of {len(df):,} transactions*")
    
    # Download button
    csv_buffer = io.StringIO()
    display_df[available_cols].to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Filtered CSV",
        data=csv_buffer.getvalue(),
        file_name="expense_analysis.csv",
        mime="text/csv"
    )

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #475569; font-size: 0.8rem;'>"
    "AI Expense Intelligence System | NLP + KMeans + Isolation Forest + Prophet | Built with ❤️ by @sarwannandh"
    "</div>",
    unsafe_allow_html=True
)
