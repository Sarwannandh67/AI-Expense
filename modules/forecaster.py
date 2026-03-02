"""
Module 4: Savings Forecasting (Prophet + LSTM fallback)
- Forecasts next 3 months of expenses and savings
- Uses Facebook Prophet for time-series with seasonality
- Computes financial health score
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def prepare_time_series(df: pd.DataFrame, monthly_income: float = 75000) -> pd.DataFrame:
    """Prepare monthly time series for forecasting"""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month_start'] = df['date'].dt.to_period('M').dt.to_timestamp()
    
    monthly = df.groupby('month_start').agg(
        total_expense=('amount', 'sum'),
        transaction_count=('amount', 'count'),
        avg_transaction=('amount', 'mean')
    ).reset_index()
    
    monthly['savings'] = monthly_income - monthly['total_expense']
    monthly['savings_rate'] = monthly['savings'] / monthly_income
    monthly['expense_ratio'] = monthly['total_expense'] / monthly_income
    
    monthly = monthly.rename(columns={'month_start': 'ds', 'total_expense': 'y'})
    
    return monthly


def forecast_with_prophet(monthly_df: pd.DataFrame, periods: int = 3, monthly_income: float = 75000):
    """Forecast using Prophet"""
    try:
        from prophet import Prophet
        
        prophet_df = monthly_df[['ds', 'y']].copy()
        
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.05,
            interval_width=0.8
        )
        
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=periods, freq='MS')
        forecast = model.predict(future)
        
        # Last N rows are predictions
        predictions = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        predictions.columns = ['month', 'predicted_expense', 'lower_bound', 'upper_bound']
        predictions['predicted_savings'] = monthly_income - predictions['predicted_expense']
        predictions['predicted_expense'] = predictions['predicted_expense'].clip(lower=0)
        predictions['predicted_savings'] = predictions['predicted_savings'].clip(lower=0)
        
        return {
            'method': 'Prophet',
            'historical': monthly_df,
            'predictions': predictions,
            'model': model,
            'forecast': forecast
        }
        
    except ImportError:
        return forecast_with_simple_model(monthly_df, periods, monthly_income)


def forecast_with_simple_model(monthly_df: pd.DataFrame, periods: int = 3, monthly_income: float = 75000):
    """Simple trend-based forecasting as fallback"""
    from sklearn.linear_model import LinearRegression
    
    df = monthly_df.copy()
    df['t'] = range(len(df))
    
    X = df[['t']].values
    y = df['y'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_t = len(df)
    future_t = np.array([[last_t + i] for i in range(1, periods + 1)])
    
    preds = model.predict(future_t)
    
    # Generate future months
    last_date = df['ds'].max()
    future_months = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=periods, freq='MS'
    )
    
    std_expense = df['y'].std()
    
    predictions = pd.DataFrame({
        'month': future_months,
        'predicted_expense': preds.clip(min=0),
        'lower_bound': (preds - std_expense).clip(min=0),
        'upper_bound': (preds + std_expense).clip(min=0)
    })
    predictions['predicted_savings'] = (monthly_income - predictions['predicted_expense']).clip(min=0)
    
    return {
        'method': 'Linear Trend',
        'historical': monthly_df,
        'predictions': predictions,
        'model': model,
        'forecast': None
    }


def compute_financial_health_score(df: pd.DataFrame, monthly_income: float = 75000) -> dict:
    """
    Compute a 0-100 Financial Health Score based on:
    - Savings rate (40 pts)
    - Expense volatility (20 pts) 
    - Subscription burden (20 pts)
    - Anomaly frequency (20 pts)
    """
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    monthly_expense = df.groupby('month')['amount'].sum()
    
    savings_rate = 1 - (monthly_expense.mean() / monthly_income)
    savings_score = min(40, max(0, savings_rate * 80))  # 50% savings = 40 pts
    
    # Expense volatility
    cv = monthly_expense.std() / monthly_expense.mean() if monthly_expense.mean() > 0 else 1
    volatility_score = max(0, 20 - cv * 20)
    
    # Subscription burden
    if 'category' in df.columns:
        sub_spend = df[df['category'] == 'Subscriptions']['amount'].sum()
        total_spend = df['amount'].sum()
        sub_ratio = sub_spend / total_spend if total_spend > 0 else 0
        sub_score = max(0, 20 - sub_ratio * 100)  # penalize if >20% on subs
    else:
        sub_score = 10
    
    # Anomaly frequency (placeholder, integrate with anomaly module)
    anomaly_score = 15  # default moderate
    
    total_score = savings_score + volatility_score + sub_score + anomaly_score
    
    if total_score >= 75:
        grade, color, message = "A", "#10b981", "Excellent financial health! Keep it up."
    elif total_score >= 60:
        grade, color, message = "B", "#3b82f6", "Good financial habits with room to improve."
    elif total_score >= 45:
        grade, color, message = "C", "#f59e0b", "Fair health. Focus on reducing variable expenses."
    elif total_score >= 30:
        grade, color, message = "D", "#f97316", "Needs attention. High spending relative to income."
    else:
        grade, color, message = "F", "#ef4444", "Critical. Expenses are unsustainable. Time to budget."
    
    return {
        'score': round(total_score, 1),
        'grade': grade,
        'color': color,
        'message': message,
        'breakdown': {
            'Savings Rate': round(savings_score, 1),
            'Consistency': round(volatility_score, 1),
            'Subscription Control': round(sub_score, 1),
            'Anomaly Control': round(anomaly_score, 1)
        },
        'savings_rate': round(savings_rate * 100, 1)
    }


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from data.generate_data import generate_transactions
    
    df = generate_transactions()
    monthly_df = prepare_time_series(df)
    
    print(f"📈 Historical data: {len(monthly_df)} months")
    print(f"   Avg Monthly Expense: ₹{monthly_df['y'].mean():,.0f}")
    print(f"   Avg Monthly Savings: ₹{monthly_df['savings'].mean():,.0f}")
    
    results = forecast_with_prophet(monthly_df)
    print(f"\n🔮 Forecast Method: {results['method']}")
    print("\n📅 Next 3 Months Forecast:")
    for _, row in results['predictions'].iterrows():
        print(f"   {str(row['month'])[:7]}: Expense ₹{row['predicted_expense']:,.0f} | Savings ₹{row['predicted_savings']:,.0f}")
    
    health = compute_financial_health_score(df)
    print(f"\n💯 Financial Health Score: {health['score']}/100 (Grade: {health['grade']})")
    print(f"   {health['message']}")
    print(f"   Savings Rate: {health['savings_rate']}%")
