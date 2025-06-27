import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

def calculate_spending_summary(df):
    """
    Calculate comprehensive spending summary
    """
    if df is None or df.empty:
        return None
    
    # Basic metrics
    total_spent = df['debit'].sum()
    total_income = df['credit'].sum()
    net_flow = total_income - total_spent
    avg_transaction = df['debit'].mean()
    transaction_count = len(df)
    
    # Category breakdown
    category_spending = df.groupby('category')['debit'].sum().sort_values(ascending=False)
    top_categories = category_spending.head(5)
    
    # Monthly trends
    df['month'] = df['date'].dt.to_period('M')
    monthly_spending = df.groupby('month')['debit'].sum()
    
    # Weekly patterns
    df['day_of_week'] = df['date'].dt.day_name()
    weekly_pattern = df.groupby('day_of_week')['debit'].sum()
    
    return {
        'total_spent': total_spent,
        'total_income': total_income,
        'net_flow': net_flow,
        'avg_transaction': avg_transaction,
        'transaction_count': transaction_count,
        'category_spending': category_spending,
        'top_categories': top_categories,
        'monthly_spending': monthly_spending,
        'weekly_pattern': weekly_pattern
    }

def create_spending_charts(df):
    """
    Create various spending visualization charts
    """
    if df is None or df.empty:
        return None
    
    charts = {}
    
    # Category pie chart
    category_spending = df.groupby('category')['debit'].sum().sort_values(ascending=False)
    fig_pie = px.pie(
        values=category_spending.values,
        names=category_spending.index,
        title="Spending by Category"
    )
    charts['category_pie'] = fig_pie
    
    # Monthly spending trend
    df['month'] = df['date'].dt.to_period('M').astype(str)
    monthly_spending = df.groupby('month')['debit'].sum()
    fig_monthly = px.line(
        x=monthly_spending.index,
        y=monthly_spending.values,
        title="Monthly Spending Trend",
        labels={'x': 'Month', 'y': 'Amount Spent'}
    )
    charts['monthly_trend'] = fig_monthly
    
    # Balance over time
    fig_balance = px.line(
        df,
        x='date',
        y='balance',
        title="Account Balance Over Time"
    )
    charts['balance_trend'] = fig_balance
    
    # Daily spending heatmap
    # df['day'] = df['date'].dt.day
    # df['month_num'] = df['date'].dt.month
    # daily_spending = df.groupby(['month_num', 'day'])['debit'].sum().reset_index()
    
    # if not daily_spending.empty:
    #     pivot_daily = daily_spending.pivot(index='month_num', columns='day', values='debit').fillna(0)
    #     fig_heatmap = px.imshow(
    #         pivot_daily.values,
    #         labels=dict(x="Day of Month", y="Month", color="Spending"),
    #         title="Daily Spending Heatmap",
    #         aspect="auto"
    #     )
    #     charts['spending_heatmap'] = fig_heatmap
    
    # Top spending categories bar chart
    top_categories = category_spending.head(10)
    fig_bar = px.bar(
        x=top_categories.values,
        y=top_categories.index,
        orientation='h',
        title="Top Spending Categories",
        labels={'x': 'Amount Spent', 'y': 'Category'}
    )
    charts['top_categories'] = fig_bar
    
    return charts

def calculate_rfm_metrics(df):
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for customer
    """
    if df is None or df.empty:
        return None
    
    # Calculate reference date (latest transaction date)
    reference_date = df['date'].max()
    
    # Recency: Days since last transaction
    recency = (reference_date - df['date'].max()).days
    
    # Frequency: Number of transactions
    frequency = len(df)
    
    # Monetary: Total amount spent
    monetary = df['debit'].sum()
    
    # Calculate additional metrics
    avg_days_between_transactions = (df['date'].max() - df['date'].min()).days / max(1, frequency - 1) if frequency > 1 else 0
    
    return {
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary,
        'avg_days_between': avg_days_between_transactions,
        'first_transaction': df['date'].min(),
        'last_transaction': df['date'].max()
    }

# def predict_cash_flow(df, days_ahead=30):
#     """
#     Simple cash flow prediction based on historical patterns
#     """
#     if df is None or df.empty:
#         return None
    
#     # Calculate average daily spending
#     df_sorted = df.sort_values('date')
#     total_days = (df_sorted['date'].max() - df_sorted['date'].min()).days
#     if total_days <= 0:
#         return None
    
#     avg_daily_spending = df['debit'].sum() / max(1, total_days)
#     avg_daily_income = df['credit'].sum() / max(1, total_days)
#     net_daily_flow = avg_daily_income - avg_daily_spending
    
#     # Current balance
#     current_balance = df_sorted['balance'].iloc[-1]
    
#     # Predict future balance
#     future_dates = pd.date_range(
#         start=df_sorted['date'].max() + timedelta(days=1),
#         periods=days_ahead,
#         freq='D'
#     )
    
#     predicted_balances = []
#     balance = current_balance
    
#     for i in range(days_ahead):
#         balance += net_daily_flow
#         predicted_balances.append(balance)
    
#     prediction_df = pd.DataFrame({
#         'date': future_dates,
#         'predicted_balance': predicted_balances
#     })
    
#     # Identify potential low balance days
#     low_balance_threshold = 100  # Alert if balance goes below $100
#     low_balance_days = prediction_df[prediction_df['predicted_balance'] < low_balance_threshold]
    
#     return {
#         'prediction_df': prediction_df,
#         'avg_daily_spending': avg_daily_spending,
#         'avg_daily_income': avg_daily_income,
#         'net_daily_flow': net_daily_flow,
#         'current_balance': current_balance,
#         'predicted_balance_30d': predicted_balances[-1] if predicted_balances else current_balance,
#         'low_balance_alerts': low_balance_days
#     }
def predict_cash_flow(df, days_ahead=30, threshold=100):
    """
    Predict cash flow using SARIMA for the next month (days_ahead).
    Returns a dictionary similar to pipeline-ready output.
    """
    if df is None or df.empty:
        return None
 
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['only_date'] = df['date'].dt.date
    df_daily = df.groupby('only_date').last().reset_index()
    df_daily = df_daily[['only_date', 'balance']].rename(columns={'only_date': 'date'})
 
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily.set_index('date', inplace=True)
    df_daily = df_daily.asfreq('D')
    df_daily['balance'].fillna(method='ffill', inplace=True)
 
    # SARIMA model fit
    model = SARIMAX(df_daily['balance'],
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 30),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)
 
    # Forecast
    forecast = results.get_forecast(steps=days_ahead)
    pred = forecast.predicted_mean
    ci = forecast.conf_int()
 
    prediction_df = pd.DataFrame({
        'date': pred.index,
        'predicted_balance': pred.values,
        'lower_bound': ci.iloc[:, 0].values,
        'upper_bound': ci.iloc[:, 1].values
    })
 
    # Alerts for low balance
    low_balance_alerts = prediction_df[prediction_df['predicted_balance'] < threshold]
 
    # Metrics
    current_balance = df_daily['balance'].iloc[-1]
    predicted_balance_30d = pred.values[-1] if not pred.empty else current_balance
    net_daily_flow = (predicted_balance_30d - current_balance) / days_ahead if days_ahead > 0 else 0
 
    spending = df[df['debit'] > 0].groupby(df['date'].dt.date)['debit'].sum()
    income = df[df['credit'] > 0].groupby(df['date'].dt.date)['credit'].sum()
 
    avg_daily_spending = spending.mean() if not spending.empty else 0
    avg_daily_income = income.mean() if not income.empty else 0
 
    return {
        'prediction_df': prediction_df,
        'avg_daily_spending': avg_daily_spending,
        'avg_daily_income': avg_daily_income,
        'net_daily_flow': net_daily_flow,
        'current_balance': current_balance,
        'predicted_balance_30d': predicted_balance_30d,
        'low_balance_alerts': low_balance_alerts
    }
 
# Example usage:
# df = pd.read_csv("brenda_newman.csv")
# forecast_data = predict_cash_flow_sarima(df)
# show_cash_flow_forecast(forecast_data, df)



def generate_insights(df, rfm_metrics, cash_flow_prediction):
    """
    Generate personalized insights based on analysis
    """
    insights = []
    
    if df is None or df.empty:
        return ["No transaction data available for analysis."]
    
    # Spending insights
    total_spent = df['debit'].sum()
    category_spending = df.groupby('category')['debit'].sum().sort_values(ascending=False)
    top_category = category_spending.index[0] if not category_spending.empty else "Unknown"
    top_category_amount = category_spending.iloc[0] if not category_spending.empty else 0
    top_category_pct = (top_category_amount / total_spent * 100) if total_spent > 0 else 0
    
    insights.append(f"💳 Your highest spending category is '{top_category}' accounting for {top_category_pct:.1f}% (${top_category_amount:.2f}) of total expenses.")
    
    # RFM insights
    if rfm_metrics:
        if rfm_metrics['recency'] == 0:
            insights.append("🔥 You made a transaction today - you're an active user!")
        elif rfm_metrics['recency'] <= 7:
            insights.append(f"✅ You made your last transaction {rfm_metrics['recency']} days ago - good activity level.")
        else:
            insights.append(f"⚠️ Your last transaction was {rfm_metrics['recency']} days ago - consider reviewing your account.")
        
        avg_transaction = rfm_metrics['monetary'] / rfm_metrics['frequency'] if rfm_metrics['frequency'] > 0 else 0
        insights.append(f"📊 You average ${avg_transaction:.2f} per transaction with {rfm_metrics['frequency']} total transactions.")
    
    # Cash flow insights
    if cash_flow_prediction:
        if cash_flow_prediction['net_daily_flow'] > 0:
            insights.append(f"📈 Great news! You have positive cash flow of ${cash_flow_prediction['net_daily_flow']:.2f} per day on average.")
        else:
            insights.append(f"📉 Warning: You have negative cash flow of ${abs(cash_flow_prediction['net_daily_flow']):.2f} per day on average.")
        
        predicted_30d = cash_flow_prediction['predicted_balance_30d']
        current = cash_flow_prediction['current_balance']
        
        if predicted_30d > current:
            insights.append(f"🎯 Your balance is expected to increase to ${predicted_30d:.2f} in 30 days.")
        else:
            insights.append(f"⚠️ Your balance may decrease to ${predicted_30d:.2f} in 30 days - consider budget adjustments.")
        
        if not cash_flow_prediction['low_balance_alerts'].empty:
            first_low_date = cash_flow_prediction['low_balance_alerts']['date'].iloc[0]
            insights.append(f"🚨 Low balance alert: Your balance may drop below $100 around {first_low_date.strftime('%Y-%m-%d')}.")
    
    # Seasonal insights
    df['month'] = df['date'].dt.month
    monthly_avg = df.groupby('month')['debit'].mean()
    if not monthly_avg.empty:
        highest_month = monthly_avg.idxmax()
        lowest_month = monthly_avg.idxmin()
        highest_month_name = calendar.month_name[highest_month]
        lowest_month_name = calendar.month_name[lowest_month]
        insights.append(f"📅 You tend to spend most in {highest_month_name} and least in {lowest_month_name}.")
    
    return insights
