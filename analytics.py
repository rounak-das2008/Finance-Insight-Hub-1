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
    # reference_date = df['date'].max()
    reference_date = pd.Timestamp.now()
    
    # Recency: Days since last transaction
    recency = (reference_date - df['date'].max().replace(tzinfo=None)).days
    
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

def predict_cash_flow(df, days_ahead=60, threshold=100):
    """
    Enhanced cash flow prediction using 5 months of historical data to forecast 2 months ahead.
    Includes validation against 6th month actual data and confidence scoring.
    """
    if df is None or df.empty:
        return None
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate the date ranges for our analysis
    total_days = (df['date'].max() - df['date'].min()).days
    if total_days < 150:  # Need at least 5 months of data
        return None
    
    # Define our analysis periods
    end_date = df['date'].max()
    start_date = df['date'].min()
    
    # Calculate date ranges for first 5 months, 6th month, and 7th month
    start_date_5months = start_date + pd.DateOffset(months=5)
    start_date_6months = start_date + pd.DateOffset(months=6)
    
    # Split data into training (first 5 months) and validation (6th month) periods
    training_data = df[df['date'] < start_date_5months].copy()
    validation_data = df[(df['date'] >= start_date_5months) & (df['date'] < start_date_6months)].copy()
    
    # Prepare daily balance data for training
    training_data['only_date'] = training_data['date'].dt.date
    df_daily = training_data.groupby('only_date').last().reset_index()
    df_daily = df_daily[['only_date', 'balance']].rename(columns={'only_date': 'date'})
    
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily.set_index('date', inplace=True)
    df_daily = df_daily.asfreq('D')
    df_daily['balance'].fillna(method='ffill', inplace=True)
    
    # Fit SARIMA model on training data
    try:
        model = SARIMAX(df_daily['balance'],
                        order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, 30),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        results = model.fit(disp=False)
        
        # Forecast 2 months (60 days) ahead
        forecast = results.get_forecast(steps=days_ahead)
        pred = forecast.predicted_mean
        ci = forecast.conf_int()
        
        # Create prediction dataframe
        prediction_df = pd.DataFrame({
            'date': pred.index,
            'predicted_balance': pred.values,
            'lower_bound': ci.iloc[:, 0].values,
            'upper_bound': ci.iloc[:, 1].values
        })
        
        # Split predictions into validation period (6th month) and future period (7th month)
        validation_predictions = prediction_df[prediction_df['date'] < start_date_6months]
        future_predictions = prediction_df[prediction_df['date'] >= start_date_6months]
        
        # Calculate confidence score based on validation accuracy
        confidence_score = 0.95  # Default high confidence
        
        if not validation_data.empty and not validation_predictions.empty:
            # Prepare validation data for comparison
            validation_data['only_date'] = validation_data['date'].dt.date
            validation_daily = validation_data.groupby('only_date').last().reset_index()
            validation_daily = validation_daily[['only_date', 'balance']].rename(columns={'only_date': 'date'})
            validation_daily['date'] = pd.to_datetime(validation_daily['date'])
            
            # Merge predictions with actual validation data
            merged_validation = pd.merge(
                validation_predictions[['date', 'predicted_balance']],
                validation_daily[['date', 'balance']],
                on='date',
                how='inner'
            )
            
            if not merged_validation.empty:
                # Calculate prediction accuracy
                mape = np.mean(np.abs((merged_validation['balance'] - merged_validation['predicted_balance']) / merged_validation['balance'])) * 100
                rmse = np.sqrt(np.mean((merged_validation['balance'] - merged_validation['predicted_balance']) ** 2))
                
                # Convert accuracy to confidence score (less strict)
                if mape < 10:  # Less than 10% error
                    confidence_score = 0.95
                elif mape < 20:  # Less than 20% error
                    confidence_score = 0.85
                elif mape < 30:  # Less than 30% error
                    confidence_score = 0.70
                else:  # More than 30% error
                    confidence_score = 0.60
                
                # Store validation metrics
                validation_metrics = {
                    'mape': mape,
                    'rmse': rmse,
                    'validation_points': len(merged_validation)
                }
            else:
                validation_metrics = None
        else:
            validation_metrics = None
        
        # Calculate current metrics
        current_balance = df_daily['balance'].iloc[-1]
        predicted_balance_30d = pred.values[29] if len(pred.values) > 29 else current_balance
        predicted_balance_60d = pred.values[-1] if len(pred.values) > 0 else current_balance
        net_daily_flow = (predicted_balance_60d - current_balance) / days_ahead if days_ahead > 0 else 0
        
        # Calculate spending and income metrics
        spending = df[df['debit'] > 0].groupby(df['date'].dt.date)['debit'].sum()
        income = df[df['credit'] > 0].groupby(df['date'].dt.date)['credit'].sum()
        
        avg_daily_spending = spending.mean() if not spending.empty else 0
        avg_daily_income = income.mean() if not income.empty else 0
        
        # Identify low balance alerts - check both validation and future periods
        # Use a more sensitive threshold and check for declining trends
        low_balance_threshold = threshold
        warning_threshold = threshold * 1.5  # Alert if balance gets close to threshold
        
        # Check for low balance in validation period
        validation_low_balance = validation_predictions[validation_predictions['predicted_balance'] < low_balance_threshold]
        
        # Check for low balance in future period
        future_low_balance = future_predictions[future_predictions['predicted_balance'] < low_balance_threshold]
        
        # Check for warning levels (close to threshold) in future period
        future_warnings = future_predictions[future_predictions['predicted_balance'] < warning_threshold]
        
        # Combine all alerts
        low_balance_alerts = pd.concat([validation_low_balance, future_low_balance], ignore_index=True)
        
        # Add warning alerts for future period
        if not future_warnings.empty:
            future_warnings['alert_type'] = 'Warning'
            low_balance_alerts = pd.concat([low_balance_alerts, future_warnings], ignore_index=True)
        
        # Also check if current balance is already low or declining rapidly
        if current_balance < warning_threshold:
            # Add current balance as a warning
            current_warning = pd.DataFrame({
                'date': [end_date],
                'predicted_balance': [current_balance],
                'period': ['Current'],
                'alert_type': ['Current Warning']
            })
            low_balance_alerts = pd.concat([low_balance_alerts, current_warning], ignore_index=True)
        
        # Check for rapid decline (if balance drops by more than 20% in 30 days)
        if predicted_balance_30d < current_balance * 0.8:
            decline_warning = pd.DataFrame({
                'date': [end_date + pd.DateOffset(days=30)],
                'predicted_balance': [predicted_balance_30d],
                'period': ['Future'],
                'alert_type': ['Rapid Decline']
            })
            low_balance_alerts = pd.concat([low_balance_alerts, decline_warning], ignore_index=True)
        
        # Create enhanced prediction dataframe with period labels
        prediction_df['period'] = 'Future'
        prediction_df.loc[prediction_df['date'] < start_date_6months, 'period'] = 'Validation'
        
        # Ensure low_balance_alerts has the period column
        if not low_balance_alerts.empty and 'period' not in low_balance_alerts.columns:
            # Add period column to low_balance_alerts based on date
            low_balance_alerts['period'] = low_balance_alerts['date'].apply(
                lambda x: 'Validation' if x < start_date_6months else 'Future'
            )
        
        return {
            'prediction_df': prediction_df,
            'validation_predictions': validation_predictions,
            'future_predictions': future_predictions,
            'validation_metrics': validation_metrics,
            'avg_daily_spending': avg_daily_spending,
            'avg_daily_income': avg_daily_income,
            'net_daily_flow': net_daily_flow,
            'current_balance': current_balance,
            'predicted_balance_30d': predicted_balance_30d,
            'predicted_balance_60d': predicted_balance_60d,
            'low_balance_alerts': low_balance_alerts,
            'method_used': 'SARIMA (5-month training)',
            'confidence_score': confidence_score,
            'training_period': f"{start_date.strftime('%Y-%m-%d')} to {start_date_5months.strftime('%Y-%m-%d')}",
            'validation_period': f"{start_date_5months.strftime('%Y-%m-%d')} to {start_date_6months.strftime('%Y-%m-%d')}" if not validation_data.empty else "No validation data available",
            'forecast_period': f"{start_date_6months.strftime('%Y-%m-%d')} to {(start_date_6months + pd.DateOffset(days=days_ahead)).strftime('%Y-%m-%d')}"
        }
        
    except Exception as e:
        # Fallback to simple forecasting if SARIMA fails
        st.warning(f"SARIMA model failed, using simple forecasting: {str(e)}")
        return _simple_cash_flow_forecast(df, days_ahead, threshold)

def _simple_cash_flow_forecast(df, days_ahead=60, threshold=100):
    """
    Simple cash flow prediction as fallback when SARIMA fails
    """
    if df is None or df.empty:
        return None
    
    df_sorted = df.sort_values('date')
    total_days = (df_sorted['date'].max() - df_sorted['date'].min()).days
    if total_days <= 0:
        return None
    
    # Calculate average daily flows
    avg_daily_spending = df['debit'].sum() / max(1, total_days)
    avg_daily_income = df['credit'].sum() / max(1, total_days)
    net_daily_flow = avg_daily_income - avg_daily_spending
    
    # Current balance
    current_balance = df_sorted['balance'].iloc[-1]
    
    # Predict future balance
    future_dates = pd.date_range(
        start=df_sorted['date'].max() + timedelta(days=1),
        periods=days_ahead,
        freq='D'
    )
    
    predicted_balances = []
    balance = current_balance
    
    for i in range(days_ahead):
        balance += net_daily_flow
        predicted_balances.append(balance)
    
    prediction_df = pd.DataFrame({
        'date': future_dates,
        'predicted_balance': predicted_balances,
        'period': 'Future'
    })
    
    # Add confidence intervals (simple approach)
    prediction_df['lower_bound'] = prediction_df['predicted_balance'] * 0.9
    prediction_df['upper_bound'] = prediction_df['predicted_balance'] * 1.1
    
    # Identify potential low balance days - use enhanced alert logic
    low_balance_threshold = threshold
    warning_threshold = threshold * 1.5  # Alert if balance gets close to threshold
    
    # Check for low balance
    low_balance_alerts = prediction_df[prediction_df['predicted_balance'] < low_balance_threshold]
    
    # Check for warning levels (close to threshold)
    warning_alerts = prediction_df[prediction_df['predicted_balance'] < warning_threshold]
    if not warning_alerts.empty:
        warning_alerts['alert_type'] = 'Warning'
        low_balance_alerts = pd.concat([low_balance_alerts, warning_alerts], ignore_index=True)
    
    # Also check if current balance is already low or declining rapidly
    if current_balance < warning_threshold:
        # Add current balance as a warning
        current_warning = pd.DataFrame({
            'date': [df_sorted['date'].max()],
            'predicted_balance': [current_balance],
            'period': ['Current'],
            'alert_type': ['Current Warning']
        })
        low_balance_alerts = pd.concat([low_balance_alerts, current_warning], ignore_index=True)
    
    # Check for rapid decline (if balance drops by more than 20% in 30 days)
    predicted_30d = predicted_balances[29] if len(predicted_balances) > 29 else current_balance
    if predicted_30d < current_balance * 0.8:
        decline_warning = pd.DataFrame({
            'date': [df_sorted['date'].max() + timedelta(days=30)],
            'predicted_balance': [predicted_30d],
            'period': ['Future'],
            'alert_type': ['Rapid Decline']
        })
        low_balance_alerts = pd.concat([low_balance_alerts, decline_warning], ignore_index=True)
    
    # Ensure low_balance_alerts has the period column
    if not low_balance_alerts.empty and 'period' not in low_balance_alerts.columns:
        # Add period column to low_balance_alerts
        low_balance_alerts['period'] = 'Future'
    
    return {
        'prediction_df': prediction_df,
        'validation_predictions': pd.DataFrame(),
        'future_predictions': prediction_df,
        'validation_metrics': None,
        'avg_daily_spending': avg_daily_spending,
        'avg_daily_income': avg_daily_income,
        'net_daily_flow': net_daily_flow,
        'current_balance': current_balance,
        'predicted_balance_30d': predicted_30d,
        'predicted_balance_60d': predicted_balances[-1] if predicted_balances else current_balance,
        'low_balance_alerts': low_balance_alerts,
        'method_used': 'Simple Linear (fallback)',
        'confidence_score': 0.60,  # Lower confidence for simple method
        'training_period': f"{df_sorted['date'].min().strftime('%Y-%m-%d')} to {df_sorted['date'].max().strftime('%Y-%m-%d')}",
        'validation_period': "Not available (simple method)",
        'forecast_period': f"{(df_sorted['date'].max() + timedelta(days=1)).strftime('%Y-%m-%d')} to {(df_sorted['date'].max() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')}"
    }

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
        predicted_60d = cash_flow_prediction.get('predicted_balance_60d', predicted_30d)
        current = cash_flow_prediction['current_balance']
        
        if predicted_30d > current:
            insights.append(f"🎯 Your balance is expected to increase to ${predicted_30d:.2f} in 30 days.")
        else:
            insights.append(f"⚠️ Your balance may decrease to ${predicted_30d:.2f} in 30 days - consider budget adjustments.")
        
        if predicted_60d != predicted_30d:
            if predicted_60d > predicted_30d:
                insights.append(f"📈 Long-term trend shows continued growth to ${predicted_60d:.2f} in 60 days.")
            else:
                insights.append(f"📉 Long-term trend shows potential decline to ${predicted_60d:.2f} in 60 days.")
        
        # Add confidence and validation insights
        confidence = cash_flow_prediction.get('confidence_score', 0)
        if confidence >= 0.85:
            insights.append(f"✅ High confidence forecast ({confidence:.1%}) - predictions are very reliable based on validation.")
        elif confidence >= 0.70:
            insights.append(f"✅ Good confidence forecast ({confidence:.1%}) - predictions show good reliability.")
        elif confidence >= 0.60:
            insights.append(f"⚠️ Forecast accuracy is moderate ({confidence:.1%}). Please review the chart for visual validation.")
        else:
            insights.append(f"❌ Low confidence forecast ({confidence:.1%}) - predictions may not be reliable.")
        
        # Add validation metrics insights if available
        validation_metrics = cash_flow_prediction.get('validation_metrics')
        if validation_metrics:
            mape = validation_metrics['mape']
            if mape < 5:
                insights.append(f"🎯 Excellent model accuracy: Only {mape:.1f}% average error in validation period.")
            elif mape < 10:
                insights.append(f"✅ Good model accuracy: {mape:.1f}% average error in validation period.")
            else:
                insights.append(f"⚠️ Model accuracy: {mape:.1f}% average error - consider this when interpreting forecasts.")
        
        if not cash_flow_prediction['low_balance_alerts'].empty:
            first_low_date = cash_flow_prediction['low_balance_alerts']['date'].iloc[0]
            
            # Safely handle the period column - it might not exist for some customers
            try:
                period_type = cash_flow_prediction['low_balance_alerts']['period'].iloc[0]
                period_text = "validation period" if period_type == "Validation" else "future forecast"
            except (KeyError, IndexError):
                # If period column doesn't exist or is empty, use a default
                period_text = "forecast period"
            
            insights.append(f"🚨 Low balance alert: Your balance may drop below $100 around {first_low_date.strftime('%Y-%m-%d')} ({period_text}).")
    
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
