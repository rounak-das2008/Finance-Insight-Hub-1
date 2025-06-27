import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_customer_data, get_customer_profile
from analytics import calculate_spending_summary, create_spending_charts, calculate_rfm_metrics, predict_cash_flow, generate_insights
from recommendations import generate_product_recommendations, create_recommendation_summary, get_cross_selling_opportunities
from clustering import get_customer_cluster
from chatbot import render_chatbot_interface, show_chatbot_status
import io
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

def show_customer_dashboard(customer_name):
    """
    Display customer-specific dashboard
    """
    st.title(f"👤 Welcome - {customer_name.replace('_', ' ').title()}")
    
    # Load customer data
    customer_data = load_customer_data(customer_name)
    
    if customer_data is None:
        st.error(f"No transaction data found for customer: {customer_name}")
        st.info("Please ensure your transaction data file is available in the data folder.")
        return
    
    # Get customer profile
    profile = get_customer_profile(customer_name)
    
    # Calculate analytics
    spending_summary = calculate_spending_summary(customer_data)
    rfm_metrics = calculate_rfm_metrics(customer_data)
    cash_flow_prediction = predict_cash_flow(customer_data)
    insights = generate_insights(customer_data, rfm_metrics, cash_flow_prediction)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "💳 Transaction Analysis", 
        "🔮 Cash Flow Forecast", 
        "🎯 Recommendations",
        "🤖 Financial Assistant",
        "📁 Export Data"
    ])
    
    with tab1:
        show_customer_overview(profile, spending_summary, rfm_metrics, insights, customer_data)
    
    with tab2:
        show_transaction_analysis(customer_data, spending_summary)
    
    with tab3:
        forecast_data = predict_cash_flow(customer_data)
        # show_cash_flow_forecast(cash_flow_prediction, customer_data)
        show_cash_flow_forecast(forecast_data, customer_data)

    
    with tab4:
        show_customer_recommendations(customer_name, customer_data)
    
    with tab5:
        show_chatbot_tab(customer_name)
    
    with tab6:
        show_export_options(customer_data, spending_summary)

def show_customer_overview(profile, spending_summary, rfm_metrics, insights, customer_data):
    """
    Show customer overview section
    """
    st.header("📊 Account Overview")
    
    if not profile or not spending_summary or not rfm_metrics:
        st.error("Unable to load customer profile data.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Balance",
            f"${profile['current_balance']:,.2f}"
        )
    
    with col2:
        st.metric(
            "Total Spent",
            f"${spending_summary['total_spent']:,.2f}"
        )
    
    with col3:
        st.metric(
            "Total Transactions",
            f"{spending_summary['transaction_count']:,}"
        )
    
    with col4:
        st.metric(
            "Avg Transaction",
            f"${spending_summary['avg_transaction']:,.2f}"
        )
    
    # RFM Metrics
    st.subheader("📈 Customer Activity Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Days Since Last Transaction",
            f"{rfm_metrics['recency']} days"
        )
    
    with col2:
        st.metric(
            "Transaction Frequency",
            f"{rfm_metrics['frequency']} transactions"
        )
    
    with col3:
        st.metric(
            "Total Monetary Value",
            f"${rfm_metrics['monetary']:,.2f}"
        )
    
    # Account period
    st.info(f"📅 Data Period: {profile['date_range']['start'].strftime('%Y-%m-%d')} to {profile['date_range']['end'].strftime('%Y-%m-%d')}")
    
    # Personalized insights
    st.subheader("💡 Personalized Insights")
    
    for insight in insights:
        st.write(f"• {insight}")

    # Recent transactions
    st.subheader("📋 Recent Transactions")

    # --- Time period filter ---
    min_date = customer_data['date'].min().date()
    max_date = customer_data['date'].max().date()

    # Create 3 columns for filters side by side
    col1, col2, col3 = st.columns(3)

    with col1:
        start_date, end_date = st.date_input(
            "Select date range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    with col2:
        sort_column = st.selectbox(
            "Sort by:",
            options=["Date", "Debit", "Credit"],
            index=0
        )

    with col3:
        sort_order = st.selectbox(
            "Sort order:",
            options=["Descending", "Ascending"],
            index=0
        )

    # Filter by date range
    filtered_transactions = customer_data[
        (customer_data['date'].dt.date >= start_date) &
        (customer_data['date'].dt.date <= end_date)
    ]

    # Determine ascending or descending
    ascending = True if sort_order == "Ascending" else False

    # Sort by selected column
    if sort_column == "Date":
        filtered_transactions = filtered_transactions.sort_values('date', ascending=ascending)
    elif sort_column == "Debit":
        filtered_transactions = filtered_transactions.sort_values('debit', ascending=ascending)
    elif sort_column == "Credit":
        filtered_transactions = filtered_transactions.sort_values('credit', ascending=ascending)

    # Limit to top 10 after sorting
    filtered_transactions = filtered_transactions.head(10)

    # Select and copy required columns
    display_transactions = filtered_transactions[['date', 'category', 'debit', 'credit', 'balance']].copy()

    # Format columns
    display_transactions['date'] = display_transactions['date'].dt.strftime('%Y-%m-%d %H:%M')
    display_transactions['debit'] = display_transactions['debit'].apply(lambda x: f"${x:,.2f}" if x > 0 else "-")
    display_transactions['credit'] = display_transactions['credit'].apply(lambda x: f"${x:,.2f}" if x > 0 else "-")
    display_transactions['balance'] = display_transactions['balance'].apply(lambda x: f"${x:,.2f}")

    # Capitalize first letter of each column name
    display_transactions.columns = [col.capitalize() for col in display_transactions.columns]

    # Show dataframe without the index column (no serial number)
    st.dataframe(display_transactions.reset_index(drop=True), use_container_width=True)


def show_transaction_analysis(customer_data, spending_summary):
    """
    Show detailed transaction analysis
    """
    st.header("💳 Transaction Analysis")
    
    if customer_data is None or spending_summary is None:
        st.error("Unable to load transaction data.")
        return
    
    # Create visualizations
    charts = create_spending_charts(customer_data)
    
    if charts:
        # Category breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            if 'category_pie' in charts:
                st.plotly_chart(charts['category_pie'], use_container_width=True)
        
        with col2:
            if 'top_categories' in charts:
                st.plotly_chart(charts['top_categories'], use_container_width=True)
        
        # Time series analysis
        if 'monthly_trend' in charts:
            st.plotly_chart(charts['monthly_trend'], use_container_width=True)
        
        if 'balance_trend' in charts:
            st.plotly_chart(charts['balance_trend'], use_container_width=True)
        
        # Spending heatmap
        if 'spending_heatmap' in charts:
            st.plotly_chart(charts['spending_heatmap'], use_container_width=True)
    
    # Top spending categories table
    st.subheader("🏆 Top Spending Categories")
    
    top_categories_df = spending_summary['category_spending'].head(10).reset_index()
    top_categories_df.columns = ['Category', 'Amount Spent']
    top_categories_df['Percentage'] = (top_categories_df['Amount Spent'] / spending_summary['total_spent'] * 100).round(2)
    top_categories_df['Amount Spent'] = top_categories_df['Amount Spent'].apply(lambda x: f"${x:,.2f}")
    top_categories_df['Percentage'] = top_categories_df['Percentage'].apply(lambda x: f"{x}%")
    
    st.dataframe(top_categories_df, use_container_width=True)
    

def run_sarima_forecast(df, forecast_days=30, threshold=100):
    """
    Generate SARIMA-based cash flow forecast from customer transaction data.
    Returns a dictionary with metrics and prediction DataFrame.
    """
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['only_date'] = df['date'].dt.date
    df_daily = df.groupby('only_date').last().reset_index()
    df_daily = df_daily[['only_date', 'balance']].rename(columns={'only_date': 'date'})
 
    # Ensure daily frequency and forward-fill
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily.set_index('date', inplace=True)
    df_daily = df_daily.asfreq('D')
    df_daily['balance'].fillna(method='ffill', inplace=True)
 
    # Fit SARIMA
    model = SARIMAX(df_daily['balance'],
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 30),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit(disp=False)
 
    # Forecast
    forecast = results.get_forecast(steps=forecast_days)
    pred = forecast.predicted_mean
    ci = forecast.conf_int()
 
    # Build prediction dataframe
    prediction_df = pd.DataFrame({
        'date': pred.index,
        'predicted_balance': pred.values,
        'lower_bound': ci.iloc[:, 0].values,
        'upper_bound': ci.iloc[:, 1].values
    })
 
    # Identify low balance alerts
    low_balance_alerts = prediction_df[prediction_df['predicted_balance'] < threshold]
 
    # Compute metrics
    current_balance = df_daily['balance'].iloc[-1]
    predicted_balance_30d = pred.values[-1]
    net_daily_flow = (predicted_balance_30d - current_balance) / forecast_days
 
    spending = df[df['debit'] > 0].groupby(df['date'].dt.date)['debit'].sum()
    income = df[df['credit'] > 0].groupby(df['date'].dt.date)['credit'].sum()
 
    avg_daily_spending = spending.mean() if not spending.empty else 0
    avg_daily_income = income.mean() if not income.empty else 0
 
    return {
        'current_balance': current_balance,
        'predicted_balance_30d': predicted_balance_30d,
        'net_daily_flow': net_daily_flow,
        'prediction_df': prediction_df,
        'low_balance_alerts': low_balance_alerts,
        'avg_daily_spending': avg_daily_spending,
        'avg_daily_income': avg_daily_income
    }
 
# Example usage in a pipeline:
# df = pd.read_csv("customer_brenda_newman.csv")
# forecast_data = run_sarima_forecast(df)
# show_cash_flow_forecast(forecast_data, df)

def show_cash_flow_forecast(cash_flow_prediction, customer_data):
    """
    Show cash flow forecast section
    """
    st.header("🔮 Cash Flow Forecast")
    
    if cash_flow_prediction is None:
        st.error("Unable to generate cash flow prediction.")
        return
    
    # Key forecast metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Balance",
            f"${cash_flow_prediction['current_balance']:,.2f}"
        )
    
    with col2:
        color = "normal" if cash_flow_prediction['net_daily_flow'] >= 0 else "inverse"
        st.metric(
            "Daily Net Flow",
            f"${cash_flow_prediction['net_daily_flow']:,.2f}",
            delta=None,
            delta_color=color
        )
    
    with col3:
        predicted_balance = cash_flow_prediction['predicted_balance_30d']
        current_balance = cash_flow_prediction['current_balance']
        delta = predicted_balance - current_balance
        
        st.metric(
            "Predicted Balance (30 days)",
            f"${predicted_balance:,.2f}",
            delta=f"${delta:,.2f}"
        )
    
    # Cash flow chart
    if 'prediction_df' in cash_flow_prediction:
        prediction_df = cash_flow_prediction['prediction_df']

        # Combine historical balance with predictions
        historical_data = customer_data[['date', 'balance']].copy()
        historical_data['type'] = 'Historical'
        
        future_data = prediction_df[['date', 'predicted_balance']].copy()
        future_data = future_data.rename(columns={'predicted_balance': 'balance'})
        future_data['type'] = 'Predicted'
        
        combined_data = pd.concat([historical_data, future_data], ignore_index=True)
        
        # Set custom colors for the plot
        color_map = {'Historical': 'blue', 'Predicted': 'green'}
        
        fig = px.line(
            combined_data,
            x='date',
            y='balance',
            color='type',
            title="Balance Trend: Historical vs Predicted",
            labels={'balance': 'Account Balance', 'date': 'Date'},
            color_discrete_map=color_map  # Set the color mapping here
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Low balance alerts
    if not cash_flow_prediction['low_balance_alerts'].empty:
        st.warning("⚠️ **Low Balance Alerts**")
        
        # low_balance_df = cash_flow_prediction['low_balance_alerts'].copy()
        # low_balance_df['date'] = low_balance_df['date'].dt.strftime('%Y-%m-%d')
        # low_balance_df['predicted_balance'] = low_balance_df['predicted_balance'].apply(lambda x: f"${x:.2f}")
        
        # st.dataframe(low_balance_df, use_container_width=True)
        
        st.info("💡 Consider adjusting your spending or adding funds to avoid low balance situations.")
    else:
        st.success("✅ No low balance alerts for the next 30 days!")
    
    # Spending breakdown
    st.subheader("💰 Daily Spending Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Average Daily Spending",
            f"${cash_flow_prediction['avg_daily_spending']:,.2f}"
        )
    
    with col2:
        st.metric(
            "Average Daily Income",
            f"${cash_flow_prediction['avg_daily_income']:,.2f}"
        )

def show_customer_recommendations(customer_name, customer_data):
    """
    Show product recommendations section
    """
    st.header("🎯 Personalized Recommendations")
    
    # Generate recommendations
    recommendations = generate_product_recommendations(customer_name, customer_data)
    
    if not recommendations:
        st.info("No specific product recommendations available at this time. Continue using our services to receive personalized suggestions!")
        return
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"{i}. {rec['product']['product_name']} - {rec['category']}", expanded=i==1):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Why this fits you:** {rec['reason']}")
                st.write(f"**Product Details:** {rec['product']['description']}")
                st.write(f"**Category:** {rec['product']['category']}")
            
            with col2:
                relevance_pct = int(rec['relevance_score'] * 100)
                st.metric("Relevance", f"{relevance_pct}%")
                
                if st.button(f"Learn More", key=f"learn_more_{i}"):
                    st.info("Contact your relationship manager for more details about this product.")
    
    # Recommendation summary
    st.subheader("📋 Recommendation Summary")
    summary = create_recommendation_summary(recommendations)
    st.markdown(summary)

def show_chatbot_tab(customer_name):
    """Show chatbot interface tab"""
    st.header("🤖 Your Personal Financial Assistant")
    
    st.markdown("""
    Welcome to your AI-powered financial assistant! I can help you with:
    
    - **📊 Spending Analysis**: Get insights into your spending patterns and trends
    - **💡 Budgeting Advice**: Receive personalized tips to improve your financial health
    - **🔮 Forecast Questions**: Ask about future balance predictions and cash flow
    - **🎯 Product Recommendations**: Get advice on which banking products suit your needs
    - **💰 Financial Planning**: General financial advice based on your transaction history
    
    Your data is completely private and secure - I only access your personal financial information to provide better assistance.
    """)
    
    # Show AI service status
    show_chatbot_status()
    
    # Render the chatbot interface
    render_chatbot_interface(customer_name)

def show_export_options(customer_data, spending_summary):
    """
    Show data export options
    """
    st.header("📁 Export Your Data")
    
    if customer_data is None:
        st.error("No data available for export.")
        return
    
    st.write("Download your transaction data and analysis results:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Transaction Data")
        
        # Prepare transaction data for export
        export_data = customer_data.copy()
        export_data['date'] = export_data['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        csv_buffer = io.StringIO()
        export_data.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="Download Transactions (CSV)",
            data=csv_data,
            file_name=f"transactions_{st.session_state.username}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.subheader("📈 Spending Analysis")
        
        if spending_summary:
            # Create analysis summary
            analysis_data = {
                'Metric': [
                    'Total Spent',
                    'Total Income',
                    'Net Flow',
                    'Average Transaction',
                    'Transaction Count',
                    'Top Category',
                    'Top Category Amount'
                ],
                'Value': [
                    f"${spending_summary['total_spent']:,.2f}",
                    f"${spending_summary['total_income']:,.2f}",
                    f"${spending_summary['net_flow']:,.2f}",
                    f"${spending_summary['avg_transaction']:,.2f}",
                    spending_summary['transaction_count'],
                    spending_summary['top_categories'].index[0] if not spending_summary['top_categories'].empty else 'N/A',
                    f"${spending_summary['top_categories'].iloc[0]:,.2f}" if not spending_summary['top_categories'].empty else 'N/A'
                ]
            }
            
            analysis_df = pd.DataFrame(analysis_data)
            
            csv_buffer = io.StringIO()
            analysis_df.to_csv(csv_buffer, index=False)
            analysis_csv = csv_buffer.getvalue()
            
            st.download_button(
                label="Download Analysis (CSV)",
                data=analysis_csv,
                file_name=f"analysis_{st.session_state.username}.csv",
                mime="text/csv"
            )
    
    st.info("💡 **Privacy Note:** Your personal financial data is kept secure and is only accessible to you when logged in.")
