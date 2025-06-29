import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_loader import load_all_customer_data, get_available_customers
from clustering import prepare_clustering_features, perform_customer_clustering, analyze_clusters, create_cluster_visualizations
from recommendations import analyze_product_interest_trends
from analytics import calculate_spending_summary, calculate_rfm_metrics, predict_cash_flow, generate_insights, create_spending_charts
from data_loader import load_customer_data, get_customer_profile
import io

def show_admin_dashboard():
    """
    Display admin dashboard with aggregated analytics
    """
    st.title("🏦 Bank Admin Dashboard")
    st.write("Comprehensive overview of customer analytics and product opportunities")
    
    # Load all customer data
    with st.spinner("Loading customer data..."):
        all_customer_data = load_all_customer_data()
    
    if not all_customer_data:
        st.error("No customer data available for analysis.")
        st.info("Please ensure customer transaction files are available in the data folder.")
        return
    
    # Perform clustering analysis
    with st.spinner("Performing customer segmentation..."):
        features_df, customer_names = prepare_clustering_features(all_customer_data)
        clustering_results = perform_customer_clustering(features_df) if features_df is not None else None
        cluster_analysis = analyze_clusters(clustering_results) if clustering_results else None
    
    # Create tabs for different admin views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "👥 Customer Segments", 
        "🎯 Product Analytics", 
        "📈 Performance Metrics",
        "🔍 Customer Lookup",
        "📁 Export Reports"
    ])
    
    with tab1:
        show_admin_overview(all_customer_data)
    
    with tab2:
        show_customer_segmentation(clustering_results, cluster_analysis, all_customer_data)
    
    with tab3:
        show_product_analytics(all_customer_data, clustering_results, cluster_analysis)
    
    with tab4:
        show_performance_metrics(all_customer_data, clustering_results, cluster_analysis)
    
    with tab5:
        show_customer_lookup(all_customer_data)
    
    with tab6:
        show_admin_export_options(all_customer_data, clustering_results, cluster_analysis)

def show_admin_overview(all_customer_data):
    """
    Show admin overview with key metrics
    """
    st.header("📊 Business Overview")
    
    # Calculate aggregate metrics
    total_customers = len(all_customer_data)
    total_transactions = 0
    total_volume = 0
    total_balance = 0
    active_customers = 0
    
    customer_metrics = []
    
    for customer_name, data in all_customer_data.items():
        if data is None or data.empty:
            continue
        
        transactions = len(data)
        volume = data['debit'].sum()
        balance = data['balance'].iloc[-1] if not data.empty else 0
        
        # Consider customer active if they had transactions in last 30 days
        latest_transaction = data['date'].max().replace(tzinfo=None)
        days_since_last = (pd.Timestamp.now() - latest_transaction).days
        # days_since_last = (pd.to_datetime("2024-06-30") - latest_transaction).days
        is_active = days_since_last <= 30
        
        total_transactions += transactions
        total_volume += volume
        total_balance += balance
        
        if is_active:
            active_customers += 1
        
        customer_metrics.append({
            'customer': customer_name,
            'transactions': transactions,
            'volume': volume,
            'balance': balance,
            'is_active': is_active,
            'days_since_last': days_since_last
        })
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        st.metric("Active Customers", f"{active_customers:,}")
    
    with col3:
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    with col4:
        st.metric("Total Transaction Volume", f"${total_volume:,.0f}")
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_transactions = total_transactions / max(1, total_customers)
        st.metric("Avg Transactions per Customer", f"{avg_transactions:.1f}")
    
    with col2:
        avg_volume = total_volume / max(1, total_customers)
        st.metric("Avg Volume per Customer", f"${avg_volume:,.0f}")
    
    with col3:
        avg_balance = total_balance / max(1, total_customers)
        st.metric("Avg Customer Balance", f"${avg_balance:,.0f}")
    
    with col4:
        activity_rate = (active_customers / max(1, total_customers)) * 100
        st.metric("Customer Activity Rate", f"{activity_rate:.1f}%")
    
    # Customer distribution charts
    st.subheader("📈 Customer Analytics")
    
    if customer_metrics:
        metrics_df = pd.DataFrame(customer_metrics)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Transaction volume distribution
            fig_volume = px.histogram(
                metrics_df,
                x='volume',
                nbins=20,
                title="Customer Transaction Volume Distribution",
                labels={'volume': 'Transaction Volume ($)', 'count': 'Number of Customers'}
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        
        with col2:
            # Balance distribution
            fig_balance = px.histogram(
                metrics_df,
                x='balance',
                nbins=20,
                title="Customer Balance Distribution",
                labels={'balance': 'Account Balance ($)', 'count': 'Number of Customers'}
            )
            st.plotly_chart(fig_balance, use_container_width=True)
        
        # Activity analysis
        activity_data = metrics_df['is_active'].value_counts()
        fig_activity = px.pie(
            values=activity_data.values,
            names=['Active' if x else 'Inactive' for x in activity_data.index],
            title="Customer Activity Status"
        )
        st.plotly_chart(fig_activity, use_container_width=True)

def show_customer_segmentation(clustering_results, cluster_analysis, all_customer_data):
    """
    Show customer segmentation analysis
    """
    st.header("👥 Customer Segmentation")
    
    if not clustering_results or not cluster_analysis:
        st.warning("Customer segmentation analysis is not available.")
        st.info("Ensure sufficient customer data is available for clustering analysis.")
        return
    
    # Cluster overview
    st.subheader("🎯 Customer Segments Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster sizes
        cluster_data = []
        for cluster_id, info in cluster_analysis.items():
            cluster_data.append({
                'Segment': info['name'],
                'Size': info['size'],
                'Avg Monetary': info['avg_monetary'],
                'Avg Frequency': info['avg_frequency'],
                'Avg Recency': info['avg_recency']
            })
        
        cluster_df = pd.DataFrame(cluster_data)
        st.dataframe(cluster_df, use_container_width=True)
    
    with col2:
        # Cluster size pie chart
        fig_pie = px.pie(
            cluster_df,
            values='Size',
            names='Segment',
            title="Customer Distribution by Segment"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Detailed cluster analysis
    st.subheader("📊 Segment Characteristics")
    
    # Create cluster visualizations
    cluster_visualizations = create_cluster_visualizations(clustering_results, cluster_analysis)
    
    if cluster_visualizations:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'monetary_by_cluster' in cluster_visualizations:
                st.plotly_chart(cluster_visualizations['monetary_by_cluster'], use_container_width=True)
        
        with col2:
            if 'frequency_by_cluster' in cluster_visualizations:
                st.plotly_chart(cluster_visualizations['frequency_by_cluster'], use_container_width=True)
        
        # if 'feature_heatmap' in cluster_visualizations:
        #     st.plotly_chart(cluster_visualizations['feature_heatmap'], use_container_width=True)
    
    # Segment details
    st.subheader("🔍 Segment Details")
    
    for cluster_id, info in cluster_analysis.items():
        with st.expander(f"{info['name']} ({info['size']} customers)", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Characteristics:**")
                for char in info['characteristics']:
                    st.write(f"• {char}")
            
            with col2:
                st.write("**Key Metrics:**")
                st.write(f"• Average Monetary: ${info['avg_monetary']:,.2f}")
                st.write(f"• Average Frequency: {info['avg_frequency']:.1f} transactions")
                st.write(f"• Average Recency: {info['avg_recency']:.1f} days")
            
            # st.write("**Customers in this segment:**")
            # customer_list = ', '.join([name.replace('_', ' ').title() for name in info['customers'][:10]])
            # if len(info['customers']) > 10:
            #     customer_list += f" ... and {len(info['customers']) - 10} more"
            # st.write(customer_list)

def show_product_analytics(all_customer_data, clustering_results, cluster_analysis):
    """
    Show product analytics and recommendations
    """
    st.header("🎯 Product Analytics")
    
    if not all_customer_data:
        st.warning("No customer data available for product analysis.")
        return
    
    # Analyze product interest trends
    product_trends = analyze_product_interest_trends(all_customer_data, clustering_results, cluster_analysis)
    
    if not product_trends:
        st.warning("Product analytics not available.")
        return
    
    # Top product opportunities
    st.subheader("🏆 Top Product Opportunities")
    
    top_products = product_trends['top_products']
    if top_products:
        product_opportunity_data = []
        for product_name, data in top_products[:5]:
            product_opportunity_data.append({
                'Product': product_name,
                'Interested Customers': data['count'],
                'Avg Relevance Score': f"{data['avg_score']:.2f}",
                'Category': data['category'],
                'Market Potential': f"{(data['count'] / len(all_customer_data) * 100):.1f}%"
            })
        
        product_df = pd.DataFrame(product_opportunity_data)
        st.dataframe(product_df, use_container_width=True)
        
        # Product interest chart
        fig_products = px.bar(
            product_df,
            x='Interested Customers',
            y='Product',
            title="Product Interest Levels",
            orientation='h'
        )
        st.plotly_chart(fig_products, use_container_width=True)
    
    # Segment-based product analysis
    st.subheader("📊 Product Interest by Customer Segment")
    
    if cluster_analysis and 'cluster_interest' in product_trends:
        cluster_interest = product_trends['cluster_interest']
        
        for cluster_name, products in cluster_interest.items():
            if products:
                st.write(f"**{cluster_name}:**")
                
                segment_products = []
                for product_name, interest_data in products.items():
                    segment_products.append({
                        'Product': product_name,
                        'Interest Count': interest_data['count'],
                        'Avg Score': f"{interest_data['total_score'] / interest_data['count']:.2f}"
                    })
                
                if segment_products:
                    segment_df = pd.DataFrame(segment_products)
                    segment_df = segment_df.sort_values('Interest Count', ascending=False)
                    st.dataframe(segment_df.head(3), use_container_width=True)
                
                st.write("---")
    
    # Product category analysis
    st.subheader("📈 Product Category Trends")
    
    if 'product_interest' in product_trends:
        category_interest = {}
        for product_name, data in product_trends['product_interest'].items():
            category = data['category']
            if category not in category_interest:
                category_interest[category] = {'count': 0, 'total_score': 0}
            category_interest[category]['count'] += data['count']
            category_interest[category]['total_score'] += data['total_score']
        
        category_data = []
        for category, data in category_interest.items():
            avg_score = data['total_score'] / data['count'] if data['count'] > 0 else 0
            category_data.append({
                'Category': category,
                'Total Interest': data['count'],
                'Average Score': avg_score
            })
        
        if category_data:
            category_df = pd.DataFrame(category_data)
            
            fig_categories = px.bar(
                category_df,
                x='Category',
                y='Total Interest',
                title="Product Category Interest Levels"
            )
            st.plotly_chart(fig_categories, use_container_width=True)

def show_performance_metrics(all_customer_data, clustering_results, cluster_analysis):
    """
    Show performance metrics and KPIs
    """
    st.header("📈 Performance Metrics")
    
    if not all_customer_data:
        st.warning("No data available for performance analysis.")
        return
    
    # Calculate comprehensive metrics
    performance_data = []
    category_totals = {}
    monthly_trends = {}
    
    for customer_name, data in all_customer_data.items():
        if data is None or data.empty:
            continue
        
        spending_summary = calculate_spending_summary(data)
        rfm_metrics = calculate_rfm_metrics(data)
        
        if spending_summary and rfm_metrics:
            performance_data.append({
                'customer': customer_name,
                'total_spent': spending_summary['total_spent'],
                'total_income': spending_summary['total_income'],
                'transaction_count': spending_summary['transaction_count'],
                'avg_transaction': spending_summary['avg_transaction'],
                'recency': rfm_metrics['recency'],
                'frequency': rfm_metrics['frequency'],
                'monetary': rfm_metrics['monetary']
            })
            
            # Aggregate category spending
            for category, amount in spending_summary['category_spending'].items():
                if category not in category_totals:
                    category_totals[category] = 0
                category_totals[category] += amount
            
            # Aggregate monthly trends
            for month, amount in spending_summary['monthly_spending'].items():
                month_str = str(month)
                if month_str not in monthly_trends:
                    monthly_trends[month_str] = 0
                monthly_trends[month_str] += amount
    
    if not performance_data:
        st.error("Unable to calculate performance metrics.")
        return
    
    performance_df = pd.DataFrame(performance_data)
    
    # KPI Summary
    st.subheader("🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = performance_df['total_spent'].sum()
        st.metric("Total Transaction Volume", f"${total_revenue:,.0f}")
    
    with col2:
        avg_customer_value = performance_df['monetary'].mean()
        st.metric("Avg Customer Value", f"${avg_customer_value:,.0f}")
    
    with col3:
        avg_frequency = performance_df['frequency'].mean()
        st.metric("Avg Transaction Frequency", f"{avg_frequency:.1f}")
    
    with col4:
        avg_recency = performance_df['recency'].mean()
        st.metric("Avg Days Since Last Transaction", f"{avg_recency:.1f}")
    
    # Distribution analysis
    st.subheader("📊 Customer Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer value distribution
        fig_value = px.histogram(
            performance_df,
            x='monetary',
            nbins=15,
            title="Customer Value Distribution",
            labels={'monetary': 'Customer Value ($)', 'count': 'Number of Customers'}
        )
        st.plotly_chart(fig_value, use_container_width=True)
    
    with col2:
        # Transaction frequency distribution
        fig_freq = px.histogram(
            performance_df,
            x='frequency',
            nbins=15,
            title="Transaction Frequency Distribution",
            labels={'frequency': 'Number of Transactions', 'count': 'Number of Customers'}
        )
        st.plotly_chart(fig_freq, use_container_width=True)
    
    # Category performance
    st.subheader("💳 Category Performance")
    
    if category_totals:
        category_df = pd.DataFrame([
            {'Category': cat, 'Total Volume': vol}
            for cat, vol in sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
        ])
        
        fig_categories = px.bar(
            category_df.head(10),
            x='Category',
            y='Total Volume',
            title="Top Spending Categories (Bank-wide)"
        )
        fig_categories.update_xaxes(tickangle=45)
        st.plotly_chart(fig_categories, use_container_width=True)
    
    # Monthly trends
    st.subheader("📅 Monthly Trends")
    
    if monthly_trends:
        monthly_df = pd.DataFrame([
            {'Month': month, 'Total Volume': volume}
            for month, volume in sorted(monthly_trends.items())
        ])
        
        fig_monthly = px.line(
            monthly_df,
            x='Month',
            y='Total Volume',
            title="Monthly Transaction Volume Trend"
        )
        fig_monthly.update_xaxes(tickangle=45)
        st.plotly_chart(fig_monthly, use_container_width=True)

def show_admin_export_options(all_customer_data, clustering_results, cluster_analysis):
    """
    Show admin export options
    """
    st.header("📁 Export Reports")
    
    st.write("Export comprehensive reports and analytics data:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👥 Customer Analytics")
        
        if all_customer_data:
            # Prepare customer summary data
            customer_summary = []
            for customer_name, data in all_customer_data.items():
                if data is None or data.empty:
                    continue
                
                spending_summary = calculate_spending_summary(data)
                rfm_metrics = calculate_rfm_metrics(data)
                
                if spending_summary and rfm_metrics:
                    customer_summary.append({
                        'Customer': customer_name.replace('_', ' ').title(),
                        'Total Spent': spending_summary['total_spent'],
                        'Total Income': spending_summary['total_income'],
                        'Net Flow': spending_summary['net_flow'],
                        'Transaction Count': spending_summary['transaction_count'],
                        'Average Transaction': spending_summary['avg_transaction'],
                        'Recency (Days)': rfm_metrics['recency'],
                        'Frequency': rfm_metrics['frequency'],
                        'Monetary Value': rfm_metrics['monetary'],
                        'Current Balance': data['balance'].iloc[-1] if not data.empty else 0
                    })
            
            if customer_summary:
                summary_df = pd.DataFrame(customer_summary)
                csv_buffer = io.StringIO()
                summary_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="Download Customer Analytics",
                    data=csv_buffer.getvalue(),
                    file_name="customer_analytics_report.csv",
                    mime="text/csv"
                )
    
    with col2:
        st.subheader("🎯 Segment Analysis")
        
        if cluster_analysis:
            # Prepare cluster analysis data
            cluster_export = []
            for cluster_id, info in cluster_analysis.items():
                cluster_export.append({
                    'Cluster ID': cluster_id,
                    'Segment Name': info['name'],
                    'Customer Count': info['size'],
                    'Characteristics': ', '.join(info['characteristics']),
                    'Average Monetary': info['avg_monetary'],
                    'Average Frequency': info['avg_frequency'],
                    'Average Recency': info['avg_recency'],
                    'Customer List': ', '.join(info['customers'])
                })
            
            cluster_df = pd.DataFrame(cluster_export)
            csv_buffer = io.StringIO()
            cluster_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="Download Segment Analysis",
                data=csv_buffer.getvalue(),
                file_name="customer_segments_report.csv",
                mime="text/csv"
            )
    
    with col3:
        st.subheader("📊 Performance Report")
        
        if all_customer_data:
            # Create comprehensive performance report
            report_data = {
                'Metric': [
                    'Total Customers',
                    'Total Transaction Volume',
                    'Average Customer Value',
                    'Average Transactions per Customer',
                    'Total Number of Transactions'
                ],
                'Value': []
            }
            
            # Calculate metrics
            total_customers = len([d for d in all_customer_data.values() if d is not None and not d.empty])
            total_volume = sum([d['debit'].sum() for d in all_customer_data.values() if d is not None and not d.empty])
            total_transactions = sum([len(d) for d in all_customer_data.values() if d is not None and not d.empty])
            
            report_data['Value'] = [
                total_customers,
                f"${total_volume:,.2f}",
                f"${total_volume / max(1, total_customers):,.2f}",
                f"{total_transactions / max(1, total_customers):.1f}",
                total_transactions
            ]
            
            report_df = pd.DataFrame(report_data)
            csv_buffer = io.StringIO()
            report_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="Download Performance Report",
                data=csv_buffer.getvalue(),
                file_name="performance_report.csv",
                mime="text/csv"
            )
    
    st.info("💡 **Data Privacy:** All exported reports contain aggregated data only. Individual customer details are protected according to banking regulations.")

def show_customer_lookup(all_customer_data):
    """
    Show customer budget insights lookup for admin support
    """
    st.header("🔍 Customer Budget Insights Lookup")
    
    st.markdown("""
    **Purpose**: This tool helps administrators investigate customer budget forecast issues and provide support.
    
    **When to use**: When customers report problems with their budget forecasts or need assistance with financial insights.
    """)
    
    # Get available customers
    available_customers = get_available_customers()
    
    if not available_customers:
        st.error("No customer data available for lookup.")
        return
    
    # Customer selection
    st.subheader("📋 Select Customer")
    
    # Format customer names for display
    customer_display_names = [name.replace('_', ' ').title() for name in available_customers]
    customer_mapping = dict(zip(customer_display_names, available_customers))
    
    selected_display_name = st.selectbox(
        "Choose customer to analyze:",
        customer_display_names,
        help="Select a customer to view their detailed budget insights and forecasts"
    )
    
    if not selected_display_name:
        st.info("Please select a customer to view their insights.")
        return
    
    selected_customer = customer_mapping[selected_display_name]
    
    # Load customer data
    with st.spinner(f"Loading data for {selected_display_name}..."):
        customer_data = load_customer_data(selected_customer)
        customer_profile = get_customer_profile(selected_customer)
    
    if customer_data is None or customer_data.empty:
        st.error(f"No transaction data found for {selected_display_name}")
        return
    
    # Calculate analytics
    spending_summary = calculate_spending_summary(customer_data)
    rfm_metrics = calculate_rfm_metrics(customer_data)
    cash_flow_prediction = predict_cash_flow(customer_data)
    insights = generate_insights(customer_data, rfm_metrics, cash_flow_prediction)
    charts = create_spending_charts(customer_data)
    
    # Display customer summary
    st.subheader(f"👤 {selected_display_name} - Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Balance",
            f"${customer_profile['current_balance']:,.2f}"
        )
    
    with col2:
        st.metric(
            "Total Spent",
            f"${spending_summary['total_spent']:,.2f}"
        )
    
    with col3:
        st.metric(
            "Transaction Count",
            f"{spending_summary['transaction_count']:,}"
        )
    
    with col4:
        if cash_flow_prediction:
            predicted_balance = cash_flow_prediction['predicted_balance_30d']
            current_balance = cash_flow_prediction['current_balance']
            delta = predicted_balance - current_balance
            st.metric(
                "30-Day Forecast",
                f"${predicted_balance:,.2f}",
                delta=f"${delta:,.2f}"
            )
    
    # Customer issues and flags
    st.subheader("⚠️ Potential Issues & Flags")
    
    issues = []
    
    # Check for potential issues
    if cash_flow_prediction:
        # Enhanced low balance alert detection
        if not cash_flow_prediction['low_balance_alerts'].empty:
            low_balance_df = cash_flow_prediction['low_balance_alerts']
            
            # Check for critical alerts (below $100)
            critical_alerts = low_balance_df[low_balance_df['predicted_balance'] < 100]
            if not critical_alerts.empty:
                issues.append("🔴 **Critical Low Balance Alert**: Customer may experience balance below $100")
            
            # Check for warning alerts (below $150)
            warning_alerts = low_balance_df[
                (low_balance_df['predicted_balance'] >= 100) & 
                (low_balance_df['predicted_balance'] < 150)
            ]
            if not warning_alerts.empty:
                issues.append("🟡 **Low Balance Warning**: Customer balance may drop below $150")
            
            # Check for rapid decline alerts
            if 'alert_type' in low_balance_df.columns:
                rapid_decline = low_balance_df[low_balance_df['alert_type'] == 'Rapid Decline']
                if not rapid_decline.empty:
                    issues.append("🟡 **Rapid Balance Decline**: Customer balance is declining rapidly (20%+ in 30 days)")
        
        if cash_flow_prediction['net_daily_flow'] < -50:
            issues.append(f"🟡 **High Daily Spending**: Net daily outflow of ${abs(cash_flow_prediction['net_daily_flow']):.2f}")
        
        confidence = cash_flow_prediction.get('confidence_score', 0)
        if confidence < 0.6:
            issues.append(f"🟡 **Low Forecast Confidence**: Prediction confidence is {confidence:.1%}")
        
        # Check for negative cash flow trends
        predicted_30d = cash_flow_prediction.get('predicted_balance_30d', 0)
        current_balance = cash_flow_prediction.get('current_balance', 0)
        if predicted_30d < current_balance * 0.9:  # 10% decline
            issues.append(f"🟡 **Declining Balance Trend**: Balance expected to decrease by {((current_balance - predicted_30d) / current_balance * 100):.1f}% in 30 days")
    
    if rfm_metrics and rfm_metrics['recency'] > 30:
        issues.append(f"🟡 **Inactive Customer**: Last transaction was {rfm_metrics['recency']} days ago")
    
    if issues:
        for issue in issues:
            st.warning(issue)
    else:
        st.success("✅ No immediate issues detected for this customer")
    
    # Budget Forecast Visualization
    st.subheader("📈 Enhanced Budget Forecast Analysis")
    
    if cash_flow_prediction and 'prediction_df' in cash_flow_prediction:
        prediction_df = cash_flow_prediction['prediction_df']
        
        # Prepare historical data (first 5 months)
        end_date = customer_data['date'].max()
        start_date = customer_data['date'].min()
        start_date_5months = start_date + pd.DateOffset(months=5)
        start_date_6months = start_date + pd.DateOffset(months=6)
        
        historical_data = customer_data[customer_data['date'] < start_date_5months][['date', 'balance']].copy()
        historical_data['type'] = 'Historical (First 5 months)'
        
        # Prepare 6th month actual data (validation period)
        sixth_month_actual = customer_data[
            (customer_data['date'] >= start_date_5months) & 
            (customer_data['date'] < start_date_6months)
        ][['date', 'balance']].copy()
        sixth_month_actual['type'] = '6th Month (Actual)'
        
        # Prepare validation data (6th month predicted vs actual)
        validation_predictions = cash_flow_prediction.get('validation_predictions', pd.DataFrame())
        if not validation_predictions.empty:
            validation_predictions = validation_predictions.copy()
            validation_predictions['type'] = '6th Month (Forecast)'
            validation_predictions = validation_predictions.rename(columns={'predicted_balance': 'balance'})
        
        # Prepare future predictions (7th month)
        future_predictions = cash_flow_prediction.get('future_predictions', pd.DataFrame())
        if not future_predictions.empty:
            future_predictions = future_predictions.copy()
            future_predictions['type'] = '7th Month (Forecast)'
            future_predictions = future_predictions.rename(columns={'predicted_balance': 'balance'})
        
        # Combine all data
        combined_data = [historical_data]
        if not sixth_month_actual.empty:
            combined_data.append(sixth_month_actual)
        if not validation_predictions.empty:
            combined_data.append(validation_predictions)
        if not future_predictions.empty:
            combined_data.append(future_predictions)
        
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Create enhanced visualization
        fig = go.Figure()
        
        # Historical data
        hist_data = combined_df[combined_df['type'] == 'Historical (First 5 months)']
        if not hist_data.empty:
            fig.add_trace(go.Scatter(
                x=hist_data['date'],
                y=hist_data['balance'],
                mode='lines',
                name='Historical (First 5 months)',
                line=dict(color='blue', width=3),
                hovertemplate='<b>Historical</b><br>Date: %{x}<br>Balance: $%{y:,.2f}<extra></extra>'
            ))
        
        sixth_month_actual_data = combined_df[combined_df['type'] == '6th Month (Actual)']
        if not sixth_month_actual_data.empty:
            fig.add_trace(go.Scatter(
                x=sixth_month_actual_data['date'],
                y=sixth_month_actual_data['balance'],
                mode='lines',
                name='6th Month (Actual)',
                line=dict(color='purple', width=3),
                hovertemplate='<b>6th Month (Actual)</b><br>Date: %{x}<br>Balance: $%{y:,.2f}<extra></extra>'
            ))
        
        # Validation period
        val_data = combined_df[combined_df['type'] == '6th Month (Forecast)']
        if not val_data.empty:
            fig.add_trace(go.Scatter(
                x=val_data['date'],
                y=val_data['balance'],
                mode='lines',
                name='6th Month (Forecast)',
                line=dict(color='orange', width=3, dash='dash'),
                hovertemplate='<b>6th Month (Forecast)</b><br>Date: %{x}<br>Predicted: $%{y:,.2f}<extra></extra>'
            ))
        
        # Future forecast
        future_data = combined_df[combined_df['type'] == '7th Month (Forecast)']
        if not future_data.empty:
            fig.add_trace(go.Scatter(
                x=future_data['date'],
                y=future_data['balance'],
                mode='lines',
                name='7th Month (Forecast)',
                line=dict(color='green', width=3, dash='dot'),
                hovertemplate='<b>7th Month (Forecast)</b><br>Date: %{x}<br>Predicted: $%{y:,.2f}<extra></extra>'
            ))
            
            # Add confidence intervals for future forecast
            if 'lower_bound' in future_data.columns and 'upper_bound' in future_data.columns:
                fig.add_trace(go.Scatter(
                    x=future_data['date'],
                    y=future_data['upper_bound'],
                    mode='lines',
                    name='Confidence Upper',
                line=dict(color='lightgreen', width=1),
                    showlegend=False,
                    hovertemplate='<b>Upper Bound</b><br>Date: %{x}<br>Balance: $%{y:,.2f}<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                    x=future_data['date'],
                    y=future_data['lower_bound'],
                mode='lines',
                    name='Confidence Lower',
                line=dict(color='lightgreen', width=1),
                fill='tonexty',
                    fillcolor='rgba(0,255,0,0.1)',
                    showlegend=False,
                    hovertemplate='<b>Lower Bound</b><br>Date: %{x}<br>Balance: $%{y:,.2f}<extra></extra>'
            ))
        
        # Add low balance threshold line
        min_date = combined_df['date'].min()
        max_date = combined_df['date'].max()
        fig.add_trace(go.Scatter(
            x=[min_date, max_date],
            y=[100, 100],
            mode='lines',
            name='Low Balance Threshold',
            line=dict(color='red', width=2, dash='dot'),
            hovertemplate='<b>Low Balance Alert</b><br>Threshold: $100<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Enhanced Budget Forecast for {selected_display_name}: First 5 Months Historical + 6th Month (Actual vs Forecast) + 7th Month (Forecast)",
            xaxis_title="Date",
            yaxis_title="Account Balance ($)",
            hovermode='x unified',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display validation metrics if available
        validation_metrics = cash_flow_prediction.get('validation_metrics')
        if validation_metrics:
            # st.subheader("📊 Validation Metrics (6th Month: Actual vs Forecast)")
            
            # col1, col2, col3 = st.columns(3)
            # with col1:
            #     st.metric("Mean Absolute % Error", f"{validation_metrics['mape']:.2f}%")
            # with col2:
            #     st.metric("Root Mean Square Error", f"${validation_metrics['rmse']:.2f}")
            # with col3:
            #     st.metric("Validation Points", f"{validation_metrics['validation_points']}")
            
            # Interpretation of validation results
            if validation_metrics['mape'] < 5:
                st.success("✅ Excellent forecast accuracy! The model performed very well on validation data.")
            elif validation_metrics['mape'] < 10:
                st.info("✅ Good forecast accuracy. The model shows reliable predictions.")
            elif validation_metrics['mape'] < 20:
                st.warning("⚠️ Moderate forecast accuracy. Consider the predictions with caution.")
            else:
                # st.error("❌ Low forecast accuracy. Predictions may not be reliable.")
                st.warning("⚠️ Moderate forecast accuracy. Consider the predictions with caution.")

        
        # Display forecast periods
        st.subheader("📅 Forecast Periods")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Training Period:** {cash_flow_prediction['training_period']}")
        with col2:
            st.info(f"**Validation Period:** {cash_flow_prediction['validation_period']} (6th month - actual vs forecast)")
        with col3:
            st.info(f"**Future Forecast:** {cash_flow_prediction['forecast_period']} (7th month)")
        
        # Forecast method and confidence
        method_used = cash_flow_prediction.get('method_used', 'Unknown')
        confidence = cash_flow_prediction.get('confidence_score', 0.5)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🎯 **Forecast Method**: {method_used}")
        with col2:
            st.info(f"📊 **Confidence Level**: {confidence:.1%}")
    
    # Spending patterns analysis
    st.subheader("💳 Spending Pattern Analysis")
    
    if charts:
        col1, col2 = st.columns(2)
        
        with col1:
            if 'category_pie' in charts:
                st.plotly_chart(charts['category_pie'], use_container_width=True)
        
        with col2:
            if 'monthly_trend' in charts:
                st.plotly_chart(charts['monthly_trend'], use_container_width=True)
    
    # Detailed insights
    st.subheader("💡 AI-Generated Insights")
    
    for insight in insights:
        st.write(f"• {insight}")
    
    # Seasonality analysis
    if cash_flow_prediction and 'seasonality_insights' in cash_flow_prediction:
        seasonality = cash_flow_prediction['seasonality_insights']
        if seasonality['weekly_patterns']:
            st.subheader("📅 Weekly Spending Patterns")
            
            weekly_data = list(seasonality['weekly_patterns'].items())
            weekly_df = pd.DataFrame(weekly_data, columns=['Day', 'Average Spending'])
            weekly_df['Average Spending'] = weekly_df['Average Spending'].round(2)
            
            # Order days correctly
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_df['Day'] = pd.Categorical(weekly_df['Day'], categories=day_order, ordered=True)
            weekly_df = weekly_df.sort_values('Day')
            
            fig_weekly = px.bar(
                weekly_df,
                x='Day',
                y='Average Spending',
                title=f"Weekly Spending Pattern - {selected_display_name}"
            )
            st.plotly_chart(fig_weekly, use_container_width=True)
    
    # Admin action recommendations
    st.subheader("🎯 Recommended Admin Actions")
    
    recommendations = []
    
    if cash_flow_prediction:
        if not cash_flow_prediction['low_balance_alerts'].empty:
            recommendations.append("📞 **Contact Customer**: Proactively reach out about potential low balance situation")
            recommendations.append("💡 **Suggest Products**: Recommend overdraft protection or savings account")
        
        if cash_flow_prediction['avg_daily_spending'] > 100:
            recommendations.append("📋 **Budget Review**: Offer budgeting consultation services")
        
        confidence = cash_flow_prediction.get('confidence_score', 0)
        if confidence < 0.6:
            recommendations.append("📊 **Data Review**: Customer may need more transaction history for accurate forecasting")
    
    if rfm_metrics and rfm_metrics['recency'] > 30:
        recommendations.append("📧 **Re-engagement**: Send personalized offers to reactivate customer")
    
    if not recommendations:
        recommendations.append("✅ **Monitor**: Customer appears to be in good financial standing")
    
    for rec in recommendations:
        st.info(rec)
    
    # Customer support notes
    st.subheader("📝 Support Notes")
    
    support_notes = st.text_area(
        "Add support notes for this customer:",
        placeholder="Record any issues, solutions provided, or follow-up actions needed...",
        height=100
    )
    
    if st.button("Save Support Notes", type="primary"):
        if support_notes.strip():
            st.success("Support notes saved successfully!")
            # In a real system, this would save to a database
        else:
            st.warning("Please enter support notes before saving.")
