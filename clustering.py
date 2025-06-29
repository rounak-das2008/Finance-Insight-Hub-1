import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from analytics import calculate_rfm_metrics

@st.cache_data
def prepare_clustering_features(all_customer_data):
    """
    Prepare features for customer clustering
    """
    if not all_customer_data:
        return None, None
    
    features_list = []
    customer_names = []
    
    for customer_name, df in all_customer_data.items():
        if df is None or df.empty:
            continue
            
        # Calculate RFM metrics
        rfm = calculate_rfm_metrics(df)
        if not rfm:
            continue
        
        # Calculate additional features
        total_spent = df['debit'].sum()
        total_income = df['credit'].sum()
        transaction_count = len(df)
        avg_transaction_amount = df['debit'].mean()
        
        # Category spending patterns
        category_spending = df.groupby('category')['debit'].sum()
        total_categories = len(category_spending)
        
        # Top categories (normalize by creating percentage features)
        top_category_pct = (category_spending.max() / total_spent) if total_spent > 0 else 0
        
        # Time-based features
        date_range_days = (df['date'].max() - df['date'].min()).days
        transaction_frequency = transaction_count / max(1, date_range_days)
        
        # Balance volatility
        balance_std = df['balance'].std()
        balance_trend = (df['balance'].iloc[-1] - df['balance'].iloc[0]) / max(1, date_range_days)
        
        # Spending volatility
        daily_spending = df.set_index('date')['debit'].resample('D').sum()
        spending_volatility = daily_spending.std()
        
        features = {
            'recency': rfm['recency'],
            'frequency': rfm['frequency'],
            'monetary': rfm['monetary'],
            'avg_transaction_amount': avg_transaction_amount,
            'total_categories': total_categories,
            'top_category_pct': top_category_pct,
            'transaction_frequency': transaction_frequency,
            'balance_volatility': balance_std,
            'balance_trend': balance_trend,
            'spending_volatility': spending_volatility,
            'net_flow': total_income - total_spent
        }
        
        features_list.append(features)
        customer_names.append(customer_name)
    
    if not features_list:
        return None, None
    
    features_df = pd.DataFrame(features_list, index=customer_names)
    
    # Handle missing values
    features_df = features_df.fillna(0)
    
    return features_df, customer_names

@st.cache_data
def perform_customer_clustering(features_df, n_clusters=4):
    """
    Perform K-means clustering on customer features
    """
    if features_df is None or features_df.empty:
        return None
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Add cluster labels to dataframe
    features_df_clustered = features_df.copy()
    features_df_clustered['cluster'] = cluster_labels
    
    # Calculate cluster centers
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_centers_df = pd.DataFrame(
        cluster_centers,
        columns=features_df.columns,
        index=[f'Cluster_{i}' for i in range(n_clusters)]
    )
    
    # Perform PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    
    pca_df = pd.DataFrame(
        features_pca,
        columns=['PC1', 'PC2'],
        index=features_df.index
    )
    pca_df['cluster'] = cluster_labels
    pca_df['customer'] = features_df.index
    
    return {
        'features_df': features_df_clustered,
        'cluster_centers': cluster_centers_df,
        'pca_df': pca_df,
        'scaler': scaler,
        'kmeans': kmeans,
        'pca_model': pca
    }

def analyze_clusters(clustering_results):
    """
    Analyze and describe customer clusters
    """
    if not clustering_results:
        return None
    
    features_df = clustering_results['features_df']
    cluster_centers = clustering_results['cluster_centers']
    
    cluster_analysis = {}
    
    for cluster_id in range(len(cluster_centers)):
        cluster_customers = features_df[features_df['cluster'] == cluster_id]
        cluster_center = cluster_centers.iloc[cluster_id]
        
        # Cluster characteristics
        characteristics = []
        
        # RFM analysis
        if cluster_center['recency'] < 7:
            characteristics.append("Recent activity")
        elif cluster_center['recency'] > 30:
            characteristics.append("Inactive users")
        
        if cluster_center['frequency'] > features_df['frequency'].median():
            characteristics.append("High frequency")
        else:
            characteristics.append("Low frequency")
        
        if cluster_center['monetary'] > features_df['monetary'].median():
            characteristics.append("High value")
        else:
            characteristics.append("Low value")
        
        # Spending patterns
        if cluster_center['spending_volatility'] > features_df['spending_volatility'].median():
            characteristics.append("Volatile spending")
        else:
            characteristics.append("Stable spending")
        
        if cluster_center['balance_trend'] > 0:
            characteristics.append("Growing balance")
        else:
            characteristics.append("Declining balance")
        
        # Generate cluster name based on characteristics
        if "High value" in characteristics and "High frequency" in characteristics:
            cluster_name = "VIP Customers"
        elif "High value" in characteristics:
            cluster_name = "High-Value Customers"
        elif "High frequency" in characteristics:
            cluster_name = "Active Customers"
        elif "Inactive users" in characteristics:
            cluster_name = "At-Risk Customers"
        else:
            cluster_name = "Regular Customers"
        
        cluster_analysis[cluster_id] = {
            'name': cluster_name,
            'size': len(cluster_customers),
            'characteristics': characteristics,
            'customers': cluster_customers.index.tolist(),
            'avg_monetary': cluster_center['monetary'],
            'avg_frequency': cluster_center['frequency'],
            'avg_recency': cluster_center['recency']
        }
    
    return cluster_analysis

def create_cluster_visualizations(clustering_results, cluster_analysis):
    """
    Create visualizations for customer clusters
    """
    if not clustering_results or not cluster_analysis:
        return None
    
    pca_df = clustering_results['pca_df']
    features_df = clustering_results['features_df']
    
    visualizations = {}
    
    # PCA scatter plot with explained variance in axis labels
    explained_var = clustering_results['pca_model'].explained_variance_ratio_
    fig_pca = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='cluster',
        hover_data=['customer'],
        title="Customer Clusters (PCA Visualization)",
        labels={
            'cluster': 'Cluster',
            'PC1': f'PC1 ({explained_var[0]*100:.1f}% variance)',
            'PC2': f'PC2 ({explained_var[1]*100:.1f}% variance)'
        }
    )
    visualizations['pca_scatter'] = fig_pca
    
    # Cluster size pie chart
    cluster_sizes = [cluster_analysis[i]['size'] for i in cluster_analysis.keys()]
    cluster_names = [cluster_analysis[i]['name'] for i in cluster_analysis.keys()]
    
    fig_pie = px.pie(
        values=cluster_sizes,
        names=cluster_names,
        title="Customer Distribution by Cluster"
    )
    visualizations['cluster_pie'] = fig_pie
    
    # RFM comparison by cluster
    rfm_data = []
    for cluster_id, info in cluster_analysis.items():
        rfm_data.append({
            'Cluster': info['name'],
            'Recency': info['avg_recency'],
            'Frequency': info['avg_frequency'],
            'Monetary': info['avg_monetary']
        })
    
    rfm_df = pd.DataFrame(rfm_data)
    
    # RFM bar charts
    fig_monetary = px.bar(
        rfm_df,
        x='Cluster',
        y='Monetary',
        title="Average Monetary Value by Cluster"
    )
    visualizations['monetary_by_cluster'] = fig_monetary
    
    fig_frequency = px.bar(
        rfm_df,
        x='Cluster',
        y='Frequency',
        title="Average Transaction Frequency by Cluster"
    )
    visualizations['frequency_by_cluster'] = fig_frequency
    
    # Feature importance heatmap
    cluster_centers_normalized = clustering_results['cluster_centers'].div(
        clustering_results['cluster_centers'].abs().max(), axis=1
    )
    
    fig_heatmap = px.imshow(
        cluster_centers_normalized.values,
        labels=dict(x="Features", y="Clusters", color="Normalized Value"),
        y=[cluster_analysis[i]['name'] for i in range(len(cluster_analysis))],
        x=cluster_centers_normalized.columns,
        title="Cluster Characteristics Heatmap"
    )
    visualizations['feature_heatmap'] = fig_heatmap
    
    return visualizations

def get_customer_cluster(customer_name, clustering_results, cluster_analysis):
    """
    Get cluster information for a specific customer
    """
    if not clustering_results or not cluster_analysis:
        return None
    
    features_df = clustering_results['features_df']
    
    if customer_name not in features_df.index:
        return None
    
    customer_cluster = features_df.loc[customer_name, 'cluster']
    cluster_info = cluster_analysis[customer_cluster]
    
    return {
        'cluster_id': customer_cluster,
        'cluster_name': cluster_info['name'],
        'cluster_characteristics': cluster_info['characteristics'],
        'similar_customers': [c for c in cluster_info['customers'] if c != customer_name]
    }
