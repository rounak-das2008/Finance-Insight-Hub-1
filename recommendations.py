import pandas as pd
import numpy as np
import streamlit as st
from analytics import calculate_spending_summary, calculate_rfm_metrics
from clustering import get_customer_cluster
from data_loader import load_products_data

def generate_product_recommendations(customer_name, customer_data, clustering_results=None, cluster_analysis=None):
    """
    Generate personalized product recommendations based on customer profile
    """
    if customer_data is None or customer_data.empty:
        return []
    
    recommendations = []
    
    # Load products
    products_df = load_products_data()
    
    # Calculate customer metrics
    spending_summary = calculate_spending_summary(customer_data)
    rfm_metrics = calculate_rfm_metrics(customer_data)
    
    if not spending_summary or not rfm_metrics:
        return []
    
    # Get customer cluster info
    cluster_info = get_customer_cluster(customer_name, clustering_results, cluster_analysis) if clustering_results else None
    
    # Rule-based recommendations
    total_spent = spending_summary['total_spent']
    top_categories = spending_summary['top_categories']
    current_balance = customer_data['balance'].iloc[-1]
    monthly_spending = spending_summary['monthly_spending'].mean() if not spending_summary['monthly_spending'].empty else 0
    
    # High spender recommendations
    if total_spent > 5000:  # High spender threshold
        cashback_card = products_df[products_df['product_name'].str.contains('Cashback', case=False, na=False)]
        if not cashback_card.empty:
            recommendations.append({
                'product': cashback_card.iloc[0],
                'reason': f"You've spent ${total_spent:.2f} total. A cashback card could earn you rewards on your purchases.",
                'relevance_score': 0.9,
                'category': 'High Spender Reward'
            })
    
    # Travel spender recommendations
    if 'Transport' in top_categories.index or 'Travel' in top_categories.index:
        travel_card = products_df[products_df['product_name'].str.contains('Travel', case=False, na=False)]
        if not travel_card.empty:
            travel_spending = top_categories.get('Transport', 0) + top_categories.get('Travel', 0)
            recommendations.append({
                'product': travel_card.iloc[0],
                'reason': f"You spent ${travel_spending:.2f} on transport/travel. Earn bonus points on travel purchases.",
                'relevance_score': 0.8,
                'category': 'Travel Rewards'
            })
    
    # High balance recommendations
    if current_balance > 2000:
        savings_account = products_df[products_df['product_name'].str.contains('Savings', case=False, na=False)]
        if not savings_account.empty:
            recommendations.append({
                'product': savings_account.iloc[0],
                'reason': f"With a balance of ${current_balance:.2f}, you could earn more with a high-yield savings account.",
                'relevance_score': 0.7,
                'category': 'Savings Opportunity'
            })
    
    # Business spender recommendations
    business_categories = ['business_expenses', 'Business lunch']
    business_spending = sum(top_categories.get(cat, 0) for cat in business_categories)
    if business_spending > 200:
        business_account = products_df[products_df['product_name'].str.contains('Business', case=False, na=False)]
        if not business_account.empty:
            recommendations.append({
                'product': business_account.iloc[0],
                'reason': f"You spent ${business_spending:.2f} on business expenses. A business account offers better features for professional use.",
                'relevance_score': 0.8,
                'category': 'Business Banking'
            })
    
    # Investment recommendations for high earners
    total_income = customer_data['credit'].sum()
    if total_income > 10000 and current_balance > 5000:
        investment_product = products_df[products_df['product_name'].str.contains('Investment', case=False, na=False)]
        if not investment_product.empty:
            recommendations.append({
                'product': investment_product.iloc[0],
                'reason': f"With income of ${total_income:.2f} and good balance management, consider growing your wealth through investments.",
                'relevance_score': 0.6,
                'category': 'Wealth Building'
            })
    
    # Low balance / loan recommendations
    if current_balance < 500 and monthly_spending > current_balance:
        loan_product = products_df[products_df['product_name'].str.contains('Loan', case=False, na=False)]
        if not loan_product.empty:
            recommendations.append({
                'product': loan_product.iloc[0],
                'reason': "Your spending often exceeds your balance. A personal loan could help manage cash flow.",
                'relevance_score': 0.7,
                'category': 'Financial Support'
            })
    
    # Cluster-based recommendations
    if cluster_info:
        cluster_name = cluster_info['cluster_name']
        
        if cluster_name == "VIP Customers":
            premium_products = products_df[products_df['target_segment'] == 'High Spender']
            for _, product in premium_products.iterrows():
                recommendations.append({
                    'product': product,
                    'reason': f"As a VIP customer, you qualify for our premium {product['product_name']}.",
                    'relevance_score': 0.85,
                    'category': 'VIP Exclusive'
                })
        
        elif cluster_name == "At-Risk Customers":
            # Focus on retention products
            savings_products = products_df[products_df['category'] == 'Savings']
            for _, product in savings_products.iterrows():
                recommendations.append({
                    'product': product,
                    'reason': "Reactivate your banking relationship with our competitive savings options.",
                    'relevance_score': 0.6,
                    'category': 'Re-engagement'
                })
    
    # Sort by relevance score and remove duplicates
    unique_recommendations = []
    seen_products = set()
    
    for rec in sorted(recommendations, key=lambda x: x['relevance_score'], reverse=True):
        product_id = rec['product']['product_id']
        if product_id not in seen_products:
            unique_recommendations.append(rec)
            seen_products.add(product_id)
    
    return unique_recommendations[:5]  # Return top 5 recommendations

def create_recommendation_summary(recommendations):
    """
    Create a summary of product recommendations
    """
    if not recommendations:
        return "No specific product recommendations available at this time."
    
    summary = "## 🎯 Personalized Product Recommendations\n\n"
    
    for i, rec in enumerate(recommendations, 1):
        product = rec['product']
        reason = rec['reason']
        category = rec['category']
        score = rec['relevance_score']
        
        summary += f"### {i}. {product['product_name']}\n"
        summary += f"**Category:** {category} | **Relevance:** {score*100:.0f}%\n\n"
        summary += f"**Why this fits you:** {reason}\n\n"
        summary += f"**Product Details:** {product['description']}\n\n"
        summary += "---\n\n"
    
    return summary

def analyze_product_interest_trends(all_customer_data, clustering_results, cluster_analysis):
    """
    Analyze product interest trends across customer segments for admin dashboard
    """
    if not all_customer_data or not clustering_results or not cluster_analysis:
        return None
    
    products_df = load_products_data()
    
    # Generate recommendations for all customers
    all_recommendations = {}
    product_interest = {}
    cluster_interest = {}
    
    for customer_name, customer_data in all_customer_data.items():
        if customer_data is None or customer_data.empty:
            continue
        
        recommendations = generate_product_recommendations(
            customer_name, customer_data, clustering_results, cluster_analysis
        )
        
        all_recommendations[customer_name] = recommendations
        
        # Get customer cluster
        cluster_info = get_customer_cluster(customer_name, clustering_results, cluster_analysis)
        if not cluster_info:
            continue
        
        cluster_name = cluster_info['cluster_name']
        
        # Track product interest by cluster
        if cluster_name not in cluster_interest:
            cluster_interest[cluster_name] = {}
        
        for rec in recommendations:
            product_name = rec['product']['product_name']
            category = rec['product']['category']
            
            # Overall product interest
            if product_name not in product_interest:
                product_interest[product_name] = {'count': 0, 'total_score': 0, 'category': category}
            
            product_interest[product_name]['count'] += 1
            product_interest[product_name]['total_score'] += rec['relevance_score']
            
            # Cluster-specific interest
            if product_name not in cluster_interest[cluster_name]:
                cluster_interest[cluster_name][product_name] = {'count': 0, 'total_score': 0}
            
            cluster_interest[cluster_name][product_name]['count'] += 1
            cluster_interest[cluster_name][product_name]['total_score'] += rec['relevance_score']
    
    # Calculate average scores
    for product, data in product_interest.items():
        if data['count'] > 0:
            data['avg_score'] = data['total_score'] / data['count']
        else:
            data['avg_score'] = 0
    
    # Sort by interest level
    sorted_products = sorted(
        product_interest.items(),
        key=lambda x: (x[1]['count'], x[1]['avg_score']),
        reverse=True
    )
    
    return {
        'all_recommendations': all_recommendations,
        'product_interest': dict(sorted_products),
        'cluster_interest': cluster_interest,
        'top_products': sorted_products[:5]
    }

def get_cross_selling_opportunities(customer_name, customer_data, all_customer_data, clustering_results, cluster_analysis):
    """
    Identify cross-selling opportunities based on similar customers
    """
    if not clustering_results or not cluster_analysis:
        return []
    
    cluster_info = get_customer_cluster(customer_name, clustering_results, cluster_analysis)
    if not cluster_info:
        return []
    
    similar_customers = cluster_info['similar_customers']
    current_recommendations = generate_product_recommendations(customer_name, customer_data, clustering_results, cluster_analysis)
    current_product_names = [rec['product']['product_name'] for rec in current_recommendations]
    
    # Analyze what similar customers are recommended
    similar_recommendations = {}
    
    for similar_customer in similar_customers[:3]:  # Top 3 similar customers
        if similar_customer in all_customer_data:
            similar_data = all_customer_data[similar_customer]
            if similar_data is not None and not similar_data.empty:
                recs = generate_product_recommendations(similar_customer, similar_data, clustering_results, cluster_analysis)
                for rec in recs:
                    product_name = rec['product']['product_name']
                    if product_name not in current_product_names:
                        if product_name not in similar_recommendations:
                            similar_recommendations[product_name] = {
                                'product': rec['product'],
                                'count': 0,
                                'total_score': 0,
                                'reasons': []
                            }
                        similar_recommendations[product_name]['count'] += 1
                        similar_recommendations[product_name]['total_score'] += rec['relevance_score']
                        similar_recommendations[product_name]['reasons'].append(rec['reason'])
    
    # Convert to cross-selling opportunities
    cross_sell_opportunities = []
    for product_name, data in similar_recommendations.items():
        if data['count'] >= 2:  # Recommended to at least 2 similar customers
            avg_score = data['total_score'] / data['count']
            cross_sell_opportunities.append({
                'product': data['product'],
                'reason': f"Customers similar to you often benefit from this product (recommended to {data['count']} similar customers).",
                'relevance_score': avg_score * 0.7,  # Slightly lower than direct recommendations
                'category': 'Cross-selling Opportunity'
            })
    
    return sorted(cross_sell_opportunities, key=lambda x: x['relevance_score'], reverse=True)[:3]
