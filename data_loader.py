import pandas as pd
import os
import streamlit as st
from datetime import datetime, timedelta
import glob

@st.cache_data
def get_available_customers():
    """
    Get list of available customer CSV files
    """
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return []
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    customers = []
    
    for file in csv_files:
        filename = os.path.basename(file)
        if filename not in ['customers.csv', 'products.csv']:
            # Extract customer name from filename
            customer_name = filename.replace('.csv', '').lower().replace(' ', '_')
            customers.append(customer_name)
    
    return customers

@st.cache_data
def load_customer_data(customer_name):
    """
    Load customer transaction data from CSV file
    """
    data_dir = "data"
    
    # Try different filename patterns
    possible_files = [
        f"{customer_name}.csv",
        f"{customer_name.replace('_', ' ')}.csv",
        f"{customer_name.title().replace('_', '_')}.csv"
    ]
    
    for filename in possible_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                
                # Standardize column names
                df.columns = df.columns.str.lower().str.strip()
                
                # Convert date column to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Ensure required columns exist
                required_columns = ['date', 'category', 'debit', 'credit', 'balance']
                for col in required_columns:
                    if col not in df.columns:
                        st.error(f"Missing required column: {col}")
                        return None
                
                return df
            except Exception as e:
                st.error(f"Error loading data for {customer_name}: {str(e)}")
                return None
    
    # If no file found, return None
    return None

@st.cache_data
def load_all_customer_data():
    """
    Load all customer data for admin analytics
    """
    customers = get_available_customers()
    all_data = {}
    
    for customer in customers:
        data = load_customer_data(customer)
        if data is not None:
            all_data[customer] = data
    
    return all_data

@st.cache_data
def load_products_data():
    """
    Load product information
    """
    filepath = "data/products.csv"
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        # Create default products if file doesn't exist
        products = pd.DataFrame({
            'product_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006'],
            'product_name': [
                'Premium Cashback Credit Card',
                'High-Yield Savings Account',
                'Business Checking Account',
                'Personal Loan',
                'Investment Portfolio',
                'Travel Rewards Credit Card'
            ],
            'category': ['Credit Card', 'Savings', 'Checking', 'Loan', 'Investment', 'Credit Card'],
            'description': [
                '2% cashback on all purchases',
                '3.5% APY on savings',
                'No monthly fees for business',
                'Low interest personal loan',
                'Diversified investment options',
                '3x points on travel purchases'
            ],
            'target_segment': ['High Spender', 'Saver', 'Business', 'Credit Builder', 'Investor', 'Travel Enthusiast']
        })
        return products

def get_customer_profile(customer_name):
    """
    Get customer profile information
    """
    # This would typically come from a customer database
    # For now, we'll extract basic info from the transaction data
    data = load_customer_data(customer_name)
    if data is None:
        return None
    
    profile = {
        'name': customer_name.replace('_', ' ').title(),
        'total_transactions': len(data),
        'date_range': {
            'start': data['date'].min(),
            'end': data['date'].max()
        },
        'current_balance': data['balance'].iloc[-1] if not data.empty else 0,
        'avg_monthly_spending': data['debit'].sum() / max(1, len(data['date'].dt.to_period('M').unique()))
    }
    
    return profile
