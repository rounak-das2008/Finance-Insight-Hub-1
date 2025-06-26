import streamlit as st
import os
from data_loader import get_available_customers

def authenticate_user(username, password):
    """
    Simple authentication system
    Admin credentials: admin/admin123
    Customer credentials: {customer_name}/customer123
    """
    # Admin authentication
    if username == "admin" and password == "admin123":
        return True
    
    # Customer authentication
    available_customers = get_available_customers()
    if username in available_customers and password == "customer123":
        return True
    
    return False

def get_user_role(username):
    """
    Determine user role based on username
    """
    if username == "admin":
        return "admin"
    else:
        return "customer"

def check_authentication():
    """
    Check if user is authenticated
    """
    return st.session_state.get('authenticated', False)

def require_auth(func):
    """
    Decorator to require authentication
    """
    def wrapper(*args, **kwargs):
        if not check_authentication():
            st.error("Please login to access this page")
            return None
        return func(*args, **kwargs)
    return wrapper

def require_role(required_role):
    """
    Decorator to require specific role
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not check_authentication():
                st.error("Please login to access this page")
                return None
            if st.session_state.get('user_role') != required_role:
                st.error(f"Access denied. Required role: {required_role}")
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator
