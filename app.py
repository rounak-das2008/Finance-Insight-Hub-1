import streamlit as st
import pandas as pd
from auth import authenticate_user, get_user_role
from customer_dashboard import show_customer_dashboard
from admin_dashboard import show_admin_dashboard
from data_loader import load_customer_data, get_available_customers

# Configure page
st.set_page_config(
    page_title="Financial Analytics Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'username' not in st.session_state:
        st.session_state.username = None

    # Main title
    st.title("💰 Financial Analytics Platform")
    
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_authenticated_app()

def show_login_page():
    st.markdown("---")
    st.markdown("""<h2 style="text-align: center;">🔐 Login</h2>""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_role = get_user_role(username)
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        # Show demo credentials
        st.markdown("---")
        # st.info("""
        # **Demo Credentials:**
        
        # **Customer Access:**
        # - Username: james_smith (or any customer name from data folder)
        # - Password: customer123
        
        # **Admin Access:**
        # - Username: admin
        # - Password: admin123
        # """)

def show_authenticated_app():
    # Sidebar navigation
    with st.sidebar:
        st.write(f"👤 Logged in as: **{st.session_state.username}**")
        st.write(f"🏷️ Role: **{st.session_state.user_role.title()}**")
        
        if st.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
    
    # Show appropriate dashboard based on role
    if st.session_state.user_role == 'admin':
        show_admin_dashboard()
    elif st.session_state.user_role == 'customer':
        show_customer_dashboard(st.session_state.username)
    else:
        st.error("Invalid user role")

if __name__ == "__main__":
    main()
