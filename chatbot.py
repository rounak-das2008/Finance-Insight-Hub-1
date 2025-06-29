import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from typing import List, Dict, Any
from analytics import calculate_spending_summary, calculate_rfm_metrics, predict_cash_flow, generate_insights
from recommendations import generate_product_recommendations
from data_loader import load_customer_data, get_customer_profile
import requests
import time

# Check if we have access to Gemini API
def check_gemini_api():
    """Check if Gemini API key is available"""
    return os.getenv("GEMINI_API_KEY") is not None


# Initialize chat history in session state
def init_chat_history():
    """Initialize chat history for the current user"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_context' not in st.session_state:
        st.session_state.chat_context = {}

def get_customer_context(customer_name):
    """Get comprehensive customer context for chatbot"""
    try:
        # Load customer data
        customer_data = load_customer_data(customer_name)
        if customer_data is None:
            return None
        
        # Get profile and analytics
        profile = get_customer_profile(customer_name)
        spending_summary = calculate_spending_summary(customer_data)
        rfm_metrics = calculate_rfm_metrics(customer_data)
        cash_flow_prediction = predict_cash_flow(customer_data)
        insights = generate_insights(customer_data, rfm_metrics, cash_flow_prediction)
        recommendations = generate_product_recommendations(customer_name, customer_data)
        
        # Create context summary
        context = {
            'customer_name': customer_name.replace('_', ' ').title(),
            'current_balance': profile['current_balance'] if profile else 0,
            'total_spent': spending_summary['total_spent'] if spending_summary else 0,
            'total_income': spending_summary['total_income'] if spending_summary else 0,
            'transaction_count': spending_summary['transaction_count'] if spending_summary else 0,
            'avg_transaction': spending_summary['avg_transaction'] if spending_summary else 0,
            'top_categories': spending_summary['top_categories'].head(3).to_dict() if spending_summary else {},
            'recency': rfm_metrics['recency'] if rfm_metrics else 0,
            'frequency': rfm_metrics['frequency'] if rfm_metrics else 0,
            'monetary': rfm_metrics['monetary'] if rfm_metrics else 0,
            'predicted_balance_30d': cash_flow_prediction['predicted_balance_30d'] if cash_flow_prediction else 0,
            'predicted_balance_60d': cash_flow_prediction.get('predicted_balance_60d', 0) if cash_flow_prediction else 0,
            'net_daily_flow': cash_flow_prediction['net_daily_flow'] if cash_flow_prediction else 0,
            'forecast_confidence': cash_flow_prediction.get('confidence_score', 0) if cash_flow_prediction else 0,
            'forecast_method': cash_flow_prediction.get('method_used', 'Unknown') if cash_flow_prediction else 'Unknown',
            'validation_metrics': cash_flow_prediction.get('validation_metrics', {}) if cash_flow_prediction else {},
            'insights': insights if insights else [],
            'recommendations': [
                {
                    'product': rec['product']['product_name'],
                    'reason': rec['reason'],
                    'category': rec['category']
                } for rec in recommendations[:3]
            ] if recommendations else [],
            'data_period': f"{profile['date_range']['start'].strftime('%Y-%m-%d')} to {profile['date_range']['end'].strftime('%Y-%m-%d')}" if profile else "N/A"
        }
        
        return context
    except Exception as e:
        st.error(f"Error getting customer context: {str(e)}")
        return None

def create_system_prompt(customer_context):
    """Create system prompt with customer context"""
    if not customer_context:
        return "You are a helpful financial assistant."
    
    prompt = f"""You are a personal financial assistant for {customer_context['customer_name']}. You have access to their financial data and should provide personalized advice based on their specific situation.

CUSTOMER PROFILE:
- Name: {customer_context['customer_name']}
- Current Balance: ${customer_context['current_balance']:,.2f}
- Total Spent: ${customer_context['total_spent']:,.2f}
- Total Income: ${customer_context['total_income']:,.2f}
- Transaction Count: {customer_context['transaction_count']}
- Average Transaction: ${customer_context['avg_transaction']:,.2f}
- Data Period: {customer_context['data_period']}

SPENDING PATTERNS:
- Top Categories: {', '.join([f"{cat}: ${amt:.2f}" for cat, amt in customer_context['top_categories'].items()])}
- Days Since Last Transaction: {customer_context['recency']}
- Transaction Frequency: {customer_context['frequency']}
- Total Monetary Value: ${customer_context['monetary']:,.2f}

FORECAST INSIGHTS:
- Predicted Balance (30 days): ${customer_context['predicted_balance_30d']:,.2f}
- Predicted Balance (60 days): ${customer_context['predicted_balance_60d']:,.2f}
- Daily Net Flow: ${customer_context['net_daily_flow']:,.2f}
- Forecast Confidence: {customer_context['forecast_confidence']:.1%}
- Forecast Method: {customer_context['forecast_method']}

VALIDATION METRICS:
- Model Accuracy (MAPE): {customer_context['validation_metrics'].get('mape', 'N/A') if customer_context['validation_metrics'] and customer_context['validation_metrics'].get('mape') else 'N/A'}
- Validation Points: {customer_context['validation_metrics'].get('validation_points', 'N/A') if customer_context['validation_metrics'] else 'N/A'}

CURRENT INSIGHTS:
{chr(10).join(['- ' + insight for insight in customer_context['insights']])}

RECOMMENDED PRODUCTS:
{chr(10).join([f"- {rec['product']}: {rec['reason']}" for rec in customer_context['recommendations']])}

GUIDELINES:
1. Always be helpful, professional, and supportive
2. Provide specific advice based on their actual financial data
3. Focus on actionable recommendations for budgeting and financial health
4. Reference their specific spending patterns when giving advice
5. Be encouraging while being realistic about their financial situation
6. Suggest relevant banking products when appropriate
7. Keep responses concise but comprehensive
8. Always maintain confidentiality - this data is private to this customer only

Remember: You are speaking directly to {customer_context['customer_name']} about their personal finances. Be conversational but professional."""

    return prompt

def call_gemini_api(prompt, chat_history, customer_context):
    """Call Gemini API for chat response"""
    try:
        import google.genai as genai
        
        # Initialize Gemini client
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        
        # Create conversation context
        system_prompt = create_system_prompt(customer_context)
        
        # Format chat history for Gemini
        conversation_parts = [system_prompt]
        
        # Add recent chat history (last 10 exchanges)
        for message in chat_history[-10:]:
            conversation_parts.append(f"User: {message['user']}")
            conversation_parts.append(f"Assistant: {message['bot']}")
        
        # Add current prompt
        conversation_parts.append(f"User: {prompt}")
        
        full_prompt = "\n\n".join(conversation_parts)
        
        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt
        )
        
        return response.text if response.text else "I apologize, but I'm having trouble generating a response right now. Please try again."
        
    except Exception as e:
        return f"I'm currently experiencing technical difficulties. Please try again later. (Error: {str(e)})"

def call_ollama_api(prompt, chat_history, customer_context, model="llama3.2:3b"):
    """Call local Ollama API for chat response"""
    try:
        # Prepare the conversation
        system_prompt = create_system_prompt(customer_context)
        
        # Format messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history
        for message in chat_history[-10:]:  # Last 10 exchanges
            messages.append({"role": "user", "content": message['user']})
            messages.append({"role": "assistant", "content": message['bot']})
        
        # Add current message
        messages.append({"role": "user", "content": prompt})
        
        # Call Ollama API
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["message"]["content"]
        else:
            return "I'm having trouble connecting to the local AI service. Please ensure Ollama is running."
            
    except requests.exceptions.ConnectionError:
        return "I couldn't connect to the local AI service. Please make sure Ollama is installed and running on your system."
    except requests.exceptions.Timeout:
        return "The AI service is taking too long to respond. Please try with a shorter message."
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}"

def get_fallback_response(prompt, customer_context):
    """Generate a simple rule-based response as fallback"""
    prompt_lower = prompt.lower()
    
    if not customer_context:
        return "I need access to your financial data to provide personalized advice. Please ensure you're logged in and have transaction data available."
    
    # Simple keyword-based responses
    if any(word in prompt_lower for word in ['balance', 'money', 'account']):
        return f"Your current balance is ${customer_context['current_balance']:,.2f}. Based on your spending pattern, you have a daily net flow of ${customer_context['net_daily_flow']:,.2f}."
    
    elif any(word in prompt_lower for word in ['spending', 'expenses', 'spend']):
        top_cat = list(customer_context['top_categories'].keys())[0] if customer_context['top_categories'] else "various categories"
        top_amount = list(customer_context['top_categories'].values())[0] if customer_context['top_categories'] else 0
        return f"You've spent ${customer_context['total_spent']:,.2f} total, with your highest spending in {top_cat} (${top_amount:.2f}). Consider reviewing your spending in this category."
    
    elif any(word in prompt_lower for word in ['forecast', 'future', 'predict']):
        confidence_text = f" with {customer_context['forecast_confidence']:.1%} confidence" if customer_context['forecast_confidence'] > 0 else ""
        return f"Based on your current patterns, your balance is predicted to be ${customer_context['predicted_balance_30d']:,.2f} in 30 days and ${customer_context['predicted_balance_60d']:,.2f} in 60 days{confidence_text}. The forecast uses {customer_context['forecast_method']} based on 5 months of your transaction history."
    
    elif any(word in prompt_lower for word in ['recommend', 'product', 'advice']):
        if customer_context['recommendations']:
            rec = customer_context['recommendations'][0]
            return f"I recommend considering our {rec['product']}. {rec['reason']}"
        else:
            return "Continue using our services to receive personalized product recommendations based on your spending patterns."
    
    elif any(word in prompt_lower for word in ['help', 'hello', 'hi']):
        return f"Hello {customer_context['customer_name']}! I'm your personal financial assistant. I can help you with budgeting advice, spending analysis, product recommendations, and financial planning based on your transaction history."
    
    else:
        return "I can help you with questions about your spending patterns, budget planning, product recommendations, and financial insights. What would you like to know about your finances?"

def process_chat_message(user_message, customer_name):
    """Process user message and generate response"""
    # Get customer context
    customer_context = get_customer_context(customer_name)
    
    # Get chat history
    chat_history = st.session_state.get('chat_history', [])
    
    # Try different AI services in order of preference
    response = None
    
    # 1. Try Gemini API if available
    if check_gemini_api():
        try:
            response = call_gemini_api(user_message, chat_history, customer_context)
        except Exception as e:
            st.warning("Gemini API failed, trying local AI...")
    
    # 2. Try Ollama if Gemini failed or unavailable
    if not response or "technical difficulties" in response.lower():
        try:
            response = call_ollama_api(user_message, chat_history, customer_context)
        except Exception as e:
            st.warning("Local AI unavailable, using fallback responses...")
    
    # 3. Fallback to rule-based responses
    if not response or "trouble connecting" in response.lower() or "couldn't connect" in response.lower():
        response = get_fallback_response(user_message, customer_context)
    
    # Add to chat history
    chat_entry = {
        'timestamp': datetime.now().isoformat(),
        'user': user_message,
        'bot': response
    }
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append(chat_entry)
    
    # Keep only last 50 messages to manage memory
    if len(st.session_state.chat_history) > 50:
        st.session_state.chat_history = st.session_state.chat_history[-50:]
    
    return response

def render_chatbot_interface(customer_name):
    """Render the chatbot interface"""
    # Initialize chat history
    init_chat_history()
    
    # Chatbot container
    with st.container():
        st.markdown("---")
        st.subheader("🤖 Financial Assistant Chat")
        
        # Display chat history
        chat_container = st.container()
        
        with chat_container:
            if st.session_state.chat_history:
                for i, message in enumerate(st.session_state.chat_history[-10:]):  # Show last 10 messages
                    # User message
                    st.markdown(f"""
                    <div style="text-align: right; margin: 10px 0; padding: 10px; background-color: #e3f2fd; border-radius: 10px;">
                        <strong>You:</strong> {message['user']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Bot response
                    st.markdown(f"""
                    <div style="text-align: left; margin: 10px 0; padding: 10px; background-color: #f5f5f5; border-radius: 10px;">
                        <strong>💰 Financial Assistant:</strong> {message['bot']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("👋 Hi! I'm your personal financial assistant. I can help you with spending analysis, budgeting advice, and product recommendations based on your financial data. What would you like to know?")
        
        # Chat input
        with st.form("chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "Ask me about your finances...",
                    placeholder="e.g., 'How can I reduce my spending?' or 'What's my spending trend?'",
                    label_visibility="collapsed"
                )
            
            with col2:
                send_button = st.form_submit_button("Send", use_container_width=True)
            
            if send_button and user_input.strip():
                # Process the message
                with st.spinner("Thinking..."):
                    response = process_chat_message(user_input.strip(), customer_name)
                
                # Rerun to show the new message
                st.rerun()
        
        # Quick action buttons
        st.markdown("**Quick Questions:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💳 Spending Summary", key="quick_spending"):
                with st.spinner("Analyzing your spending..."):
                    response = process_chat_message("Give me a summary of my spending patterns", customer_name)
                st.rerun()
        
        with col2:
            if st.button("🔮 Budget Forecast", key="quick_forecast"):
                with st.spinner("Creating forecast..."):
                    response = process_chat_message("What does my financial forecast look like?", customer_name)
                st.rerun()
        
        with col3:
            if st.button("🎯 Product Advice", key="quick_products"):
                with st.spinner("Finding recommendations..."):
                    response = process_chat_message("What banking products would you recommend for me?", customer_name)
                st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

def show_chatbot_status():
    """Show which AI services are available"""
    with st.expander("🔧 AI Service Status", expanded=False):
        # Check Gemini API
        if check_gemini_api():
            st.success("✅ Gemini AI - Available")
        else:
            st.warning("⚠️ Gemini AI - API key not configured")
        
        # Check Ollama
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get("models", [])
                st.success(f"✅ Ollama - Available ({len(models)} models)")
                if models:
                    model_names = [model["name"] for model in models[:3]]
                    st.info(f"Available models: {', '.join(model_names)}")
            else:
                st.error("❌ Ollama - Service not responding")
        except:
            st.warning("⚠️ Ollama - Not running (install and run 'ollama serve' for local AI)")
        
        st.info("💡 If no AI services are available, the chatbot will use rule-based responses.")