# Financial Analytics Platform

## Overview

This is a Python-based financial analytics platform built with Streamlit that provides personalized budgeting and spending insights for retail banking customers. The system supports two user types: customers who can view their personal financial analytics and interact with insights, and bank admins who can access aggregated customer analytics and system-level overviews.

## System Architecture

### Frontend Architecture
- **Streamlit**: Single web application framework serving both customer and admin dashboards
- **Multi-tab interface**: Organized views for different analytics sections
- **Role-based UI**: Different interfaces based on user authentication (customer vs admin)
- **Interactive visualizations**: Plotly-based charts and graphs for data visualization

### Backend Architecture
- **File-based data storage**: CSV files for customer transactions and reference data
- **Module-based architecture**: Separated concerns across analytics, clustering, recommendations, and authentication
- **Session state management**: Streamlit session state for user authentication and data persistence
- **Caching**: Streamlit caching decorators for performance optimization

### Authentication System
- **Simple role-based authentication**: Admin (admin/admin123) and customer (customer_name/customer123) credentials
- **Session-based state management**: User authentication persisted in Streamlit session state
- **Role-based access control**: Different dashboards and features based on user role

## Key Components

### Data Management (`data_loader.py`)
- **CSV-based data storage**: Customer transaction data stored in individual CSV files
- **Dynamic customer discovery**: Automatically detects available customer data files
- **Data standardization**: Normalizes column names and data formats
- **Reference data**: Products and customer profile information

### Analytics Engine (`analytics.py`)
- **Spending analysis**: Comprehensive spending summaries and categorization
- **RFM metrics**: Recency, Frequency, Monetary analysis for customer segmentation
- **Time series analysis**: Monthly and weekly spending patterns
- **Predictive analytics**: Cash flow forecasting capabilities

### Customer Segmentation (`clustering.py`)
- **Machine learning clustering**: K-means clustering for customer segmentation
- **Feature engineering**: RFM metrics, spending patterns, and behavioral features
- **PCA visualization**: Dimensionality reduction for cluster visualization
- **Cluster analysis**: Automated insights and segment characterization

### Recommendation System (`recommendations.py`)
- **Rule-based recommendations**: Product suggestions based on spending patterns
- **Cross-selling opportunities**: Targeted product recommendations
- **Cluster-based recommendations**: Leverages customer segmentation for personalization
- **Product matching**: Maps customer profiles to available financial products

### User Interfaces
- **Customer Dashboard** (`customer_dashboard.py`): Personal analytics, insights, and recommendations
- **Admin Dashboard** (`admin_dashboard.py`): Aggregated analytics, customer segments, and system overview
- **Authentication Flow** (`auth.py`): Login management and role-based access control

## Data Flow

1. **Authentication**: Users login with role-specific credentials
2. **Data Loading**: System loads customer transaction data from CSV files
3. **Analytics Processing**: Calculates spending summaries, RFM metrics, and patterns
4. **Clustering Analysis**: Performs customer segmentation using machine learning
5. **Recommendation Generation**: Creates personalized product recommendations
6. **Dashboard Rendering**: Displays analytics through role-appropriate interfaces
7. **Export Capabilities**: Allows data export for further analysis

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework and UI components
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive data visualizations
- **Scikit-learn**: Machine learning algorithms for clustering
- **NumPy**: Numerical computing support

### Data Processing
- **CSV file handling**: Built-in Python and Pandas capabilities
- **Date/time processing**: Python datetime and Pandas date functionality
- **Statistical analysis**: NumPy and Pandas statistical functions

## Deployment Strategy

### Replit Configuration
- **Python 3.11 environment**: Modern Python version with full feature support
- **Streamlit server**: Configured to run on port 5000 with autoscale deployment
- **Parallel workflow**: Supports concurrent task execution
- **Package management**: UV lock file for dependency management

### Production Considerations
- **Scalability**: Currently designed for small to medium datasets
- **Data security**: File-based storage suitable for development/MVP stage
- **Performance**: Streamlit caching implemented for improved response times
- **Monitoring**: Basic error handling and user feedback systems

## Changelog

```
Changelog:
- June 26, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```