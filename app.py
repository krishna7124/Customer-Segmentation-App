import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load the trained model and scaler
with open('./Furniture_Model/kmeans_rfm_model.pkl', 'rb') as model_file:
    kmeans_model = joblib.load(model_file)

with open('./Furniture_Model/scaler_rfm.pkl', 'rb') as scaler_file:
    scaler = joblib.load(scaler_file)

# Function to perform EDA


def perform_eda(df):
    st.write("### Basic Statistics")
    st.dataframe(df.describe())

    st.write("### Missing Values")
    st.dataframe(df.isnull().sum())

    st.write("### Unique Values in Categorical Features")
    for column in df.select_dtypes(include=['object']).columns:
        st.write(f"{column}: {df[column].nunique()} unique values")

    st.write("### Distribution of Numerical Features")
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        fig, ax = plt.subplots()
        sns.histplot(df[column], bins=30, kde=True, ax=ax)
        st.pyplot(fig)

    st.write("### Correlation Heatmap")
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:  # Check if there are numeric columns
        corr = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns found to calculate correlation.")

# Function to calculate RFM


def calculate_rfm(df):
    # Convert 'Order Date' to datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    # Calculate RFM metrics
    current_date = df['Order Date'].max() + pd.DateOffset(days=1)
    rfm_df = df.groupby('Customer ID').agg({
        'Order Date': lambda x: (current_date - x.max()).days,  # Recency
        'Order ID': 'count',  # Frequency
        'Sales': 'sum'  # Monetary
    }).reset_index()

    rfm_df.rename(columns={'Order Date': 'Recency',
                  'Order ID': 'Frequency', 'Sales': 'Monetary'}, inplace=True)
    return rfm_df


# Streamlit UI
st.title("Customer Segmentation App")
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    customer_data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(customer_data.head())

    # Perform EDA
    perform_eda(customer_data)

    # Prepare data for clustering
    rfm_data = calculate_rfm(customer_data)
    rfm_scaled = scaler.transform(
        # Scale only RFM columns
        rfm_data[['Recency', 'Frequency', 'Monetary']])

    # Clustering model selection
    st.write("### Clustering Model Selection")
    selected_model = st.selectbox("Select Clustering Algorithm", [
                                  "KMeans", "Hierarchical", "DBSCAN"])

    if selected_model == "KMeans":
        n_clusters = st.slider("Select number of clusters:", 2, 10, 4)
        model = KMeans(n_clusters=n_clusters)
    elif selected_model == "Hierarchical":
        n_clusters = st.slider("Select number of clusters:", 2, 10, 4)
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif selected_model == "DBSCAN":
        eps = st.slider(
            "Select epsilon (eps) value for DBSCAN:", 0.1, 2.0, 0.5)
        min_samples = st.slider("Select minimum samples for DBSCAN:", 1, 10, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)

    # Fit the model
    model.fit(rfm_scaled)

    # Check the number of unique clusters
    if selected_model != "DBSCAN" or len(set(model.labels_)) > 1:
        silhouette_avg = silhouette_score(rfm_scaled, model.labels_)
        st.write(f"Silhouette Score for {selected_model}: {silhouette_avg}")
    else:
        st.warning(
            "DBSCAN produced only one cluster. Cannot calculate silhouette score.")

    # Add cluster labels to the original data
    customer_data['Cluster'] = model.labels_
    st.write("### Cluster Assignments")
    st.dataframe(customer_data)

    # Visualization of clusters
    st.write("### Clustering Visualization")
    if selected_model == "KMeans" or selected_model == "Hierarchical":
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=rfm_data['Recency'], y=rfm_data['Monetary'], hue=model.labels_, palette='deep')
        plt.title(f"Clusters using {selected_model}")
        st.pyplot(fig)

    # Additional marketing insights or features can be added here...
    st.write("### Marketing Insights")
    # Example: Analyze customer segments
    cluster_summary = customer_data.groupby('Cluster').agg({
        'Sales': ['mean', 'sum'],
        'Quantity': ['mean', 'sum'],
        # Add other relevant columns here
    }).reset_index()

    st.write("### Cluster Summary")
    st.dataframe(cluster_summary)

    # Example: Recommend strategies based on segments
    for cluster in customer_data['Cluster'].unique():
        st.write(f"**Marketing Strategy for Cluster {cluster}:**")
        if cluster == 0:
            st.write("  - Target these customers with high-value offers.")
        elif cluster == 1:
            st.write("  - Engage with loyalty programs to increase retention.")
        elif cluster == 2:
            st.write("  - Focus on upselling and cross-selling strategies.")
        else:
            st.write("  - Use personalized marketing approaches.")
