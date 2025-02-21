import streamlit as st
import pandas as pd
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
with open('models/kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Load data
data = pd.read_csv('data/mall_customers.csv')
data.columns = ["CustomerID", "Gender", "Age", "AnnualIncome", "SpendingScore"]
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Streamlit App
st.title("Customer Segmentation Using K-Means Clustering")
st.write("""
### Cluster customers based on age, annual income, and spending score.
""")

# User input for number of clusters
n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=5)

# Re-run KMeans with user-selected clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['Age', 'AnnualIncome', 'SpendingScore']])

# Show data with clusters
st.write("### Data with Cluster Labels")
st.dataframe(data)

# Plot clusters
st.write("### Cluster Visualization")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=data, 
    x='AnnualIncome', 
    y='SpendingScore', 
    hue='Cluster', 
    palette='viridis', 
    ax=ax
)
plt.title("Clusters of Customers")
st.pyplot(fig)

# Download clustered data
csv = data.to_csv(index=False)
st.download_button(
    label="Download Clustered Data as CSV",
    data=csv,
    file_name='clustered_customers.csv',
    mime='text/csv'
)
