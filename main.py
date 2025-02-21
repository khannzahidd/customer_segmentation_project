import os
import subprocess
from src.preprocessing import load_and_preprocess_data
from src.clustering import perform_kmeans_clustering

def main():
    print("==== Customer Segmentation Using K-Means Clustering ====")
    
    # Step 1: Check if the dataset exists
    dataset_path = "data/mall_customers.csv"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please ensure the file is in the correct location.")
        return
    
    # Step 2: Preprocess the data
    print("Step 1: Preprocessing the data...")
    data, features = load_and_preprocess_data(dataset_path)
    print("Preprocessing completed successfully!")

    # Step 3: Perform clustering
    print("Step 2: Clustering the data...")
    n_clusters = 5  # Default number of clusters
    kmeans = perform_kmeans_clustering(features, n_clusters=n_clusters)
    print(f"Clustering completed successfully! Model saved in 'models/kmeans_model.pkl'.")
    
    # Step 4: Run the Streamlit app
    print("Step 3: Starting the Streamlit app...")
    subprocess.run(["streamlit", "run", "streamlit_app.py"])

if __name__ == "__main__":
    main()
