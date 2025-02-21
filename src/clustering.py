from sklearn.cluster import KMeans
import pickle

def perform_kmeans_clustering(features, n_clusters=5):
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)

    # Save the model
    with open("C:\\Users\\zk319\\OneDrive\\Desktop\\CustomerSegmentation\\models\\kmeans_model.pkl", 'wb') as f:
        pickle.dump(kmeans, f)

    return kmeans

if __name__ == "__main__":
    from preprocessing import load_and_preprocess_data
    _, features = load_and_preprocess_data("C:\\Users\\zk319\\OneDrive\\Desktop\\CustomerSegmentation\\data\\Mall_Customers.csv")
    perform_kmeans_clustering(features)
