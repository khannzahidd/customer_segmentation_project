import pandas as pd

def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
   
    data.columns = ["CustomerID", "Gender", "Age", "AnnualIncome", "SpendingScore"]
    
    # Encode gender as numeric
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
    
    # Select features for clustering
    features = data[['Age', 'AnnualIncome', 'SpendingScore']]
    
    return data, features

if __name__ == "__main__":
    data, features = load_and_preprocess_data("C:\\Users\\zk319\\OneDrive\\Desktop\\CustomerSegmentation\\data\\Mall_Customers.csv")
    print(features.head())
