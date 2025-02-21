# Customer Segmentation Using K-Means Clustering

This project segments customers into distinct groups based on their purchasing behavior using the **K-Means Clustering** algorithm. The project utilizes customer data to identify patterns and group customers with similar characteristics, useful for targeted marketing, personalized recommendations, and improving customer experience.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How to Use](#how-to-use)
- [Streamlit App](#streamlit-app)
- [Model Details](#model-details)
- [License](#license)

## Project Overview

The objective of this project is to apply unsupervised learning (K-Means Clustering) to a dataset of customers and segment them into meaningful groups based on attributes like age, income, and spending habits.

The dataset includes multiple customer attributes, and the K-Means algorithm is used to divide customers into **K distinct clusters**. The project also includes visualization techniques to interpret the clusters effectively.

## Project Structure

```
customer_segmentation_project/
│
├── data/
│   ├── customer_data.csv            # Customer dataset (input data)
│
├── models/
│   ├── kmeans_model.pkl            # Trained KMeans model
│
├── src/
│   ├── preprocess.py               # Script for data preprocessing
│   ├── train_model.py              # Script to train and save the KMeans model
│   ├── streamlit_app.py            # Streamlit application for interactive visualization
│
├── requirements.txt               # List of project dependencies
├── README.md                      # Project overview (this file)
```

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
```

### Step 2: Install Dependencies

This project requires the following Python libraries:

- pandas
- scikit-learn
- matplotlib
- seaborn
- streamlit

You can install all dependencies by running the following command:

```bash
pip install -r requirements.txt
```

Alternatively, you can manually install the required libraries:

```bash
pip install pandas scikit-learn matplotlib seaborn streamlit
```

## How to Use

### Step 1: Preprocess the Data

Run the `preprocess.py` script to clean and prepare the data:

```bash
python src/preprocess.py
```

This will clean the dataset and save the preprocessed data.

### Step 2: Train the K-Means Model

Once the data is preprocessed, train the K-Means clustering model by running the `train_model.py` script. This script will:

1. Load the preprocessed data.
2. Standardize the data for clustering.
3. Train the K-Means model.
4. Save the trained model to the `models/` folder as `kmeans_model.pkl`.

Run the following command:

```bash
python src/train_model.py
```

### Step 3: Visualize the Clusters

The script will generate visualizations of the clustering results, allowing you to interpret the customer segments effectively.

## Streamlit App

You can interactively visualize the customer segments using the **Streamlit** app. The app provides an easy-to-use interface for uploading new data, predicting clusters, and exploring customer segments.

### Run the Streamlit App

Execute the following command to launch the Streamlit web app:

```bash
streamlit run src/streamlit_app.py
```

This will open a browser window where you can:
- Upload customer data.
- Visualize customer clusters.
- Explore different segmentation results interactively.

## Model Details

- **Model Used**: K-Means Clustering (`sklearn.cluster.KMeans`)
- **Number of Clusters**: Determined using the **Elbow Method** or **Silhouette Score**.
- **Evaluation**: Visual evaluation through cluster plots.

### Model Parameters:
- **n_clusters**: Number of clusters (experiment with different values)
- **max_iter**: Maximum number of iterations for the algorithm
- **random_state**: Seed for reproducibility

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to update this `README.md` file if there are any changes in the project structure or if you add new features.

 