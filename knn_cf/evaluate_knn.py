import pandas as pd
import numpy as np
import time
import tracemalloc
from knn_recommender import KNNRecommender
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_movielens_data():
    """
    Load MovieLens dataset
    Returns:
        ratings_df: DataFrame containing user ratings
        movies_df: DataFrame containing movie information
    """
    data_dir = Path('../data/ml-latest-small/ml-latest-small')
    ratings_df = pd.read_csv(data_dir / 'ratings.csv')
    ratings_df = ratings_df.rename(columns={
        'userId': 'user_id',
        'movieId': 'item_id',
        'rating': 'rating'
    })
    movies_df = pd.read_csv(data_dir / 'movies.csv')
    movies_df = movies_df.rename(columns={'movieId': 'item_id'})
    return ratings_df, movies_df


def evaluate_model(method, n_neighbors=20, n_infer=1000):
    ratings_df, _ = load_movielens_data()
    # Split train/test
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

    # Training
    tracemalloc.start()
    start_time = time.time()
    model = KNNRecommender(n_neighbors=n_neighbors, method=method)
    model.fit(train_df)
    train_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    train_mem_mb = peak / 1024 / 1024
    tracemalloc.stop()

    # Inference (predict for test set)
    test_sample = test_df.sample(n=min(n_infer, len(test_df)), random_state=42)
    y_true = []
    y_pred = []
    tracemalloc.start()
    start_time = time.time()
    for _, row in test_sample.iterrows():
        pred = model.predict(row['user_id'], row['item_id'])
        y_true.append(row['rating'])
        y_pred.append(pred)
    infer_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    infer_mem_mb = peak / 1024 / 1024
    tracemalloc.stop()

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    infer_time_per_1000 = infer_time / len(test_sample) * 1000

    return {
        'Method': f'KNN-{method}',
        'RMSE': rmse,
        'MAE': mae,
        'Train Time (s)': train_time,
        'Inference Time (sec/1000)': infer_time_per_1000,
        'Train Mem (MB)': train_mem_mb,
        'Infer Mem (MB)': infer_mem_mb
    }


def main():
    print("Evaluating KNN Collaborative Filtering...")
    results = []
    for method in ['user', 'item']:
        res = evaluate_model(method, n_neighbors=20, n_infer=1000)
        results.append(res)
    df = pd.DataFrame(results)
    print("\nPerformance and Cost:")
    print(df[['Method', 'RMSE', 'MAE', 'Train Time (s)', 'Inference Time (sec/1000)', 'Train Mem (MB)', 'Infer Mem (MB)']])

if __name__ == "__main__":
    main() 