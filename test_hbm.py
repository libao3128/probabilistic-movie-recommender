import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from hbm import HierarchicalBayesianRecommender
from tqdm import tqdm
import time
import warnings
from recommender_viz import HBMRecommenderReporter as RecommenderReporter
warnings.filterwarnings('ignore')

def load_data(sample_size=None):
    """Load MovieLens dataset"""
    print("Loading data...")
    print("Loading MovieLens 100K dataset...")
    
    # Load ratings
    ratings = pd.read_csv(r'data\ml-100k\ml-100k\u.data', sep='\t', 
                         names=['userId', 'movieId', 'rating', 'timestamp'])
    
    # Load movies
    movies = pd.read_csv(r'data\ml-100k\ml-100k\u.item', sep='|', encoding='latin-1',
                        names=['movieId', 'title', 'release_date', 'video_release', 'url'] + 
                              [f'genre_{i}' for i in range(19)])
    
    # Convert genres to list format
    genre_columns = [f'genre_{i}' for i in range(19)]
    movies['genres'] = movies[genre_columns].apply(
        lambda x: [i for i, v in enumerate(x) if v == 1], axis=1
    )
    movies = movies[['movieId', 'title', 'genres']]
    
    if sample_size:
        # Sample users
        unique_users = ratings['userId'].unique()
        sampled_users = np.random.choice(unique_users, size=sample_size, replace=False)
        ratings = ratings[ratings['userId'].isin(sampled_users)]
        movies = movies[movies['movieId'].isin(ratings['movieId'].unique())]
    
    print(f"Loaded {len(ratings)} ratings and {len(movies)} movies")
    return ratings, movies

def split_data(ratings, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    print("\nSplitting data...")
    train_ratings, test_ratings = train_test_split(
        ratings, test_size=test_size, random_state=random_state
    )
    print(f"Training set size: {len(train_ratings)}, Test set size: {len(test_ratings)}")
    return train_ratings, test_ratings

def evaluate_predictions(model, test_ratings):
    """Evaluate model predictions"""
    print("\nEvaluating predictions...")
    predictions = []
    actuals = []
    
    for _, row in tqdm(test_ratings.iterrows(), total=len(test_ratings), desc="Evaluating ratings"):

        pred = model.predict(row['userId'], row['movieId'])
        predictions.append(pred)
        actuals.append(row['rating'])

    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'predictions': predictions,
        'actuals': actuals
    }

def main():
    # Load and preprocess data
    ratings, movies = load_data()  # 使用全部用戶
    train_ratings, test_ratings = split_data(ratings)
    
    # Initialize model
    print("\nInitializing model...")
    model = HierarchicalBayesianRecommender(train_ratings, movies)
    
    # Fit model
    print("\nFitting model...")
    model.fit(n_iter=100)
    
    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_predictions(model, test_ratings)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"MSE: {results['mse']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    
    # Plot results
    reporter = RecommenderReporter(model, movies, results)
    reporter.plot_all()
    reporter.print_stats()
    
    # Generate recommendations for a sample user
    print("\nGenerating recommendations for a sample user...")
    sample_user = test_ratings['userId'].iloc[0]
    recommendations = model.get_recommendations(sample_user, n_recommendations=5)
    
    print("\nTop 5 recommendations for user", sample_user)
    for movie_id, score in recommendations:
        movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0]
        print(f"Movie: {movie_title}, Predicted Rating: {score:.2f}")

if __name__ == "__main__":
    main() 