import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

def load_data(sample_size=None):
    """Load MovieLens dataset"""
    print("Loading data...")
    print("Loading MovieLens 100K dataset...")
    
    # Load ratings
    ratings = pd.read_csv(r'..\\data\ml-latest-small\ml-latest-small\ratings.csv', header=0)
    
    # Load movies
    movies = pd.read_csv(r'..\\data\ml-latest-small\ml-latest-small\movies.csv', header=0)
    
    # Convert genres to list format
    movies['genres'] = movies['genres'].str.split('|')
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
    print("\\nSplitting data...")
    train_ratings, test_ratings = train_test_split(
        ratings, test_size=test_size, random_state=random_state
    )
    print(f"Training set size: {len(train_ratings)}, Test set size: {len(test_ratings)}")
    return train_ratings, test_ratings