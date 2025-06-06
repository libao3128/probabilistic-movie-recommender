import pandas as pd
import numpy as np
from knn_recommender import KNNRecommender
from pathlib import Path

def load_movielens_data():
    """
    Load MovieLens dataset
    Returns:
        ratings_df: DataFrame containing user ratings
        movies_df: DataFrame containing movie information
    """
    data_dir = Path('../data/ml-latest-small/ml-latest-small')
    
    # Load ratings data
    ratings_df = pd.read_csv(data_dir / 'ratings.csv')
    # Rename columns to match our code
    ratings_df = ratings_df.rename(columns={
        'userId': 'user_id',
        'movieId': 'item_id',
        'rating': 'rating'
    })
    
    # Load movie data (optional, for displaying movie titles)
    movies_df = pd.read_csv(data_dir / 'movies.csv')
    movies_df = movies_df.rename(columns={'movieId': 'item_id'})
    
    return ratings_df, movies_df

def main():
    # Load data
    print("Loading MovieLens dataset...")
    ratings_df, movies_df = load_movielens_data()
    print(f"Loading complete! Dataset contains {len(ratings_df)} ratings and {len(movies_df)} movies")
    
    # Create user-based collaborative filtering recommender
    print("\nTraining user-based collaborative filtering model...")
    user_cf = KNNRecommender(n_neighbors=20, method='user')
    user_cf.fit(ratings_df)
    
    # Create item-based collaborative filtering recommender
    print("Training item-based collaborative filtering model...")
    item_cf = KNNRecommender(n_neighbors=20, method='item')
    item_cf.fit(ratings_df)
    
    # Select a user for recommendations
    test_user_id = ratings_df['user_id'].iloc[0]  # Use the first user as an example
    
    # Get user's rated movies
    user_ratings = ratings_df[ratings_df['user_id'] == test_user_id]
    print(f"\nMovies rated by user {test_user_id}:")
    for _, row in user_ratings.merge(movies_df, on='item_id').head().iterrows():
        print(f"- {row['title']}: {row['rating']} stars")
    
    # Generate recommendations
    print(f"\nGenerating recommendations for user {test_user_id}...")
    
    print("\nUser-based collaborative filtering recommendations:")
    user_recommendations = user_cf.recommend(test_user_id, n_recommendations=5)
    for item_id, pred_rating in user_recommendations:
        movie_title = movies_df[movies_df['item_id'] == item_id]['title'].iloc[0]
        print(f"- {movie_title}: predicted rating {pred_rating:.2f}")
    
    print("\nItem-based collaborative filtering recommendations:")
    item_recommendations = item_cf.recommend(test_user_id, n_recommendations=5)
    for item_id, pred_rating in item_recommendations:
        movie_title = movies_df[movies_df['item_id'] == item_id]['title'].iloc[0]
        print(f"- {movie_title}: predicted rating {pred_rating:.2f}")

if __name__ == "__main__":
    main() 