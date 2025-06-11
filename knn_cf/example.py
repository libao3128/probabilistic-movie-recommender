import pandas as pd
from knn_recommender import KNNRecommender

def load_sample_data():
    """
    Create sample rating data for demonstration
    Returns:
        DataFrame containing sample user-item ratings
    """
    # Create sample rating data
    data = {
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'item_id': [1, 2, 3, 1, 2, 4, 1, 3, 4, 2, 3, 4],
        'rating': [5, 3, 4, 4, 5, 3, 3, 4, 5, 4, 3, 5]
    }
    return pd.DataFrame(data)

def main():
    # Load data
    ratings = load_sample_data()
    
    # Create user-based collaborative filtering recommender
    user_cf = KNNRecommender(n_neighbors=2, method='user')
    user_cf.fit(ratings)
    
    # Create item-based collaborative filtering recommender
    item_cf = KNNRecommender(n_neighbors=2, method='item')
    item_cf.fit(ratings)
    
    # Generate recommendations for user 1
    user_id = 1
    print(f"\nUser-based collaborative filtering recommendations (User {user_id}):")
    user_recommendations = user_cf.recommend(user_id)
    for item_id, pred_rating in user_recommendations:
        print(f"Item {item_id}: predicted rating {pred_rating:.2f}")
    
    print(f"\nItem-based collaborative filtering recommendations (User {user_id}):")
    item_recommendations = item_cf.recommend(user_id)
    for item_id, pred_rating in item_recommendations:
        print(f"Item {item_id}: predicted rating {pred_rating:.2f}")

if __name__ == "__main__":
    main() 