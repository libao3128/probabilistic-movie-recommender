import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class KNNRecommender:
    def __init__(self, n_neighbors=5, method='user'):
        """
        Initialize KNN recommender
        Args:
            n_neighbors: Number of neighbors to consider
            method: 'user' or 'item', indicating user-based or item-based collaborative filtering
        """
        self.n_neighbors = n_neighbors
        self.method = method
        self.ratings_matrix = None
        self.similarity_matrix = None
        
    def fit(self, ratings):
        """
        Train the model
        Args:
            ratings: DataFrame containing user_id, item_id, rating
        """
        # Create user-item rating matrix
        self.ratings_matrix = pd.pivot_table(
            ratings, 
            values='rating',
            index='user_id',
            columns='item_id',
            fill_value=0
        )
        
        # Calculate similarity matrix
        if self.method == 'user':
            self.similarity_matrix = cosine_similarity(self.ratings_matrix)
        else:  # item-based
            self.similarity_matrix = cosine_similarity(self.ratings_matrix.T)
            
    def predict(self, user_id, item_id):
        """
        Predict rating for a user-item pair
        Args:
            user_id: User ID
            item_id: Item ID
        Returns:
            Predicted rating
        """
        if self.method == 'user':
            return self._predict_user_based(user_id, item_id)
        else:
            return self._predict_item_based(user_id, item_id)
    
    def _predict_user_based(self, user_id, item_id):
        """User-based collaborative filtering prediction"""
        if user_id not in self.ratings_matrix.index or item_id not in self.ratings_matrix.columns:
            return 0
            
        # Get user index
        user_idx = self.ratings_matrix.index.get_loc(user_id)
        
        # Get K most similar users
        similar_users = np.argsort(self.similarity_matrix[user_idx])[-self.n_neighbors-1:-1][::-1]
        
        # Calculate weighted average rating
        numerator = 0
        denominator = 0
        
        for similar_user_idx in similar_users:
            similarity = self.similarity_matrix[user_idx, similar_user_idx]
            rating = self.ratings_matrix.iloc[similar_user_idx][item_id]
            
            numerator += similarity * rating
            denominator += abs(similarity)
            
        if denominator == 0:
            return 0
            
        return numerator / denominator
    
    def _predict_item_based(self, user_id, item_id):
        """Item-based collaborative filtering prediction"""
        if user_id not in self.ratings_matrix.index or item_id not in self.ratings_matrix.columns:
            return 0
            
        # Get item index
        item_idx = self.ratings_matrix.columns.get_loc(item_id)
        
        # Get user's rated items
        user_ratings = self.ratings_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        
        if len(rated_items) == 0:
            return 0
            
        # Get K most similar items
        similar_items = np.argsort(self.similarity_matrix[item_idx])[-self.n_neighbors-1:-1][::-1]
        
        # Calculate weighted average rating
        numerator = 0
        denominator = 0
        
        for similar_item_idx in similar_items:
            similarity = self.similarity_matrix[item_idx, similar_item_idx]
            rating = user_ratings.iloc[similar_item_idx]
            
            numerator += similarity * rating
            denominator += abs(similarity)
            
        if denominator == 0:
            return 0
            
        return numerator / denominator
    
    def recommend(self, user_id, n_recommendations=5):
        """
        Generate recommendations for a user
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to return
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if user_id not in self.ratings_matrix.index:
            return []
            
        # Get user's unrated items
        user_ratings = self.ratings_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index
        
        # Predict ratings for unrated items
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
            
        # Sort by predicted rating and return top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations] 