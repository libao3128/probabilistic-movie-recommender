import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse
#import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class HierarchicalBayesianRecommender:
    def __init__(self, ratings_df, movies_df):
        """
        Hierarchical Bayesian Model for Movie Recommendations
        
        Parameters:
        -----------
        ratings_df : DataFrame
            DataFrame with columns ['userId', 'movieId', 'rating']
        movies_df : DataFrame
            DataFrame with columns ['movieId', 'genres']
        """
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        self.model = None
        self.approx = None
        
        # Initialize preprocessing
        self._preprocess()
        
    def _preprocess(self):
        """Preprocess data and create mappings"""
        # Create user and movie mappings
        self.user_map = {uid: i for i, uid in enumerate(self.ratings_df['userId'].unique())}
        self.movie_map = {mid: i for i, mid in enumerate(self.movies_df['movieId'].unique())}
        
        # Create reverse mappings
        self.user_reverse_map = {v: k for k, v in self.user_map.items()}
        self.movie_reverse_map = {v: k for k, v in self.movie_map.items()}
        
        # Update indices
        self.ratings_df['uidx'] = self.ratings_df['userId'].map(self.user_map)
        self.ratings_df['midx'] = self.ratings_df['movieId'].map(self.movie_map)
        
        # Calculate dimensions
        self.num_users = len(self.user_map)
        self.num_movies = len(self.movie_map)
        
        # Create sparse rating matrix
        self.rating_matrix = sparse.csr_matrix(
            (self.ratings_df['rating'].values,
             (self.ratings_df['uidx'].values, self.ratings_df['midx'].values)),
            shape=(self.num_users, self.num_movies)
        )
    
    def _get_user_ratings(self, user_id):
        """Get all ratings for a specific user"""
        if user_id not in self.user_map:
            return pd.DataFrame()
        u_idx = self.user_map[user_id]
        user_ratings = self.ratings_df[self.ratings_df['uidx'] == u_idx]
        return user_ratings
    
    def _get_movie_ratings(self, movie_id):
        """Get all ratings for a specific movie"""
        if movie_id not in self.movie_map:
            return pd.DataFrame()
        m_idx = self.movie_map[movie_id]
        movie_ratings = self.ratings_df[self.ratings_df['midx'] == m_idx]
        return movie_ratings
    
    def fit(self, n_iter=100, n_samples=100):
        """
        Fit the hierarchical Bayesian model
        
        Parameters:
        -----------
        n_iter : int
            Number of iterations for ADVI
        """
        print("Fitting model...")
        with pm.Model() as self.model:
            # Level 3: Hyperpriors
            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=10)
            sigma_beta = pm.HalfNormal('sigma_beta', sigma=10)
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Level 2: Population parameters
            mu = pm.Normal('mu', mu=3.0, sigma=1)  # Global mean rating
            alpha = pm.Normal('alpha', mu=0, sigma=sigma_alpha, shape=self.num_users)  # User bias
            beta = pm.Normal('beta', mu=0, sigma=sigma_beta, shape=self.num_movies)  # Movie bias
            
            # Level 1: Likelihood
            pm.Normal('ratings', 
                     mu=mu + alpha[self.ratings_df['uidx']] + beta[self.ratings_df['midx']],
                     sigma=sigma,
                     observed=self.ratings_df['rating'].values)
            
            # Use ADVI for faster inference
            self.approx = pm.fit(
                method='advi',
                n=n_iter,
                progressbar=False
            )
            
            # Sample from posterior
            self.trace = self.approx.sample(n_samples)
        print("Model fitting completed.")
    
    def predict(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair using fitted model
        
        Parameters:
        -----------
        user_id : int
            User ID
        movie_id : int
            Movie ID
        n_samples : int
            Number of samples for inference
            
        Returns:
        --------
        float
            Predicted rating
        """
        if self.model is None or self.approx is None:
            raise ValueError("Model must be fitted before making predictions")
            
        if user_id not in self.user_map or movie_id not in self.movie_map:
            raise ValueError("user_id or movie_id not in training data")
            
        # Get indices
        u_idx = self.user_map[user_id]
        m_idx = self.movie_map[movie_id]
        
        trace = self.trace  
        
        # Get posterior means
        mu_mean = trace.posterior['mu'].mean(dim=("chain", "draw")).values
        alpha_mean = trace.posterior['alpha'].mean(dim=("chain", "draw")).values
        beta_mean = trace.posterior['beta'].mean(dim=("chain", "draw")).values
        
        # Calculate prediction using posterior means
        pred = mu_mean + alpha_mean[u_idx] + beta_mean[m_idx]
        
        return float(pred)

    
    def get_recommendations(self, user_id, n_recommendations=10, n_samples=100):
        """
        Get top-N movie recommendations for a user
        
        Parameters:
        -----------
        user_id : int
            User ID
        n_recommendations : int
            Number of recommendations to return
        n_samples : int
            Number of samples for inference
            
        Returns:
        --------
        list
            List of (movie_id, score) tuples
        """
        if self.model is None or self.approx is None:
            raise ValueError("Model must be fitted before making recommendations")
            
        if user_id not in self.user_map:
            raise ValueError("user_id not in training data")
            
        # Get user's rated movies
        user_rated = set(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'])
        all_movies = set(self.movie_map.keys())
        unrated_movies = list(all_movies - user_rated)
        
        # Get predictions for unrated movies
        predictions = []
        for movie_id in tqdm(unrated_movies[:20], desc="Generating recommendations"):
            try:
                pred = self.predict(user_id, movie_id, n_samples)
                predictions.append((movie_id, pred))
            except:
                continue
        
        # Sort and return top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]