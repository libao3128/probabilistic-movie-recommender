import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse
#import torch
from tqdm import tqdm
import warnings
import pickle
import json
import os
from pathlib import Path
import arviz as az
warnings.filterwarnings('ignore')

class HierarchicalBayesianRecommender:
    def __init__(self, chains=4, target_accept=0.9, draws=1000, tune=1000,
                 sigma_alpha=10, sigma_beta=10, sigma=1, mu_mean=3.0, mu_sigma=1):
        """
        Hierarchical Bayesian Model for Movie Recommendations
        
        Parameters:
        -----------
        chains : int, default=3
            Number of MCMC chains to run
        target_accept : float, default=0.9
            Target acceptance rate for NUTS sampler
        draws : int, default=1000
            Number of posterior samples to draw per chain
        tune : int, default=1000
            Number of tuning steps per chain
        sigma_alpha : float, default=10
            Prior scale for user bias variance
        sigma_beta : float, default=10
            Prior scale for movie bias variance
        sigma : float, default=1
            Prior scale for observation noise
        mu_mean : float, default=3.0
            Prior mean for global rating mean
        mu_sigma : float, default=1
            Prior scale for global rating mean
        """
        # Data attributes (will be initialized during fit)
        self.ratings_df = None
        self.movies_df = None
        self.model = None
        self.trace = None
        self._posterior_means = None
        
        # Training hyperparameters
        self.chains = chains
        self.target_accept = target_accept
        self.draws = draws
        self.tune = tune
        
        # Model hyperparameters
        self.sigma_alpha = sigma_alpha
        self.sigma_beta = sigma_beta
        self.sigma = sigma
        self.mu_mean = mu_mean
        self.mu_sigma = mu_sigma
        
        # Mapping attributes (will be initialized during fit)
        self.user_map = None
        self.movie_map = None
        self.user_reverse_map = None
        self.movie_reverse_map = None
        self.num_users = None
        self.num_movies = None
        self.rating_matrix = None
        
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
    
    def fit(self, ratings_df, movies_df):
        """
        Fit the hierarchical Bayesian model using MCMC (NUTS)
        
        Parameters:
        -----------
        ratings_df : DataFrame
            DataFrame with columns ['userId', 'movieId', 'rating']
        movies_df : DataFrame
            DataFrame with columns ['movieId', 'genres']
        """
        # Store and preprocess data
        self.ratings_df = ratings_df.copy()
        self.movies_df = movies_df.copy()
        self._preprocess()
        
        print(f"Fitting model with MCMC...")
        print(f"Configuration: chains={self.chains}, draws={self.draws}, tune={self.tune}, target_accept={self.target_accept}")
        
        with pm.Model() as self.model:
            # Level 3: Hyperpriors
            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=self.sigma_alpha)
            sigma_beta = pm.HalfNormal('sigma_beta', sigma=self.sigma_beta)
            sigma = pm.HalfNormal('sigma', sigma=self.sigma)
            
            # Level 2: Population parameters
            mu = pm.Normal('mu', mu=self.mu_mean, sigma=self.mu_sigma)  # Global mean rating
            alpha = pm.Normal('alpha', mu=0, sigma=sigma_alpha, shape=self.num_users)  # User bias
            beta = pm.Normal('beta', mu=0, sigma=sigma_beta, shape=self.num_movies)  # Movie bias
            
            # Level 1: Likelihood
            pm.Normal('ratings', 
                     mu=mu + alpha[self.ratings_df['uidx']] + beta[self.ratings_df['midx']],
                     sigma=sigma,
                     observed=self.ratings_df['rating'].values)
            
            # Use MCMC (NUTS) for inference
            self.trace = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                target_accept=self.target_accept,
                progressbar=True,
                return_inferencedata=True
            )
        trace = self.trace
        
        self._posterior_means = {
                'mu': trace.posterior['mu'].mean(dim=("chain", "draw")).values,
                'alpha': trace.posterior['alpha'].mean(dim=("chain", "draw")).values,
                'beta': trace.posterior['beta'].mean(dim=("chain", "draw")).values
            }
        
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
            
        Returns:
        --------
        float
            Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        if user_id not in self.user_map or movie_id not in self.movie_map:
            raise ValueError("user_id or movie_id not in training data")
        
        assert hasattr(self, '_posterior_means'), "Posterior means not calculated"
            
        # Get indices
        u_idx = self.user_map[user_id]
        m_idx = self.movie_map[movie_id]
        
        # Get posterior means
        mu_mean = self._posterior_means['mu']
        alpha_mean = self._posterior_means['alpha']
        beta_mean = self._posterior_means['beta']
        
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
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
            
        if user_id not in self.user_map:
            raise ValueError("user_id not in training data")
        
        assert hasattr(self, '_posterior_means'), "Posterior means not calculated"
            
        # Get user's rated movies
        user_rated = set(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'])
        all_movies = set(self.movie_map.keys())
        unrated_movies = list(all_movies - user_rated)
        
        # Get predictions for unrated movies
        predictions = []
        for movie_id in tqdm(unrated_movies[:20], desc="Generating recommendations"):
            try:
                pred = self.predict(user_id, movie_id)
                predictions.append((movie_id, pred))
            except:
                continue
        
        # Sort and return top-N
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]
    
    def save(self, filepath):
        """
        Save the trained model to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save the model (without extension)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        filepath = Path(filepath)
        
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save trace using arviz
        trace_path = str(filepath) + "_trace.nc"
        self.trace.to_netcdf(trace_path)
        
        # Save model state (everything except trace and model)
        model_state = {
            'chains': self.chains,
            'target_accept': self.target_accept,
            'draws': self.draws,
            'tune': self.tune,
            'sigma_alpha': self.sigma_alpha,
            'sigma_beta': self.sigma_beta,
            'sigma': self.sigma,
            'mu_mean': self.mu_mean,
            'mu_sigma': self.mu_sigma,
            'user_map': self.user_map,
            'movie_map': self.movie_map,
            'user_reverse_map': self.user_reverse_map,
            'movie_reverse_map': self.movie_reverse_map,
            'num_users': self.num_users,
            'num_movies': self.num_movies,
            '_posterior_means': self._posterior_means
        }
        
        state_path = str(filepath) + "_state.pkl"
        with open(state_path, 'wb') as f:
            pickle.dump(model_state, f)
        
        # Save data frames
        data_path = str(filepath) + "_data.pkl"
        data_dict = {
            'ratings_df': self.ratings_df,
            'movies_df': self.movies_df
        }
        with open(data_path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"Model saved to {filepath}")
        print(f"Files created:")
        print(f"  - {trace_path}")
        print(f"  - {state_path}")
        print(f"  - {data_path}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a trained model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model (without extension)
            
        Returns:
        --------
        HierarchicalBayesianRecommender
            Loaded model instance
        """
        filepath = Path(filepath)
        
        # Check if all required files exist
        trace_path = str(filepath) + "_trace.nc"
        state_path = str(filepath) + "_state.pkl"
        data_path = str(filepath) + "_data.pkl"
        
        for path in [trace_path, state_path, data_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required file not found: {path}")
        
        # Load model state
        with open(state_path, 'rb') as f:
            model_state = pickle.load(f)
        
        # Create new instance with loaded hyperparameters
        model = cls(
            chains=model_state['chains'],
            target_accept=model_state['target_accept'],
            draws=model_state['draws'],
            tune=model_state['tune'],
            sigma_alpha=model_state['sigma_alpha'],
            sigma_beta=model_state['sigma_beta'],
            sigma=model_state['sigma'],
            mu_mean=model_state['mu_mean'],
            mu_sigma=model_state['mu_sigma']
        )
        
        model.model = pm.Model()
        
        # Load trace
        model.trace = az.from_netcdf(trace_path)
        
        # Load data
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        model.ratings_df = data_dict['ratings_df']
        model.movies_df = data_dict['movies_df']
        
        # Restore mappings and other attributes
        model.user_map = model_state['user_map']
        model.movie_map = model_state['movie_map']
        model.user_reverse_map = model_state['user_reverse_map']
        model.movie_reverse_map = model_state['movie_reverse_map']
        model.num_users = model_state['num_users']
        model.num_movies = model_state['num_movies']
        model._posterior_means = model_state['_posterior_means']
        
        # Recreate sparse rating matrix
        model.rating_matrix = sparse.csr_matrix(
            (model.ratings_df['rating'].values,
             (model.ratings_df['uidx'].values, model.ratings_df['midx'].values)),
            shape=(model.num_users, model.num_movies)
        )
        
        
        print(f"Model loaded from {filepath}")
        return model
    
    def get_model_info(self):
        """
        Get information about the model configuration and training
        
        Returns:
        --------
        dict
            Model information
        """
        info = {
            'model_type': 'Hierarchical Bayesian Recommender',
            'is_fitted': self.is_fitted,
            'hyperparameters': {
                'chains': self.chains,
                'target_accept': self.target_accept,
                'draws': self.draws,
                'tune': self.tune,
                'sigma_alpha': self.sigma_alpha,
                'sigma_beta': self.sigma_beta,
                'sigma': self.sigma,
                'mu_mean': self.mu_mean,
                'mu_sigma': self.mu_sigma
            }
        }
        
        if self.is_fitted:
            info.update({
                'data_info': {
                    'num_users': self.num_users,
                    'num_movies': self.num_movies,
                    'num_ratings': len(self.ratings_df),
                    'sparsity': 1 - (len(self.ratings_df) / (self.num_users * self.num_movies))
                },
                'posterior_stats': {
                    'global_mean': float(self._posterior_means['mu']),
                    'user_bias_std': float(np.std(self._posterior_means['alpha'])),
                    'movie_bias_std': float(np.std(self._posterior_means['beta']))
                }
            })
        
        return info
    
    @property
    def is_fitted(self):
        return self.model is not None and self.trace is not None