import pandas as pd
import numpy as np
from collections import defaultdict
from PAMI.frequentPattern.basic import FPGrowth as alg
from problog.program import PrologString
from problog import get_evaluatable
from problog.logic import Term, Constant, Var
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import sys
import multiprocessing as mp
from functools import partial

def parse_result(result):
    """
    Parse ProbLog evaluation results.
    
    Args:
        result (dict): ProbLog evaluation result
        
    Returns:
        DataFrame: Parsed results with userId, movieId, probability
    """
    parsed = pd.DataFrame(columns=['userId', 'movieId', 'probability'])
    for key, value in result.items():
        if isinstance(key, Term):
            user = str(key.args[0])
            movie = str(key.args[1])
            parsed.loc[len(parsed)] = {
                'userId': int(user[4:]),  # Extract userId from Term
                'movieId': int(movie[5:]),  # Extract movieId from Term
                'probability': value
            }
    return parsed

def process_batch(batch_world_str):
        """
        Process a batch of queries in parallel.
        
        Args:
            batch_world_str (tuple): Tuple containing batch of queries and world string
        """
        batch, world_str = batch_world_str
        query_str = "\n".join(batch)
        try:
            result = parse_result(
                get_evaluatable().create_from(
                    PrologString(world_str + query_str)
                ).evaluate()
            )
            return result
        except Exception as e:
            print(f"Error in batch: {e}")
            return pd.DataFrame(columns=['userId', 'movieId', 'probability'])
        
class MarkovLogicNetworkwithUserGenreModel:
    """
    Markov Logic Network for Movie Recommendation
    
    This method integrates frequent pattern mining and probabilistic logic programming 
    to build an interpretable, rule-based recommendation model.
    """
    
    def __init__(self, preference_rating_threshold=4.0, preference_threshold=0.05, 
                 min_support=0.01, alpha=0.5):
        """
        Initialize the MLN recommender.
        
        Args:
            preference_rating_threshold (float): Threshold for rating to be considered preference
            preference_threshold (float): Threshold for preference to be considered significant
            min_support (float): Minimum support for frequent patterns
            alpha (float): Base probability for genre preference
            batch_size (int): Batch size for inference
        """
        self.preference_rating_threshold = preference_rating_threshold
        self.preference_threshold = preference_threshold
        self.min_support = min_support
        self.alpha = alpha

        # Genre list
        self.genres = [
            "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
            "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "IMAX"
        ]
        
        # Model components
        self.user_preferences = None
        self.frequent_patterns = None
        self.pattern_dict = {}
        self.transactions = None
        self.movie_stats = None
        self.overall_mean = None
        self.overall_std = None
        
    def extract_user_preferences(self, train_data, movies):
        """
        Extract user preferences from rating data.
        
        Args:
            train_data (DataFrame): Training data with userId, movieId, rating
            movies (DataFrame): Movie data with movieId and genres
        """
        # Filter ratings above threshold
        binary_ratings = train_data[train_data["rating"] >= self.preference_rating_threshold]
        
        # Initialize user preferences matrix
        self.user_preferences = pd.DataFrame(
            index=binary_ratings['userId'].unique(), 
            columns=self.genres, 
            dtype=float
        ).fillna(0)
        
        # Calculate preferences based on movie genres
        for row in binary_ratings.itertuples():
            if row.movieId in movies.index:
                movie_genres = movies.loc[row.movieId, 'genres']
                total_utility = len(movie_genres)
                for genre in movie_genres:
                    if genre in self.user_preferences.columns:
                        self.user_preferences.at[row.userId, genre] += 1/total_utility
        
        # Normalize and binarize preferences
        normalized_user_preferences = self.user_preferences.div(
            self.user_preferences.sum(axis=1), axis=0
        )
        binary_user_preferences = normalized_user_preferences >= self.preference_threshold
        
        # Create transactions for frequent pattern mining
        self.transactions = defaultdict(list)
        for userId in binary_user_preferences.index:
            self.transactions[userId] = binary_user_preferences.columns[
                binary_user_preferences.loc[userId]
            ].tolist()
    
    def mine_frequent_patterns(self):
        """
        Mine frequent patterns from user preferences using FP-Growth algorithm.
        """
        # Convert transactions to DataFrame format for PAMI
        transactions_df = pd.DataFrame(columns=["Transactions"])
        for userId, items in self.transactions.items():
            transactions_df.loc[userId, 'Transactions'] = ",".join(list(set(items)))
        
        # Run FP-Growth algorithm
        obj = alg.FPGrowth(iFile=transactions_df, minSup=self.min_support, sep=',')
        obj.mine()
        
        # Get frequent patterns
        self.frequent_patterns = obj.getPatternsAsDataFrame()
        self.frequent_patterns['Patterns'] = self.frequent_patterns['Patterns'].apply(
            lambda x: ' '.join(sorted(x.split()))
        )
        
        # Create pattern dictionary for rule generation
        self.pattern_dict = {
            row['Patterns']: row['Support'] 
            for index, row in self.frequent_patterns.iterrows()
        }
    
    def generate_logic_rules(self, movies, output_dir='mln'):
        """
        Generate ProbLog rules for the MLN.
        
        Args:
            movies (DataFrame): Movie data with movieId and genres
            output_dir (str): Directory to save the generated files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate world rules
        with open(f'{output_dir}/world.pl', 'w') as f:
            base_prob = 0.9
            # Rules for movies with different number of genres
            for i in range(1, 6):
                f.write(f'{base_prob/i}::likes(U, M):- prefers(U, G), has_genre(M, G), has_{i}_genre(M).\n')
            f.write(f'{base_prob/6.0}::likes(U, M):- prefers(U, G), has_genre(M, G), has_more_than_5_genre(M).\n')
            
            # User preferences facts
            for userId, items in self.transactions.items():
                for item in items:
                    f.write(f'prefers(user{userId}, {item.lower()}).\n')
            
            # Movie genre facts
            movies.index = movies['movieId']
            for movieId, row in movies.iterrows():
                if len(row['genres']) <= 5:
                    f.write(f'has_{len(row["genres"])}_genre(movie{movieId}).\n')
                else:
                    f.write(f'has_more_than_5_genre(movie{movieId}).\n')
                    
                for genre in row['genres']:
                    if genre == '(no genres listed)':
                        continue
                    f.write(f'has_genre(movie{movieId}, {genre.lower()}).\n')
        
        # Generate preference rules from frequent patterns
        with open(f'{output_dir}/preference.pl', 'w') as f:
            # rules 定義
            for row in self.frequent_patterns.itertuples(index=True):
                items = row.Patterns.split()
                if len(items) < 2:
                    continue
        for i in range(len(items)):
            body = ' '.join(sorted(items[:i]+items[i+1:]))
            if body not in self.pattern_dict:
                continue
            print(body)
            total_support = self.pattern_dict[body]
            head = items[i]
            sub_total_support = row.Support
            f.write(f'{sub_total_support/total_support:.2f}::prefers(u, {head.lower()}) :-')
            for item in body.split():
                f.write(f'prefers(u, {item.lower()})')
                if item != body.split()[-1]:
                    f.write(', ')
            f.write(f'.\n')
    
    def calculate_movie_stats(self, data):
        """
        Calculate movie rating statistics for inference.
        
        Args:
            data (DataFrame): Complete rating data
        """
        self.overall_std = data['rating'].std()
        self.overall_mean = data['rating'].mean()
        
        self.movie_stats = data.groupby('movieId')['rating'].agg(['mean', 'std', 'count'])
        
        # Handle movies with few ratings
        self.movie_stats['mean'] = self.movie_stats.apply(
            lambda row: self.overall_mean if row['count'] < 10 else row['mean'], axis=1
        )
        self.movie_stats['std'] = self.movie_stats.apply(
            lambda row: self.overall_std if row['count'] < 10 else row['std'], axis=1
        )
    
    def infer_rating(self, movie_id, prob_likes):
        """
        Convert probability of liking to rating using normal distribution.
        
        Args:
            mu (float): Mean rating for the movie
            sigma (float): Standard deviation of rating for the movie
            prob_likes (float): Probability of liking the movie
            
        Returns:
            float: Inferred rating
        """
        if movie_id in self.movie_stats.index:
            mu = self.movie_stats.loc[movie_id, 'mean']
            sigma = self.movie_stats.loc[movie_id, 'std']
        else:
            mu = self.overall_mean
            sigma = self.overall_std
        p = np.clip(prob_likes, 0.0, 1.0)
        score = np.clip(norm(mu, sigma).ppf(p), 0.5, 5.0)
        return score
    
    def predict(self, test_data, output_dir='mln', batch_size=1000, n_jobs=-1):
        """
        Make predictions for test data.
        
        Args:
            test_data (DataFrame): Test data with userId, movieId, rating
            output_dir (str): Directory containing the generated rules
            batch_size (int): Batch size for inference
            n_jobs (int): Number of parallel jobs. -1 means using all processors
        Returns:
            DataFrame: Predictions with userId, movieId, probability, rating
        """
        # Generate queries
        with open(f'{output_dir}/query.pl', 'w') as f:
            for row in test_data.itertuples():
                f.write(f'query(likes(user{row.userId}, movie{row.movieId})).\n')
        
        # Load rules
        world_str = (open(f'{output_dir}/world.pl').read() + 
                    open(f'{output_dir}/preference.pl').read())
        queries = open(f'{output_dir}/query.pl').read().splitlines()
        
        # Split queries into batches
        batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
        

        
        # Initialize multiprocessing pool
        n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        pool = mp.Pool(processes=n_jobs)
        
        # Process batches in parallel
        prediction = pd.DataFrame(columns=['userId', 'movieId', 'probability'])
        with tqdm(total=len(queries)) as pbar:
            pbar.set_description("Evaluating queries")
            results = []
            # Create iterable where each item is (batch, world_str)
            batch_args = [(batch, world_str) for batch in batches]
            for result in pool.imap_unordered(process_batch, batch_args):
                results.append(result)
                pbar.update(batch_size)
        
        pool.close()
        pool.join()
        
        # Combine results
        prediction = pd.concat(results, ignore_index=True)
        
        # Convert probabilities to ratings
        if not prediction.empty:
            prediction['rating'] = prediction.apply(
                lambda x: self.infer_rating(
                    x.movieId,
                    x.probability
                ), axis=1
            )
            
        # Merge predictions with test data while preserving order
        prediction = pd.merge(
            test_data[['userId', 'movieId']],
            prediction,
            on=['userId','movieId'],
            how='left',
            sort=False
        )
        return prediction
    
    def fit(self, ratings, movies, output_dir='mln'):
        """
        Complete training and prediction pipeline.
        
        Args:
            train_data (DataFrame): Training data
            movies (DataFrame): Movie data
            output_dir (str): Output directory for intermediate files
            
        Returns:
            DataFrame: Predictions
        """
        self.ratings = ratings.copy()
        self.movies = movies.copy()
        self.movies.index = self.movies['movieId']
        
        print("Extracting user preferences...")
        self.extract_user_preferences(self.ratings, self.movies)
        
        print("Mining frequent patterns...")
        self.mine_frequent_patterns()
        
        print("Generating logic rules...")
        self.generate_logic_rules(self.movies, output_dir)
        
        print("Calculating movie statistics...")
        self.calculate_movie_stats(self.ratings)
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate predictions using RMSE and MAE.
        
        Args:
            y_true (array-like): True ratings
            y_pred (array-like): Predicted ratings
            
        Returns:
            dict: Evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Remove NaN values
        mask = ~(pd.isna(y_true) | pd.isna(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'n_predictions': len(y_pred_clean)
        }    
