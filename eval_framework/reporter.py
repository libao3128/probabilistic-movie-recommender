import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

class BaseRecommenderReporter:
    """
    Base class for visualizing and reporting results of recommender system models.
    Usage:
        reporter = BaseRecommenderReporter()
        results = reporter.evaluate_predictions(model, test_ratings)
        reporter.plot_all(results)
        reporter.print_stats(results)
    """
    def __init__(self):
        """
        Base reporter for recommender systems.
        """
        self.model = None
        self.results = None

    def evaluate_model(self, model, test_ratings):
        """Evaluate model predictions"""
        self.model = model
        
        print("\nEvaluating predictions...")
        predictions = []
        actuals = []
        start_time = time.time()
        
        for _, row in tqdm(test_ratings.iterrows(), total=len(test_ratings), desc="Evaluating ratings"):
            pred = model.predict(row['userId'], row['movieId'])
            predictions.append(pred)
            actuals.append(row['rating'])

        end_time = time.time()
        
        return self._calculate_metrics(predictions, actuals, start_time-end_time)

    def _calculate_metrics(self, predictions, actuals, inference_times=None):
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        self.results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'predictions': predictions,
            'actuals': actuals
        }
        
        # Add inference time metrics if provided
        if inference_times is not None:
            self.results.update({
                'inference_times': inference_times,
                'inference_time_per_prediction': inference_times/len(predictions)
            })
        
        return self.results
    
    def plot_all(self, save_path='recommender_results.png'):
        """Plot evaluation results (universal for all models)."""
        print("\nGenerating universal plots...")
        
        # Determine number of plots based on available data
        n_plots = 3

            
        fig = plt.figure(figsize=(20, 5) if n_plots == 4 else (15, 5))
        gs = fig.add_gridspec(1, n_plots)

        # Plot 1: Actual vs Predicted
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(self.results['actuals'], self.results['predictions'], alpha=0.5)
        ax1.plot([0, 5], [0, 5], 'r--')
        ax1.set_xlabel('Actual Ratings')
        ax1.set_ylabel('Predicted Ratings')
        ax1.set_title('Actual vs Predicted Ratings')

        # Plot 2: Error Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        errors = np.array(self.results['predictions']) - np.array(self.results['actuals'])
        sns.histplot(errors, kde=True, ax=ax2)
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Prediction Errors')

        # Plot 3: Rating Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        sns.histplot(self.results['actuals'], kde=True, ax=ax3, label='Actual')
        sns.histplot(self.results['predictions'], kde=True, ax=ax3, label='Predicted')
        ax3.set_xlabel('Rating')
        ax3.set_ylabel('Count')
        ax3.set_title('Distribution of Ratings')
        ax3.legend()

        plt.tight_layout()
        plt.show()
        plt.close()  

    def print_stats(self):
        """Print universal evaluation metrics."""
        print("\nUniversal Evaluation Metrics:")
        if 'mse' in self.results:
            print(f"MSE: {self.results['mse']:.4f}")
        if 'rmse' in self.results:
            print(f"RMSE: {self.results['rmse']:.4f}")
        if 'mae' in self.results:
            print(f"MAE: {self.results['mae']:.4f}")
        if 'inference_time_per_prediction' in self.results:
            print(f"Inference Time per Prediction: {self.results['inference_time_per_prediction']:.4f} seconds")

class HBMRecommenderReporter(BaseRecommenderReporter):
    """
    Reporter for HBM models, adds parameter distribution plots and statistics.
    Usage:
        reporter = HBMRecommenderReporter(model, movies, results)
        reporter.plot_all()
        reporter.print_stats()
    """
    def __init__(self):
        super().__init__()
        self.trace = None
        self.mu_mean = None
        self.alpha_mean = None
        self.beta_mean = None
        
    def evaluate_model(self, model, test_ratings):
        super().evaluate_model(model, test_ratings)
        self.trace = model.trace
        self.mu_mean = self.trace.posterior['mu'].mean(dim=("chain", "draw")).values
        self.alpha_mean = self.trace.posterior['alpha'].mean(dim=("chain", "draw")).values
        self.beta_mean = self.trace.posterior['beta'].mean(dim=("chain", "draw")).values

    def plot_all(self, save_path='hbm_results.png'):
        """Plot universal and HBM-specific results."""
        # First plot universal
        super().plot_all(save_path=save_path)
        # Then add HBM-specific parameter plot
        print("Adding HBM parameter distribution plot...")
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        sns.kdeplot(data=self.alpha_mean, ax=ax, label='User Bias (α)')
        sns.kdeplot(data=self.beta_mean, ax=ax, label='Movie Bias (β)')
        ax.axvline(x=self.mu_mean, color='r', linestyle='--', label=f'Global Mean (μ) = {self.mu_mean:.2f}')
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of HBM Model Parameters')
        ax.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    def print_stats(self, movies: pd.DataFrame = None):
        super().print_stats()
        print("\nHBM Model Parameter Statistics:")
        print(f"Global Mean (μ): {self.mu_mean:.3f}")
        print(f"User Bias (α) - Mean: {np.mean(self.alpha_mean):.3f}, Std: {np.std(self.alpha_mean):.3f}")
        print(f"Movie Bias (β) - Mean: {np.mean(self.beta_mean):.3f}, Std: {np.std(self.beta_mean):.3f}")
        
        if movies is not None:
            movie_bias = pd.DataFrame({
                'movieId': list(self.model.movie_map.keys()),
                'bias': self.beta_mean
            })

            movie_bias = movie_bias.merge(movies[['movieId', 'title']], on='movieId')
            print("\nTop 5 Movies by Positive Bias:")
            print(movie_bias.sort_values('bias', ascending=False).head().to_string())
            print("\nTop 5 Movies by Negative Bias:")
            print(movie_bias.sort_values('bias').head().to_string()) 
            
class MLNRecommenderReporter(BaseRecommenderReporter):
    """
    Reporter for MLN models, adds parameter distribution plots and statistics.
    Usage:
        reporter = MLNRecommenderReporter(model, movies, results)
        reporter.plot_all()
        reporter.print_stats()
    """
    def __init__(self):
        super().__init__()

    def evaluate_model(self, model, test_ratings, batch_size=1000, n_jobs=-1):
        self.model = model
        
        print("\nEvaluating predictions...")
        start_time = time.time()
        predictions = model.predict(test_ratings, batch_size=batch_size, n_jobs=n_jobs)
        end_time = time.time()
        
        return self._calculate_metrics(predictions['rating'], test_ratings['rating'], start_time-end_time)
        
class PMFRecommenderReporter(BaseRecommenderReporter):
    """
    Reporter for PMF models, adds parameter distribution plots and statistics.
    Usage:
        reporter = PMFRecommenderReporter(model, movies, results)
        reporter.plot_all()
        reporter.print_stats()
    """
    def __init__(self):
        super().__init__()
        
    def evaluate_model(self, model, test_ratings, batch_size=1000, n_jobs=-1):
        self.model = model
        
        print("\nEvaluating predictions...")
        start_time = time.time()
        predictions = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
        pred_cache = {}
        for i, row in tqdm(test_ratings.iterrows(), total=len(test_ratings), desc="Evaluating ratings"):
            if row['userId'] not in pred_cache:
                pred_cache[row['userId']] = model.predict(row['userId'])
            predictions.loc[i] = {
                'userId': row['userId'], 'movieId': row['movieId'], 'rating': pred_cache[row['userId']][int(row['movieId'])]}
        end_time = time.time()
        
        return self._calculate_metrics(predictions['rating'], test_ratings['rating'], start_time-end_time)
        
    def plot_all(self, save_path='pmf_results.png'):
        """Plot universal and PMF-specific results."""
        # First plot universal
        super().plot_all(save_path=save_path)

        # Check performance by plotting train and test errors
        plt.plot(range(self.model.epoch), self.model.rmse_train, marker='o', label='Training Data')
        plt.plot(range(self.model.epoch), self.model.rmse_test, marker='v', label='Test Data')
        plt.title('The MovieLens Dataset Learning Curve')
        plt.xlabel('Number of Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()