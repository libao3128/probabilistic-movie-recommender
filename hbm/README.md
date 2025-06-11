# Hierarchical Bayesian Recommender System

A probabilistic recommendation system based on hierarchical Bayesian modeling using PyMC for movie rating prediction and recommendation generation.

## Table of Contents
- [Overview](#overview)
- [Mathematical Formulation](#mathematical-formulation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Model Configuration](#model-configuration)
- [Performance Considerations](#performance-considerations)

## Overview

The Hierarchical Bayesian Recommender (HBM) implements a three-level Bayesian hierarchy to model user-movie rating relationships:

- **Level 3 (Hyperpriors)**: Controls the variance of user and movie biases
- **Level 2 (Population Parameters)**: Models global mean rating and individual user/movie biases
- **Level 1 (Likelihood)**: Observes actual ratings with Gaussian noise

This approach provides uncertainty quantification, handles sparse data well, and offers interpretable parameters through the bias terms.

## Mathematical Formulation

### Model Hierarchy

**Level 3: Hyperpriors**
```
σ_α ~ HalfNormal(σ_α_prior)     # User bias variance
σ_β ~ HalfNormal(σ_β_prior)     # Movie bias variance  
σ   ~ HalfNormal(σ_prior)       # Observation noise
```

**Level 2: Population Parameters**
```
μ     ~ Normal(μ_mean, μ_σ)                    # Global mean rating
α_u   ~ Normal(0, σ_α)  for u ∈ Users         # User bias
β_m   ~ Normal(0, σ_β)  for m ∈ Movies        # Movie bias
```

**Level 1: Likelihood**
```
r_um  ~ Normal(μ + α_u + β_m, σ)              # Observed rating
```

### Prediction Formula

For a user `u` and movie `m`, the predicted rating is:
```
ŷ_um = μ + α_u + β_m
```

Where:
- `μ`: Global mean rating across all users and movies
- `α_u`: User-specific bias (deviation from global mean)
- `β_m`: Movie-specific bias (deviation from global mean)

## Installation

```bash
pip install pymc pandas numpy scikit-learn matplotlib seaborn tqdm
```

## Quick Start

```python
import pandas as pd
from hbm import HierarchicalBayesianRecommender
from sklearn.model_selection import train_test_split

# Load your data
ratings = pd.read_csv('ratings.csv')  # columns: ['userId', 'movieId', 'rating']
movies = pd.read_csv('movies.csv')    # columns: ['movieId', 'title', 'genres']

# Split data
train_ratings, test_ratings = train_test_split(ratings, test_size=0.2)

# Initialize and train model
model = HierarchicalBayesianRecommender(
    chains=4,
    draws=1000,
    tune=1000,
    target_accept=0.9
)

model.fit(train_ratings, movies)

# Make predictions
prediction = model.predict(user_id=123, movie_id=456)

# Get recommendations
recommendations = model.get_recommendations(user_id=123, n_recommendations=10)
```

## API Reference

### Class: `HierarchicalBayesianRecommender`

#### Constructor

```python
HierarchicalBayesianRecommender(
    chains=4,
    target_accept=0.9,
    draws=1000,
    tune=1000,
    sigma_alpha=10,
    sigma_beta=10,
    sigma=1,
    mu_mean=3.0,
    mu_sigma=1
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chains` | int | 4 | Number of MCMC chains to run |
| `target_accept` | float | 0.9 | Target acceptance rate for NUTS sampler (0.8-0.99) |
| `draws` | int | 1000 | Number of posterior samples per chain |
| `tune` | int | 1000 | Number of tuning steps per chain |
| `sigma_alpha` | float | 10 | Prior scale for user bias variance |
| `sigma_beta` | float | 10 | Prior scale for movie bias variance |
| `sigma` | float | 1 | Prior scale for observation noise |
| `mu_mean` | float | 3.0 | Prior mean for global rating mean |
| `mu_sigma` | float | 1 | Prior scale for global rating mean |

#### Methods

##### `fit(ratings_df, movies_df)`

Trains the model using MCMC sampling.

**Parameters:**
- `ratings_df` (DataFrame): Training ratings with columns `['userId', 'movieId', 'rating']`
- `movies_df` (DataFrame): Movie information with columns `['movieId', 'title', 'genres']`

**Returns:** None

##### `predict(user_id, movie_id)`

Predicts rating for a user-movie pair.

**Parameters:**
- `user_id` (int): User identifier
- `movie_id` (int): Movie identifier

**Returns:** 
- `float`: Predicted rating

**Raises:**
- `ValueError`: If model not fitted or user/movie not in training data

##### `get_recommendations(user_id, n_recommendations=10)`

Generates top-N movie recommendations for a user.

**Parameters:**
- `user_id` (int): User identifier
- `n_recommendations` (int): Number of recommendations to return

**Returns:**
- `list`: List of `(movie_id, predicted_rating)` tuples, sorted by rating

#### Properties

##### `is_fitted`
Returns `True` if model has been trained, `False` otherwise.

## Examples

### Basic Usage

```python
# Initialize with default parameters
model = HierarchicalBayesianRecommender()
model.fit(train_ratings, movies)

# Make prediction
rating = model.predict(user_id=1, movie_id=10)
print(f"Predicted rating: {rating:.2f}")
```

### Custom Configuration

```python
# For high-precision applications
model = HierarchicalBayesianRecommender(
    chains=6,           # More chains for better convergence
    draws=2000,         # More samples for precision
    tune=2000,          # More tuning
    target_accept=0.95, # Higher acceptance rate
    sigma_alpha=5,      # Tighter user bias prior
    sigma_beta=5        # Tighter movie bias prior
)
```

### Fast Prototyping

```python
# For quick experimentation
model = HierarchicalBayesianRecommender(
    chains=2,
    draws=500,
    tune=500,
    target_accept=0.85
)
```

### Handling Cold Start

```python
# For new users/movies, the model will raise ValueError
try:
    rating = model.predict(new_user_id, movie_id)
except ValueError as e:
    print(f"Cannot predict: {e}")
    # Use fallback strategy (e.g., global mean)
```

## Model Configuration

### Choosing Hyperparameters

**MCMC Parameters:**
- **chains**: 2-6 chains typical. More chains improve convergence detection.
- **draws**: 1000-5000. More draws give better posterior estimates.
- **tune**: Usually equal to draws. More tuning improves sampler efficiency.
- **target_accept**: 0.8-0.99. Higher values reduce step size but improve accuracy.

**Prior Parameters:**
- **sigma_alpha/sigma_beta**: Controls user/movie bias variability
  - Smaller values (1-5): Assume users/movies are similar
  - Larger values (10-20): Allow more individual variation
- **mu_mean**: Set to expected rating scale mean (e.g., 3.0 for 1-5 scale)
- **sigma**: Controls noise level in observations

### Convergence Diagnostics

```python
# Check R-hat values (should be < 1.1)
import arviz as az
az.summary(model.trace)

# Plot traces
az.plot_trace(model.trace)
```

## Performance Considerations

### Computational Complexity
- **Time**: O(iterations × (users + movies + ratings))
- **Memory**: O(users + movies + chains × samples)

### Scalability Guidelines

| Dataset Size | Recommended Configuration |
|--------------|---------------------------|
| Small (<10K ratings) | Default settings |
| Medium (10K-100K) | `chains=4, draws=1000` |
| Large (100K-1M) | `chains=2, draws=500` |
| Very Large (>1M) | Consider sampling users/movies |

### Optimization Tips

1. **Data Sampling**: For large datasets, sample a subset of users for faster training
```python
# Sample 1000 users for faster training
sampled_users = ratings['userId'].drop_duplicates().sample(1000)
train_subset = ratings[ratings['userId'].isin(sampled_users)]
```

2. **Parallel Chains**: Ensure your system can handle multiple chains
```python
# Reduce chains if memory constrained
model = HierarchicalBayesianRecommender(chains=2)
```

3. **Early Stopping**: Monitor convergence and stop if R-hat < 1.01

## Model Advantages

✅ **Uncertainty Quantification**: Provides prediction intervals  
✅ **Handles Sparsity**: Works well with limited user-movie interactions  
✅ **Interpretable**: Clear bias parameters for users and movies  
✅ **Robust**: Hierarchical structure prevents overfitting  
✅ **Flexible**: Easy to extend with additional features  

## Model Limitations

⚠️ **Computational Cost**: MCMC sampling is slower than matrix factorization  
⚠️ **Cold Start**: Cannot predict for new users/movies  
⚠️ **Linear Model**: Only captures additive effects, no user-movie interactions  
⚠️ **Scalability**: Memory usage grows with number of users/movies  

## Comparison with Other Methods

| Method | Speed | Accuracy | Interpretability | Cold Start |
|--------|-------|----------|------------------|------------|
| HBM | Slow | Medium | High | Poor |
| Matrix Factorization | Fast | High | Low | Poor |
| Content-Based | Medium | Medium | Medium | Good |
| Neural Networks | Fast | High | Low | Medium |

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{hierarchical_bayesian_recommender,
  title={Hierarchical Bayesian Recommender System},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
``` 