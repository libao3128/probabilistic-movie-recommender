# KNN-based Collaborative Filtering

This project implements a KNN-based collaborative filtering recommendation system, supporting both user-based and item-based approaches.

## Features

- User-based Collaborative Filtering
- Item-based Collaborative Filtering
- Cosine similarity for user/item similarity calculation
- Rating prediction and item recommendation functionality
- Support for MovieLens dataset integration

## Requirements

```bash
pip install -r requirements.txt
```

## Project Structure

- `knn_recommender.py`: Core implementation of the KNN recommender
- `real_data_example.py`: Example using MovieLens dataset
- `requirements.txt`: Project dependencies

## Usage

### Basic Usage

```python
from knn_recommender import KNNRecommender

# Create recommender instance
recommender = KNNRecommender(n_neighbors=20, method='user')  # or method='item'

# Train the model
recommender.fit(ratings_data)

# Get recommendations
recommendations = recommender.recommend(user_id=1, n_recommendations=5)
```

### Using with MovieLens Dataset

```python
# Load and prepare data
ratings_df, movies_df = load_movielens_data()

# Create and train recommenders
user_cf = KNNRecommender(n_neighbors=20, method='user')
user_cf.fit(ratings_df)

item_cf = KNNRecommender(n_neighbors=20, method='item')
item_cf.fit(ratings_df)

# Generate recommendations
recommendations = user_cf.recommend(user_id=1, n_recommendations=5)
```

## Parameters

- `n_neighbors`: Number of neighbors to consider (default: 5)
- `method`: 'user' or 'item', indicating the type of collaborative filtering
- `n_recommendations`: Number of items to recommend (default: 5)

## Data Format

The input data should be a pandas DataFrame with the following columns:
- `user_id`: User identifier
- `item_id`: Item identifier
- `rating`: User's rating for the item

## Example Output

```
Loading MovieLens dataset...
Loading complete! Dataset contains 100836 ratings and 9742 movies

Training user-based collaborative filtering model...
Training item-based collaborative filtering model...

Movies rated by user 1:
- Toy Story (1995): 4.0 stars
- Jumanji (1995): 4.0 stars
- Grumpier Old Men (1995): 3.0 stars

User-based collaborative filtering recommendations:
- Star Wars: Episode IV - A New Hope (1977): predicted rating 4.5
- The Matrix (1999): predicted rating 4.3
...
```

## Notes

- The implementation uses cosine similarity for calculating user/item similarities
- For large datasets, consider adjusting the `n_neighbors` parameter
- The system automatically handles missing ratings and new users/items 