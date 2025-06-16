# Probabilistic Movie Recommender

## 📘 Overview

This project implements a movie recommendation system using the MovieLens dataset, focusing on **probabilistic modeling** techniques. We aim to explore various approaches such as:

- Probabilistic Matrix Factorization (PMF)
- Collaborative Filtering (User- and Item-based KNN)
- Content-Based Filtering using genre metadata
- Hybrid models (linear combination and deep hybrid)
- Potential extensions: Bayesian modeling, neural-symbolic reasoning

Evaluation is conducted using both **rating prediction** (RMSE, MAE) and **top-K recommendation** metrics (Precision@K, Recall@K, NDCG@K).

---

## ⚙️ Prerequisites

### Data

MovieLens offers datasets of various sizes. For faster iteration during development, use a smaller dataset; for final experiments, use the full dataset.

#### Final Results

1. Download the MovieLens 32M dataset from the [GroupLens website](https://grouplens.org/datasets/movielens/).
2. Extract it to `data/ml-32m` in your project directory.

#### Development

1. Download the MovieLens "latest small" dataset from the [GroupLens website](https://grouplens.org/datasets/movielens/).
2. Extract it to `data/ml-latest-small` in your project directory.

## 📊 Benchmark Results

The following table summarizes the performance based on "latest small" of different recommendation models evaluated in this project:

### Performance

| Model                                     | RMSE   | MAE    | Use Genre?  |
|-------------------------------------------|--------|--------|-------------|
| Bayesian PMF                              | 0.8531 | 0.6555 | X           |
| Probabilistic Matrix Factorization (PMF)  | 0.9089 | 0.6950 | X           |
| HBM (Hierarchical Bayesian Model)         | 0.8745 | 0.6683 | X           |
| SPN (Sum-Product Network)                 | 1.2404 | 0.8812 | v           |
| Frequent Pattern Mining + MLN             | 1.2627 | 0.9455 | v           |
| KNN-user                                 | 2.7536 | 2.4982 | X           |
| KNN-item                                 | 2.9454 | 2.6990 | X           |

> Note: Lower RMSE and MAE indicate better predictive accuracy.

### Cost

| Model                                     | Training Time (s)| Inference Time (sec per 1000 predictions)| Memory Usage (MB)|
|-------------------------------------------|------------------|-------------------------|------------------|
| Bayesian PMF                              | 3422             | 8.7762                  |                  |
| Probabilistic Matrix Factorization (PMF)  | 491              | 1.5866                  |                  |
| HBM (Hierarchical Bayesian Model)         | 713              | 0.0445                  |                  |
| SPN (Sum-Product Network)                 | 212              | 0.1901                  |                  |
| Frequent Pattern Mining + MLN             | 6.5              | 12.0487                 |                  |
| KNN-user                                 | 0.09             | 0.67                    | 94.28            |
| KNN-item                                 | 0.34             | 0.76                    | 699.48           |

## 📝 Tasks

- [x] Set up the pipeline for training and evaluating different models **[Leo]**
- [x] Implement Probabilistic Matrix Factorization **[Leo]**
- [x] Develop Sum-Product Networks (SPN) **[Angela]**
- [x] Explore Probabilistic Programming approaches **[SSC]**
- [x] 5/27 Project Check-in **[ALL]**
- [x] Hierarchical Bayesian Model **[Leo]**
- [x] Frequent Pattern Mining + Markov Logic Network **[Leo]**
