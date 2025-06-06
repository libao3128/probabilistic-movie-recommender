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

The following table summarizes the performance (on the test set) of different recommendation models evaluated in this project:

| Model                                     | RMSE   | MAE    |
|-------------------------------------------|--------|--------|
| Bayesian PMF                              | 0.8642 | 0.6551 |
| Probabilistic Matrix Factorization (PMF)  | 0.9043 | 0.6915 |
| HBM (Hierarchical Bayesian Model)         | 0.9380 | 0.7406 |
| SPN (Sum-Product Network)                 | 1.3454 | 0.9134 |
| Frequent Pattern Mining + MLN             | 1.2094 | 0.8935 |

> Note: Lower RMSE and MAE indicate better predictive accuracy.


## 📝 Tasks

- [x] Set up the pipeline for training and evaluating different models **[Leo]**
- [x] Implement Probabilistic Matrix Factorization **[Leo]**
- [x] Develop Sum-Product Networks (SPN) **[Angela]**
- [x] Explore Probabilistic Programming approaches **[SSC]**
- [x] 5/27 Project Check-in **[ALL]**
- [x] Hierarchical Bayesian Model **[Leo]**
- [x] Frequent Pattern Mining + Markov Logic Network **[Leo]**
