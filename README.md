# Probabilistic Movie Recommender

## 📘 Overview

This project implements a movie recommendation system using the MovieLens dataset, focusing on **probabilistic modeling** techniques. We explore and compare the following models:

- K-Nearest Neighbors (User-based and Item-based)
- Probabilistic Matrix Factorization (PMF)
- Bayesian Probabilistic Matrix Factorization
- Hierarchical Bayesian Model (HBM)
- Frequent Pattern Mining + Markov Logic Network (FP + MLN)
- Sum-Product Network (SPN)

Models are evaluated on both **rating prediction accuracy** (RMSE, MAE), **computational efficiency** (training/inference time), and **interpretability**.  
The goal is to provide insights into the trade-offs between performance, scalability, and explainability for probabilistic recommenders in real-world applications.

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

### 📊 Performance

| Model      | RMSE   | MAE    | Uses Genre? |
|------------|--------|--------|-------------|
| KNN (user) | 2.7536 | 2.4982 | No          |
| KNN (item) | 2.9454 | 2.6990 | No          |
| PMF        | 0.9089 | 0.6950 | No          |
| BPMF       | 0.8531 | 0.6555 | No          |
| HBM        | 0.8745 | 0.6683 | No          |
| SPN        | 1.2404 | 0.8812 | Yes         |
| FPMLN      | 1.2627 | 0.9455 | Yes         |

> Note: Lower RMSE and MAE indicate better predictive accuracy.

---

### ⏳ Cost

| Model      | Training Time (s) | Inference Time (sec per 1000 predictions) |
|------------|-------------------|------------------------------------------|
| KNN (user) | 0.09              | 0.67                                     |
| KNN (item) | 0.34              | 0.76                                     |
| PMF        | 491               | 1.5866                                   |
| BPMF       | 3422              | 8.7762                                   |
| HBM        | 713               | 0.0445                                   |
| SPN        | 212               | 0.1901                                   |
| FPMLN      | 6.5               | 12.0487                                  |

