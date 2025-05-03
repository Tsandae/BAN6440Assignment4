# BAN6440Assignment4
# COVID-19 K-Means Clustering Application

This project uses the K-Means clustering algorithm to analyze and group states based on the number of confirmed COVID-19 cases and deaths. It helps identify patterns and trends in the spread and impact of the virus.

## Features

- Loads and processes a COVID-19 dataset
- Applies K-Means clustering to group states with similar case/death statistics
- Uses the Elbow Method to determine the optimal number of clusters
- Provides clear visualizations of clustering results
- Includes unit tests to ensure code reliability

## Dataset

The dataset used (`01-01-2022.csv`) contains global COVID-19 statistics from January 1, 2022. The relevant features are `Confirmed` cases and `Deaths`.

## Requirements

- Python 3.8+
- pandas
- matplotlib
- scikit-learn
- unittest
