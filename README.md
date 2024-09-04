# Stock-Prices-Analysis-EDA-Modeling
📊 Stock Prices Analysis of MasterCard and Visa (2008-2024) 💳

This project involves analyzing stock prices for MasterCard and Visa from 2008 to 2024, including exploratory data analysis (EDA) and predictive modeling.

---

## Project Overview

- **Dataset:** Stock prices data for MasterCard and Visa from Kaggle. [Download here](https://www.kaggle.com/datasets/ranatalha71/stock-prices-of-mastercard-and-visa-2008-2024).

- **Project Goal:** Analyze stock prices, explore trends, and build predictive models to forecast future stock prices.
- **Model Performance:** (Include performance metrics if applicable)

---

## Repository Structure

- **`notebooks/`**
  - **`visa & mastercard stockmarket.ipyn:`** Jupyter Notebook containing the  workflow, from data preprocessing and exploratory data analysis
  - **`stockpredictionUsing LSTM.ipynb.ipyn:`** Jupyter Notebook containing the entire workflow, from data preprocessing and exploratory data analysis to model training and prediction.
 
- **`data/`**
  - **`MVR.csv:`** Raw dataset containing historical stock prices for MasterCard and Visa.

---

## Tools and Libraries Used

- **Pandas:** For data manipulation and preprocessing.
- **NumPy:** For numerical operations and array handling.
- **Matplotlib/Seaborn:** For data visualization and exploratory data analysis.
- **Scikit-learn:** For building and evaluating machine learning models.

---

## Steps Followed in the Project

### 1. Data Loading and Exploration
- Load the dataset into Pandas DataFrames.
- Explore key features, such as stock prices over time and distributions.



###  2.Data Preprocessing
Visualize stock prices for MasterCard and Visa over time.
Analyze trends and patterns.
python

import matplotlib.pyplot as plt
import seaborn as sns
### 3. Exploratory Data Analysis (EDA)
- Visualize stock prices for MasterCard and Visa over time.

### 4. Predictive Modeling
Build and evaluate machine learning models to forecast future stock prices.
python

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Prepare data for modeling
X = data[['Open_M', 'High_M', 'Low_M', 'Volume_M']]
y = data['Close_M']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

Handle missing values and perform feature engineering .
Scale features for consistency.

### Model Performance
The performance of the predictive model was evaluated using the following metrics:

**Mean Squared Error (MSE):** Measures the average squared difference between the predicted and actual values. Lower MSE indicates better model performance.

**R-squared Score:** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. An R-squared score close to 1 indicates a better fit of the model to the data.
These metrics provide insights into the model’s accuracy and its ability to explain the variability in stock prices. The lower the MSE and the higher the R-squared score, the better the model’s performance.

## Motivation

The motivation behind this project was to gain insights into the stock market behavior of two major financial companies, MasterCard and Visa, over a substantial period. By analyzing their stock prices, we can uncover trends and patterns that could be valuable for investors, analysts, and financial enthusiasts.

## Purpose

This project was built to explore historical stock price data for MasterCard and Visa, perform exploratory data analysis (EDA), and develop predictive models to forecast future stock prices. It aims to provide a deeper understanding of the stock price dynamics of these companies.

## Problem Statement

Investors and analysts often seek to understand stock price trends and predict future movements to make informed decisions. This project addresses the need for a thorough analysis and forecasting model for the stock prices of MasterCard and Visa, which can help in making data-driven investment decisions.

## What I Learned

- How to handle and preprocess financial time-series data.
- Techniques for exploratory data analysis (EDA) in the context of stock prices.
- Building and evaluating predictive models using historical stock price data.
- Visualization of stock price trends to uncover patterns and insights.

git clone https://github.com/Sandrakimiring/stock-prices-analysis.git
cd stock-prices-analysis
