# mercari_price_suggestion

## Project Overview
A machine learning project that predicts the pricing of items sold on the Mercari e-commerce platform.

## What is this Project?
This project tackles the Mercari Price Suggestion Challenge. It uses data science and machine learning techniques to automatically suggest appropriate prices for products based on features like category, brand name, condition, and text descriptions.

## How it was done
The project workflow is documented in Jupyter Notebooks. It starts with comprehensive Exploratory Data Analysis (`EDA.ipynb`) to understand the dataset. Then, `feature_engineering_colab.ipynb` handles the preprocessing of text and categorical data. Finally, `pipeline.ipynb` integrates these steps into a unified model pipeline (typically involving algorithms like Ridge Ridge Regression or LightGBM) to forecast item prices.

## Why it was done
To apply advanced natural language processing and regression techniques to a real-world, large-scale e-commerce pricing problem.

## Tech Stack
- Python
- Pandas & NumPy
- Scikit-learn / LightGBM (Machine Learning)
- Jupyter Notebook / Google Colab

## Key Features
- In-depth Exploratory Data Analysis of an e-commerce dataset.
- Extensive feature engineering on text descriptions and categorical metadata.
- End-to-end machine learning pipeline for price prediction.

## File Structure
- `EDA.ipynb`: Notebook containing exploratory data analysis and visualizations.
- `feature_engineering_colab.ipynb`: Notebook detailing the data transformation processes.
- `pipeline.ipynb`: The final machine learning pipeline and model evaluation.

## Local Setup (if applicable)
1. Clone the repository.
2. Install necessary libraries: `pip install pandas numpy scikit-learn matplotlib seaborn jupyter lightgbm`.
3. Open the notebooks in Jupyter to run the analysis and models. Note that the original datasets from Kaggle may need to be downloaded to run the code.