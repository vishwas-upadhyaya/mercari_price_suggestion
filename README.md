# Mercari Price Suggestion Challenge

## Project Overview
This repository contains a machine learning pipeline to predict the price of products on Mercari, Japan’s biggest community-powered shopping app. The objective is to build an algorithm that automatically suggests the right product prices given text descriptions, product names, category names, brand names, item conditions, and shipping information. 

## Deep Technical Details (Architecture, Pipeline)

### Architecture
The best performing model in this repository is a Multi-Input Deep Learning Architecture built using the **TensorFlow/Keras Functional API**:
- **Text Inputs (`name`, `item_description`):** Tokenized, padded to fixed lengths, and fed into `Embedding` layers to capture semantic relationships.
- **Categorical Inputs (`brand_name`, `subcategories 1-5`):** Integer encoded and mapped via `Embedding` layers. Categories are derived by splitting the hierarchical `category_name` feature.
- **Numerical/Meta Inputs (`shipping`, `item_condition_id`, text lengths):** Scaled using `MinMaxScaler` and passed directly as dense input features.
- **Hidden Layers:** All embeddings and numerical features are flattened (or pooled) and merged using a `Concatenate` layer. The combined feature vector is then passed through fully connected `Dense` layers with `Dropout` regularization to predict the continuous `price` target.
- *Note:* The repository also contains experiments with baseline ML models such as `SGDRegressor`, `Lasso`, `Ridge`, `SVR`, `RandomForestRegressor`, `XGBRegressor`, and `LGBMRegressor`.

### Data Pipeline
The end-to-end data processing pipeline includes:
1. **Text Preprocessing:** 
   - Decontraction (e.g., expanding "won't" to "will not").
   - Removing special characters, punctuation, and emojis using Regex.
   - Stop-word removal and Lemmatization using `NLTK` (`WordNetLemmatizer`).
2. **Feature Engineering:**
   - Splitting `category_name` into five granular subcategories (`subcat1` to `subcat5`).
   - Extracting meta-features like word counts of `name` and `item_description`.
3. **Encoding & Scaling:**
   - Text features are tokenized and padded (`pad_sequences`).
   - Numerical features are normalized using Scikit-Learn's `MinMaxScaler`.
4. **Prediction:**
   - Preprocessed arrays are passed to a pre-trained Keras model (`best_model.h5`) to generate the final price prediction.

## Tech Stack
- **Python** (Data manipulation and modeling)
- **TensorFlow / Keras** (Deep Learning framework for embeddings and the final regression model)
- **Scikit-Learn** (Preprocessing, scaling, traditional ML models, grid search)
- **NLTK & WordCloud** (Natural Language Processing and text visualization)
- **XGBoost & LightGBM** (Gradient boosting regressors)
- **Pandas, NumPy, SciPy** (Data manipulation and sparse matrix operations)
- **Matplotlib & Seaborn** (Exploratory Data Analysis)