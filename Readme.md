# YouTube Monetization Modeler

This is a complete end-to-end Machine Learning project to predict YouTube `ad_revenue_usd` using video performance metrics like views, likes, comments, and watch time. The project uses advanced regression models to uncover how these factors drive revenue and provides actionable insights for content creators.

## Overview

The repository contains:
1. **`model_training.ipynb`**: A comprehensive Jupyter Notebook covering data loading, cleaning, exploratory data analysis, training multiple regression models, model evaluation, and saving the best-performing model.
2. **`app.py`**: A beautiful and interactive Streamlit web application that utilizes the trained model to provide real-time revenue predictions based on user inputs.
3. **`youtube_ad_revenue.csv`**: The dataset used for modeling (approx. 120,000 rows).

## Model Evaluation Results

Five models were trained and compared:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor (Best Model)

**Best Model Metrics (Gradient Boosting):**
- **R² Score**: ~0.95
- **RMSE**: ~13.71
- **MAE**: ~4.56

## Installation

1. Make sure you have Python 3.8+ installed.
2. Install the required dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn joblib
```

## How to Run

### Step 1: Run the Jupyter Notebook (Optional)
If you want to re-train the models and explore the data, run the Jupyter Notebook:
```bash
jupyter notebook model_training.ipynb
```
Follow the cells sequentially. The final cell exports the best model to `model.pkl`. Note: The `model.pkl` is already provided for you if you've run the provided `train_script.py`.

### Step 2: Run the Streamlit App
To launch the interactive web application, open your terminal/command prompt, navigate to the project directory, and run:
```bash
streamlit run app.py
```

This will automatically open the app in your default web browser (typically at `http://localhost:8501`). Enter your video metrics in the sidebar to get instant ad revenue predictions!

## Key Findings & Actionable Insights
- **Watch Time is King**: The model indicates that `watch_time_minutes` is typically the most significant predictor of Ad Revenue, often far outweighing raw view counts.
- **Engagement Matters**: Features like `likes` and `comments` strongly correlate with higher earnings, as YouTube's algorithm favors highly engaging content.
- **Actionable Takeaway**: Creators should focus on producing longer, captivating content that maximizes viewer retention rather than purely optimizing for clicks (views). Engaging with the audience to drive likes and comments also provides a notable boost.
