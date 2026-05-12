import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="YouTube Revenue Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
# Adds a professional modern look with highlighted metric cards that adapt to dark/light themes.
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background-color: #262730;
            color: white;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
# Expected categorical values
CATEGORIES = ['Entertainment', 'Gaming', 'Lifestyle', 'Music', 'Tech']
COUNTRIES = ['CA', 'DE', 'IN', 'UK', 'US']
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_models():
    """
    Load the pre-trained machine learning model and scaler from pickle files.
    Uses @st.cache_resource to load only once per session and save memory.
    """
    model = None
    scaler = None
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model.pkl. Please ensure it exists and matches the current scikit-learn version. Details: {e}")
        
    if os.path.exists(SCALER_PATH):
        try:
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
        except Exception as e:
            st.warning(f"Error loading scaler.pkl: {e}")
            
    return model, scaler

def preprocess_input(data_df, scaler):
    """
    Preprocess the input dataframe to match the exact format the model expects.
    This includes feature engineering, encoding categorical variables, and scaling.
    """
    df = data_df.copy()
    
    # 1. Feature Engineering: Calculate Engagement Rate
    df['engagement_rate'] = (df['likes'] + df['comments']) / df['views']
    # Handle division by zero or infinites if views are 0
    df['engagement_rate'] = df['engagement_rate'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # 2. Categorical Encoding
    # Initialize all expected categorical dummy columns to 0
    for cat in CATEGORIES:
        df[f'category_{cat}'] = 0
    for cntry in COUNTRIES:
        df[f'country_{cntry}'] = 0
        
    # Set the user-selected category and country to 1
    if 'category' in df.columns:
        for idx, row in df.iterrows():
            cat_col = f"category_{row['category']}"
            if cat_col in df.columns:
                df.at[idx, cat_col] = 1
                
    if 'country' in df.columns:
        for idx, row in df.iterrows():
            cntry_col = f"country_{row['country']}"
            if cntry_col in df.columns:
                df.at[idx, cntry_col] = 1
                
    # 3. Feature Ordering
    expected_cols = [
        'views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 
        'subscribers', 
        'category_Entertainment', 'category_Gaming', 'category_Lifestyle', 
        'category_Music', 'category_Tech', 
        'country_CA', 'country_DE', 'country_IN', 'country_UK', 'country_US'
    ]
    
    # Ensure all expected columns exist in the DataFrame
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
            
    # Subset to keep only expected columns in exact order
    df_processed = df[expected_cols]
    
    # 4. Scaling
    # Apply standard scaling if a scaler was loaded
    if scaler is not None:
        try:
            # Note: We assume the scaler was fit on all these features in this order.
            df_processed[expected_cols] = scaler.transform(df_processed[expected_cols])
        except Exception as e:
            st.warning(f"Failed to apply scaling: {e}")
            
    return df_processed

def get_performance_category(engagement_rate):
    """
    Determine the performance category based on the calculated engagement rate.
    """
    if engagement_rate >= 0.05:
        return "Excellent 🌟"
    elif engagement_rate >= 0.02:
        return "Good 👍"
    else:
        return "Needs Improvement 📉"

# --- MAIN APP ---
def main():
    # Load models
    model, scaler = load_models()
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Select Application Mode", ["Single Prediction", "Batch Prediction", "About System"])
    
    # --- ABOUT SECTION ---
    if app_mode == "About System":
        st.title("ℹ️ About the System")
        st.markdown("""
        ### AI-Powered YouTube Revenue Prediction
        This dashboard uses a trained Machine Learning model to forecast potential ad revenue for YouTube videos based on early engagement metrics and channel data.
        
        **Workflow:**
        1. **Data Input:** Provide video statistics (views, likes, comments, etc.).
        2. **Feature Engineering:** The app calculates engagement rate automatically.
        3. **Preprocessing:** Categorical data is one-hot encoded behind the scenes.
        4. **Prediction:** The `model.pkl` processes the inputs and estimates revenue.
        
        Built with Streamlit, Scikit-Learn, Pandas, and Plotly.
        """)
        return

    # --- MAIN TITLE ---
    st.title("📈 YouTube Revenue Prediction System")
    st.markdown("Analyze engagement metrics and predict estimated ad revenue instantly.")
    
    # Stop execution if model couldn't be loaded
    if model is None:
        st.error("Cannot proceed: The prediction model is missing or incompatible.")
        st.stop()
        
    # --- SINGLE PREDICTION SECTION ---
    if app_mode == "Single Prediction":
        st.header("1. Enter Video Metrics")
        
        # Create a visually pleasing two-column layout for input forms
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Engagement Data")
            views = st.number_input("Total Views", min_value=1, value=15000, step=1000)
            likes = st.number_input("Total Likes", min_value=0, value=1200, step=100)
            comments = st.number_input("Total Comments", min_value=0, value=150, step=10)
            
            # Validation: Warn if engagement is unrealistically high compared to views
            if likes + comments > views:
                st.warning("⚠️ Warning: Likes and Comments combined exceed Total Views. Please double-check your inputs.")
                
        with col2:
            st.subheader("Video & Channel Data")
            watch_time = st.number_input("Watch Time (Minutes)", min_value=0.0, value=75000.0, step=1000.0)
            video_length = st.number_input("Video Length (Minutes)", min_value=0.0, value=12.5, step=0.5)
            subscribers = st.number_input("Channel Subscribers", min_value=0, value=100000, step=5000)
            
        st.subheader("Categorical Details")
        col3, col4 = st.columns(2)
        with col3:
            category = st.selectbox("Video Content Category", CATEGORIES)
        with col4:
            country = st.selectbox("Primary Audience Location", COUNTRIES)
            
        # Prediction Trigger
        st.markdown("---")
        if st.button("Generate Revenue Prediction 🚀", use_container_width=True, type="primary"):
            with st.spinner("Processing data and running model..."):
                # 1. Collect inputs into a DataFrame
                input_data = {
                    'views': views,
                    'likes': likes,
                    'comments': comments,
                    'watch_time_minutes': watch_time,
                    'video_length_minutes': video_length,
                    'subscribers': subscribers,
                    'category': category,
                    'country': country
                }
                input_df = pd.DataFrame([input_data])
                
                # 2. Preprocess data
                processed_df = preprocess_input(input_df, scaler)
                
                # 3. Make Prediction
                try:
                    prediction_array = model.predict(processed_df)
                    predicted_revenue = max(0, prediction_array[0]) # Ensure no negative revenue
                    
                    # 4. Calculate Additional Metrics
                    engagement_rate = (likes + comments) / views
                    perf_category = get_performance_category(engagement_rate)
                    
                    # --- DISPLAY PREDICTION RESULTS ---
                    st.header("Prediction Results")
                    
                    # Highlighted Metric Cards Layout
                    rc1, rc2, rc3 = st.columns(3)
                    with rc1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style='margin-bottom: 5px; color: gray;'>Estimated Revenue</h3>
                            <h2 style='color: #28a745; margin-top: 0;'>${predicted_revenue:,.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    with rc2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style='margin-bottom: 5px; color: gray;'>Engagement Rate</h3>
                            <h2 style='color: #007bff; margin-top: 0;'>{engagement_rate * 100:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    with rc3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style='margin-bottom: 5px; color: gray;'>Performance</h3>
                            <h2 style='margin-top: 0;'>{perf_category}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # --- DATA VISUALIZATION ---
                    st.markdown("### Metrics Visualization Overview")
                    
                    # Create a bar chart comparing core inputs using Plotly
                    chart_data = pd.DataFrame({
                        'Metric': ['Views', 'Subscribers', 'Likes', 'Comments'],
                        'Value': [views, subscribers, likes, comments]
                    })
                    
                    # Use log scale for better visualization since views/subs are much higher than likes/comments
                    fig = px.bar(
                        chart_data, 
                        x='Metric', 
                        y='Value', 
                        text='Value', 
                        title="Input Metrics Breakdown (Log Scale for Visualization)",
                        color='Metric', 
                        template='plotly_white',
                        log_y=True
                    )
                    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"An error occurred while running the prediction model: {e}")
                    
    # --- BATCH PREDICTION SECTION ---
    elif app_mode == "Batch Prediction":
        st.header("Batch Revenue Prediction")
        st.markdown("Upload a CSV dataset containing multiple video records to predict their revenues simultaneously.")
        
        # Create and provide a sample dataset for download
        sample_data = pd.DataFrame({
            'views': [15000, 55000, 5000],
            'likes': [1200, 4500, 150],
            'comments': [150, 400, 20],
            'watch_time_minutes': [75000.0, 320000.0, 15000.0],
            'video_length_minutes': [12.5, 18.0, 5.0],
            'subscribers': [100000, 250000, 10000],
            'category': ['Entertainment', 'Gaming', 'Tech'],
            'country': ['US', 'CA', 'IN']
        })
        
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=sample_data.to_csv(index=False).encode('utf-8'),
            file_name="sample_batch_data.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # File Uploader
        uploaded_file = st.file_uploader("Upload your dataset (.csv format)", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read dataset
                batch_df = pd.read_csv(uploaded_file)
                st.write("### Data Preview")
                st.dataframe(batch_df.head())
                
                if st.button("Run Batch Prediction ⚙️"):
                    with st.spinner("Processing batch dataset..."):
                        # Preprocess
                        processed_batch = preprocess_input(batch_df, scaler)
                        
                        # Predict
                        predictions = model.predict(processed_batch)
                        
                        # Compile Results into a new DataFrame
                        results_df = batch_df.copy()
                        results_df['Predicted_Revenue ($)'] = np.round(np.maximum(0, predictions), 2)
                        
                        # Calculate Engagement Rate and Performance Category
                        results_df['Engagement_Rate'] = (results_df['likes'] + results_df['comments']) / results_df['views']
                        results_df['Engagement_Rate'].replace([np.inf, -np.inf], 0, inplace=True)
                        results_df['Engagement_Rate'] = results_df['Engagement_Rate'].fillna(0)
                        
                        results_df['Performance_Category'] = results_df['Engagement_Rate'].apply(get_performance_category)
                        
                        st.success(f"Batch Prediction Completed for {len(results_df)} records!")
                        
                        # Display results
                        st.dataframe(results_df[['views', 'category', 'Predicted_Revenue ($)', 'Engagement_Rate', 'Performance_Category']].head(10))
                        
                        # Generate Downloadable Output
                        csv_results = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="💾 Download Full Prediction Results (CSV)",
                            data=csv_results,
                            file_name="batch_predictions_results.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Error processing the uploaded CSV file: {e}. Please ensure it matches the required template.")
                
    # --- FOOTER ---
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; padding: 10px;'>
            YouTube Revenue Prediction System | Powered by <b>Streamlit</b> & <b>Scikit-Learn</b>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
