
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib


# Set page configuration
st.set_page_config(layout='wide', page_title='Rental Property Market Analysis')

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset information", "Data analysis", "Prediction"])

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('rent_data_cleaned.csv')

df = load_data()

# --- Page 1: Dataset Information ---
if page == "Dataset information":
    st.title("🏠 Rental Market Data Understanding")
    
    # Rental-related image 
    st.image('https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1353&q=80', 
             caption='Housing and Rental Property Analytics', use_container_width=True)
    
    st.write("### Dataset Preview")
    st.dataframe(df)
    
    st.write("### Column Descriptions")
    
    cols = {
        "bathrooms": "The number of bathrooms in the rental unit.",
        "bedrooms": "The number of bedrooms in the rental unit.",
        "price": "The monthly rental cost in USD (Target Variable).",
        "square_feet": "The total living area size of the property in square feet.",
        "cityname": "The name of the city where the property is located.",
        "state": "The state abbreviation where the property is located.",
        "year": "The year the rental was listed (2018-2019).",
        "month": "The month of the year the rental was listed.",
        "day": "The day of the month the rental was listed.",
        "dist_miles": "The distance metric related to the property's specific location."
    }
    
    for col, desc in cols.items():
        with st.expander(f"**{col}**"):
            st.write(desc)

# --- Page 2: Data Analysis ---
elif page == "Data analysis":
    st.title("📊 Exploratory Data Analysis")
    
    analysis_type = st.tabs(["Univariate", "Bivariate", "Multivariate"])
    
    with analysis_type[0]:
        st.subheader("Univariate Analysis")
        
        for col in df.columns :
            fig = px.histogram(df , x = col )
            st.plotly_chart(fig, use_container_width=True)
        

    with analysis_type[1]:
        st.subheader("Bivariate Analysis")

        df_trend = df.groupby('month')['price'].mean().reset_index()
        fig_1 = px.line(df_trend, x='month', y='price', title="Rental Price Trends Over Time")
        fig_1.update_xaxes(type='category')
        st.plotly_chart(fig_1,use_container_width=True)


        df_grouped = df.groupby('bedrooms')['price'].mean().sort_values(ascending = False).reset_index()
        fig_2 = px.bar(df_grouped, x="bedrooms", y="price", title="Mean Rental Price per Bedroom",text_auto='.2s') 
        fig_2.update_xaxes(type='category')
        st.plotly_chart(fig_2,use_container_width=True)


        state_expensive = df.groupby('state')['price'].mean().sort_values(ascending=False).head(10).reset_index()
        fig_top_state = px.bar(state_expensive, x='state', y='price',title="Top 10 Most Expensive States (Average Rent)",text_auto='.2s',color='price',color_continuous_scale='Reds')
        fig_top_state.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig_top_state,use_container_width=True)



    with analysis_type[2]:
        st.subheader("Multivariate Analysis")

        df_summary = df.groupby(['bedrooms', 'year'])['price'].mean().reset_index()
        df_summary['year'] = df_summary['year'].astype(str)
        fig_3 = px.bar(df_summary, x="bedrooms", y="price", color="year", barmode="group",title=" Average Price by Bedroom Count",text_auto='.3s',labels={"price": "Avg Rent ($)", "bedrooms": "Bedrooms"})
        fig_3.update_layout(xaxis={'categoryorder':'total descending'})
        fig_3.update_xaxes(type='category')
        st.plotly_chart(fig_3,use_container_width=True)


        
        top_10 = df['state'].value_counts().nlargest(10).index
        df_simple = df[df['state'].isin(top_10)]
        df_grouped = df_simple.groupby(['state', 'year'])['price'].mean().reset_index()
        df_grouped['year'] = df_grouped['year'].astype(str)
        fig_4 = px.bar(df_grouped, x="state", y="price",color="year", barmode="group",title="2018 vs 2019 Rent by State",text_auto='.3s',labels={"price": "Avg Rent ($)", "state": "State", "year": "Year"})
        fig_4.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig_4,use_container_width=True)
    

# --- Page 3: Prediction ---
elif page == "Prediction":
    st.title("🔮 Rental Price Prediction")
    
    st.sidebar.markdown("### Property Specifications")
    
    # Sidebar inputs
    bathrooms = st.sidebar.slider('Number of Bathrooms', 1.0, 10.0, 1.0, 0.5)
    bedrooms = st.sidebar.slider('Number of Bedrooms', 0.0, 10.0, 1.0, 1.0)
    square_feet = st.sidebar.number_input('Total Square Feet', 100, 15000, 1000)
    
    # State and Dynamic City selection
    state_list = sorted(df['state'].unique().tolist())
    state = st.sidebar.selectbox('Select State', state_list)
    cities_in_state = sorted(df[df['state'] == state]['cityname'].unique().tolist())
    cityname = st.sidebar.selectbox('Select City', cities_in_state)
    
    dist_miles = st.sidebar.number_input('Distance from Reference (miles)', value=5.0)
    
    # Date Features
    st.subheader("Listing Timeframe")
    c1, c2, c3 = st.columns(3)
    year = c1.selectbox('Year', sorted(df['year'].unique()))
    month = c2.selectbox('Month', list(range(1, 13)))
    day = c3.selectbox('Day', list(range(1, 32)))

    # Input Data Summary
    input_df = pd.DataFrame({
        'bathrooms': [bathrooms], 'bedrooms': [bedrooms], 'square_feet': [square_feet],
        'cityname': [cityname], 'state': [state], 'year': [year], 'month': [month],
        'day': [day], 'dist_miles': [dist_miles]
    })

    st.markdown("### 📋 Input Summary")
    st.table(input_df)
    
    if st.button("Calculate Estimated Rent"):
        try:
            model = joblib.load('final_rent_prediction_model.pkl')
            prediction = model.predict(input_df)
            st.success(f"💰 Estimated Monthly Rent: ${prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
