import streamlit as st
import snowflake.snowpark
from snowflake.snowpark import functions as F
from snowflake.snowpark.session import Session
import json

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import joblib
import os
from sqlalchemy import create_engine
import sys
import pandas as pd
import cachetools
import joblib
from snowflake.snowpark import types as T
from snowflake.sqlalchemy import URL
def main():

    url = URL(
    user='NEGISMOHIT97',
    password='Hollyhalston97)',
    account='eouswjo-py32288',
    warehouse='ANALYTICS_WH',
    database='TPCDS_XGBOOST',
    schema='DEMO',
    role = 'ACCOUNTADMIN'
)
    engine = create_engine(url)
    
    
    connection = engine.connect()
    query = '''
    select * from PREDICTIONS
    '''
    res = pd.read_sql(query, connection)
    st.write(res.head())
    return res

global res
if __name__ == "__main__":
    res=main()


st.title("Feature Engineering and XGBoost Training")
st.write("This is a sample Streamlit app that demonstrates feature engineering and XGBoost training.")
    
# Display advertising budget sliders and set their default values
st.header("Advertising budgets")
col1, _, col2 = st.columns([4, 1, 4])
channels = ["C_BIRTH_YEAR",  "CD_CREDIT_RATING", "CD_EDUCATION_STATUS"]
budgets = []
for channel, default, col in zip(channels, res["ACTUAL_SALES"].values, [col1, col1, col2,]):
    with col:
        budget = st.slider(channel, 0, 100, int(default), 5)
        budgets.append(budget)

