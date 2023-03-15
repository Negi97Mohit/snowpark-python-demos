# import streamlit as st
# import snowflake.snowpark
# from snowflake.snowpark import functions as F
# from snowflake.snowpark.session import Session
# import json

# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.compose import ColumnTransformer
# from xgboost import XGBRegressor
# import joblib
# import os
# from sqlalchemy import create_engine
# import sys
# import pandas as pd
# import cachetools
# import joblib
# from snowflake.snowpark import types as T
# from snowflake.sqlalchemy import URL
# def main():

#     url = URL(
#     user='NEGISMOHIT97',
#     password='Hollyhalston97)',
#     account='eouswjo-py32288',
#     warehouse='ANALYTICS_WH',
#     database='TPCDS_XGBOOST',
#     schema='DEMO',
#     role = 'ACCOUNTADMIN'
# )
#     engine = create_engine(url)
    
    
#     connection = engine.connect()
#     query = '''
#     select * from PREDICTIONS
#     '''
#     res = pd.read_sql(query, connection)
#     st.write(res.head())
#     return res

# global res
# if __name__ == "__main__":
#     res=main()


# st.title("Feature Engineering and XGBoost Training")
# st.write("This is a sample Streamlit app that demonstrates feature engineering and XGBoost training.")
    
# # Display advertising budget sliders and set their default values
# st.header("Advertising budgets")
# col1, _, col2 = st.columns([4, 1, 4])
# channels = ["C_BIRTH_YEAR",  "CD_CREDIT_RATING", "CD_EDUCATION_STATUS"]
# budgets = []
# for channel, default, col in zip(channels, res["ACTUAL_SALES"].values, [col1, col1, col2,]):
#     with col:
#         budget = st.slider(channel, 0, 100, int(default), 5)
#         budgets.append(budget)

import pandas as pd
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
import streamlit as st
import sqlalchemy
from streamlit_lottie import st_lottie
import json
import inspect

from py_scripts import part1 as p1

st.set_page_config(layout="wide")       
st.title("MId-Term Assignment")

def main():
    
    with st.container():
        st.header("Part - 1")
        st_part1()
    with st.container():
        st.header("Part - 2")
        st_part2()
        st.write("Hello World")

def st_part1():
    col1, col2 = st.columns(2,gap='small')
    with col1:    
        path = "97474-data-center.json"
        with open(path,"r") as file:
            url = json.load(file)
        st_lottie(url,
            reverse=True,
            height=400,
            width=400,
            speed=1,
            loop=True,
            quality='high',
            key='Car'
        )

        with st.expander("View Main Code"):
            p1code=inspect.getsource(p1)
            st.code(p1code,language='python')
        

    with col2:
        tables=['query1','query2','query3','query4','query5']
        tab1,tab2,tab3,tab4,tab5=st.tabs(tables)
        tab_names=[tab1,tab2,tab3,tab4,tab5]
        part1_engine = create_engine(URL(
            account = 'ieimkyf-atb67363',
            user = 'MOHITSNEGI123',
            password = 'Hollyhalston97)',
            database = 'midterm',
            schema = 'public',
            warehouse = 'COMPUTE_WH',
            role='ACCOUNTADMIN',
        ))
        
        if sqlalchemy.inspect(part1_engine).has_table("query1"):
            for table_name,tab in zip(tables,tab_names):
                with tab:
                    query="select * from "+str(table_name)
                    print(query)
                    print(table_name,"Part-1")
                    connection = part1_engine.connect()
                    res = pd.read_sql(query, connection)
                    title="Query number: "+str(table_name)
                    st.header(title)
                    st.write(res)
                part1_engine.dispose()

        else:
            p1.part1()

def st_part2():
    pass

if __name__=="__main__":
    main()

