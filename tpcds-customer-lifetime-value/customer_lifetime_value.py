import pandas as pd
import numpy as np
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
import streamlit as st
import sqlalchemy
from streamlit_lottie import st_lottie
import json
import inspect
import requests
from sklearn.metrics import mean_squared_error
import math

import plotly.express as px

st.set_page_config(layout="wide")       
st.title("MId-Term Assignment")

def accuracy_calc(df):
    return df['predicted_value']/(df['predicted_value']+df['actual_sales'])
def run_query(dob_list,education_option,gender_option,dept_option,credit_option,marital_option):        

    query_prime1="""select sum(prediction) Predicted,sum(actual_sales) Actual_Sales
                from predictions
                where c_birth_year between """ + str(dob_list[0]) +""" and """ +str( dob_list[1])

    query_prime12="""select c_birth_year,sum(prediction) Predicted_value,sum(actual_sales) Actual_Sales
                from predictions
                where c_birth_year between """ + str(dob_list[0]) +""" and """ +str( dob_list[1])


    str_pos=[1,1,1,1,1,1,1,1,1,1]
    if len(gender_option)==1:
        str_pos[5]=-1
        str_pos[0]=0
    elif len(gender_option)>=1:
        str_pos[5]=0
        str_pos[0]=1
    else:
        gender_option=gender_option+(1,)
        str_pos[5]=0
        str_pos[0]=0
        
    if len(marital_option)==1:
        str_pos[6]=-1
        str_pos[1]=0
    elif len(marital_option)>=1:
        str_pos[6]=0
        str_pos[1]=1
    else:
        marital_option=marital_option+(1,)
        str_pos[6]=0
        str_pos[1]=0

    if len(dept_option)==1:
        str_pos[7]=-1
        str_pos[2]=0
    elif len(dept_option)>=1:
        str_pos[7]=0
        str_pos[2]=1
    else:
        dept_option=dept_option+(1,)
        str_pos[7]=0
        str_pos[2]=0

    if len(credit_option)==1:
        str_pos[8]=-1
        str_pos[3]=0
    elif len(credit_option)>=1:
        str_pos[8]=0
        str_pos[3]=1
    else:
        credit_option=credit_option+(1,)
        str_pos[8]=0
        str_pos[3]=0

    if len(education_option)==1:
        str_pos[9]=-1
        str_pos[4]=0
    elif len(education_option)>=1:
        str_pos[9]=0
        str_pos[4]=1
    else:
        education_option=education_option+(1,)
        str_pos[9]=0
        str_pos[4]=0
    query_vals=[
        '''
        and cd_gender in'''+ str(gender_option),
        '''
        and cd_marital_status in'''+str(marital_option), 
        '''
        and cd_dep_count in '''+str(dept_option),
        '''
        and cd_credit_rating in '''+ str(credit_option),
        '''
        and cd_education_status in '''+ str(education_option),
        """ 
        and cd_gender in('"""+ str(gender_option[0])+"""')""",
        """
        and cd_marital_status in ('"""+str(marital_option[0])+"""')""",
        """ 
        and cd_dep_count in ('"""+str(dept_option[0])+"""')""",
        """
        and cd_credit_rating in ('"""+ str(credit_option[0])+"""')""",
        """
        and cd_education_status in  ('"""+ str(education_option[0])+"""')"""]

    for post_stat,query_val in zip(str_pos,query_vals):
        if post_stat!=0:
            query_prime1=query_prime1+query_val
            query_prime12=query_prime12+query_val

    query_prime12_last="""
            group by 1
            order by c_birth_year"""
    query_prime12=query_prime12+query_prime12_last
    return query_prime1,query_prime12

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    with st.container():
        st.header("Part - 1")
        st_part1()
    with st.container():
        st.header("Part - 2: Customer Lifetime Valuation using XGBoost")
        st_part2()

def st_part1():
    col1, col2 = st.columns(2,gap='small')
    with col1:   
        lottie_url_hello = "https://raw.githubusercontent.com/Negi97Mohit/snowpark-python-demos/main/tpcds-customer-lifetime-value/97474-data-center.json"
        lottie_hello = load_lottieurl(lottie_url_hello)
        st_lottie(lottie_hello,
            reverse=True,
            height=400,
            width=400,
            speed=1,
            loop=True,
            quality='high',
            key='Car'
        )
        with st.expander("Query table generator code"):
            p1code="""import pandas as pd
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine

queries=[
'''
with year_total as (
select c_customer_id customer_id
      ,c_first_name customer_first_name
      ,c_last_name customer_last_name
      ,c_preferred_cust_flag customer_preferred_cust_flag
      ,c_birth_country customer_birth_country
      ,c_login customer_login
      ,c_email_address customer_email_address
      ,d_year dyear
      ,sum(ss_ext_list_price-ss_ext_discount_amt) year_total
      ,'s' sale_type
from customer
    ,store_sales
    ,date_dim
where c_customer_sk = ss_customer_sk
  and ss_sold_date_sk = d_date_sk
group by c_customer_id
        ,c_first_name
        ,c_last_name
        ,c_preferred_cust_flag 
        ,c_birth_country
        ,c_login
        ,c_email_address
        ,d_year 
union all
select c_customer_id customer_id
      ,c_first_name customer_first_name
      ,c_last_name customer_last_name
      ,c_preferred_cust_flag customer_preferred_cust_flag
      ,c_birth_country customer_birth_country
      ,c_login customer_login
      ,c_email_address customer_email_address
      ,d_year dyear
      ,sum(ws_ext_list_price-ws_ext_discount_amt) year_total
      ,'w' sale_type
from customer
    ,web_sales
    ,date_dim
where c_customer_sk = ws_bill_customer_sk
  and ws_sold_date_sk = d_date_sk
group by c_customer_id
        ,c_first_name
        ,c_last_name
        ,c_preferred_cust_flag 
        ,c_birth_country
        ,c_login
        ,c_email_address
        ,d_year
        )
  select  
                  t_s_secyear.customer_id
                ,t_s_secyear.customer_first_name
                ,t_s_secyear.customer_last_name
                ,t_s_secyear.customer_preferred_cust_flag
from year_total t_s_firstyear
    ,year_total t_s_secyear
    ,year_total t_w_firstyear
    ,year_total t_w_secyear
where t_s_secyear.customer_id = t_s_firstyear.customer_id
        and t_s_firstyear.customer_id = t_w_secyear.customer_id
        and t_s_firstyear.customer_id = t_w_firstyear.customer_id
        and t_s_firstyear.sale_type = 's'
        and t_w_firstyear.sale_type = 'w'
        and t_s_secyear.sale_type = 's'
        and t_w_secyear.sale_type = 'w'
        and t_s_firstyear.dyear = 2001
        and t_s_secyear.dyear = 2001+1
        and t_w_firstyear.dyear = 2001
        and t_w_secyear.dyear = 2001+1
        and t_s_firstyear.year_total > 0
        and t_w_firstyear.year_total > 0
        and case when t_w_firstyear.year_total > 0 then t_w_secyear.year_total / t_w_firstyear.year_total else 0.0 end
            > case when t_s_firstyear.year_total > 0 then t_s_secyear.year_total / t_s_firstyear.year_total else 0.0 end
order by t_s_secyear.customer_id
        ,t_s_secyear.customer_first_name
        ,t_s_secyear.customer_last_name
        ,t_s_secyear.customer_preferred_cust_flag
limit 100''',

'''select  i_item_id
      ,i_item_desc 
      ,i_category 
      ,i_class 
      ,i_current_price
      ,sum(ws_ext_sales_price) as itemrevenue 
      ,sum(ws_ext_sales_price)*100/sum(sum(ws_ext_sales_price)) over
          (partition by i_class) as revenueratio
from	
  web_sales
      ,item 
      ,date_dim
where 
  ws_item_sk = i_item_sk 
    and i_category in ('Sports', 'Books', 'Home')
    and ws_sold_date_sk = d_date_sk
  and d_date between cast('1999-02-22' as date) 
                                and dateadd(day,30,cast('1999-02-22' as date) )
group by 
  i_item_id
        ,i_item_desc 
        ,i_category
        ,i_class
        ,i_current_price
order by 
  i_category
        ,i_class
        ,i_item_id
        ,i_item_desc
        ,revenueratio
limit 100''',

''' select avg(ss_quantity)
      ,avg(ss_ext_sales_price)
      ,avg(ss_ext_wholesale_cost)
      ,sum(ss_ext_wholesale_cost)
from store_sales
    ,store
    ,customer_demographics
    ,household_demographics
    ,customer_address
    ,date_dim
where s_store_sk = ss_store_sk
and  ss_sold_date_sk = d_date_sk and d_year = 2001
and((ss_hdemo_sk=hd_demo_sk
  and cd_demo_sk = ss_cdemo_sk
  and cd_marital_status = 'M'
  and cd_education_status = 'Advanced Degree'
  and ss_sales_price between 100.00 and 150.00
  and hd_dep_count = 3   
    )or
    (ss_hdemo_sk=hd_demo_sk
  and cd_demo_sk = ss_cdemo_sk
  and cd_marital_status = 'S'
  and cd_education_status = 'College'
  and ss_sales_price between 50.00 and 100.00   
  and hd_dep_count = 1
    ) or 
    (ss_hdemo_sk=hd_demo_sk
  and cd_demo_sk = ss_cdemo_sk
  and cd_marital_status = 'W'
  and cd_education_status = '2 yr Degree'
  and ss_sales_price between 150.00 and 200.00 
  and hd_dep_count = 1  
    ))
and((ss_addr_sk = ca_address_sk
  and ca_country = 'United States'
  and ca_state in ('TX', 'OH', 'TX')
  and ss_net_profit between 100 and 200  
    ) or
    (ss_addr_sk = ca_address_sk
  and ca_country = 'United States'
  and ca_state in ('OR', 'NM', 'KY')
  and ss_net_profit between 150 and 300  
    ) or
    (ss_addr_sk = ca_address_sk
  and ca_country = 'United States'
  and ca_state in ('VA', 'TX', 'MS')
  and ss_net_profit between 50 and 250  
    ))
''',

'''-- start query 14 in stream 0 using template query14.tpl and seed QUALIFICATION
with  cross_items as
(select i_item_sk ss_item_sk
from item,
(select iss.i_brand_id brand_id
    ,iss.i_class_id class_id
    ,iss.i_category_id category_id
from store_sales
    ,item iss
    ,date_dim d1
where ss_item_sk = iss.i_item_sk
  and ss_sold_date_sk = d1.d_date_sk
  and d1.d_year between 1999 AND 1999 + 2
intersect 
select ics.i_brand_id
    ,ics.i_class_id
    ,ics.i_category_id
from catalog_sales
    ,item ics
    ,date_dim d2
where cs_item_sk = ics.i_item_sk
  and cs_sold_date_sk = d2.d_date_sk
  and d2.d_year between 1999 AND 1999 + 2
intersect
select iws.i_brand_id
    ,iws.i_class_id
    ,iws.i_category_id
from web_sales
    ,item iws
    ,date_dim d3
where ws_item_sk = iws.i_item_sk
  and ws_sold_date_sk = d3.d_date_sk
  and d3.d_year between 1999 AND 1999 + 2)
where i_brand_id = brand_id
      and i_class_id = class_id
      and i_category_id = category_id
),
avg_sales as
(select avg(quantity*list_price) average_sales
  from (select ss_quantity quantity
            ,ss_list_price list_price
      from store_sales
          ,date_dim
      where ss_sold_date_sk = d_date_sk
        and d_year between 1999 and 1999 + 2
      union all 
      select cs_quantity quantity 
            ,cs_list_price list_price
      from catalog_sales
          ,date_dim
      where cs_sold_date_sk = d_date_sk
        and d_year between 1999 and 1999 + 2 
      union all
      select ws_quantity quantity
            ,ws_list_price list_price
      from web_sales
          ,date_dim
      where ws_sold_date_sk = d_date_sk
        and d_year between 1999 and 1999 + 2) x)
  select  channel, i_brand_id,i_class_id,i_category_id,sum(sales), sum(number_sales)
from(
      select 'store' channel, i_brand_id,i_class_id
            ,i_category_id,sum(ss_quantity*ss_list_price) sales
            , count(*) number_sales
      from store_sales
          ,item
          ,date_dim
      where ss_item_sk in (select ss_item_sk from cross_items)
        and ss_item_sk = i_item_sk
        and ss_sold_date_sk = d_date_sk
        and d_year = 1999+2 
        and d_moy = 11
      group by i_brand_id,i_class_id,i_category_id
      having sum(ss_quantity*ss_list_price) > (select average_sales from avg_sales)
      union all
      select 'catalog' channel, i_brand_id,i_class_id,i_category_id, sum(cs_quantity*cs_list_price) sales, count(*) number_sales
      from catalog_sales
          ,item
          ,date_dim
      where cs_item_sk in (select ss_item_sk from cross_items)
        and cs_item_sk = i_item_sk
        and cs_sold_date_sk = d_date_sk
        and d_year = 1999+2 
        and d_moy = 11
      group by i_brand_id,i_class_id,i_category_id
      having sum(cs_quantity*cs_list_price) > (select average_sales from avg_sales)
      union all
      select 'web' channel, i_brand_id,i_class_id,i_category_id, sum(ws_quantity*ws_list_price) sales , count(*) number_sales
      from web_sales
          ,item
          ,date_dim
      where ws_item_sk in (select ss_item_sk from cross_items)
        and ws_item_sk = i_item_sk
        and ws_sold_date_sk = d_date_sk
        and d_year = 1999+2
        and d_moy = 11
      group by i_brand_id,i_class_id,i_category_id
      having sum(ws_quantity*ws_list_price) > (select average_sales from avg_sales)
) y
group by rollup (channel, i_brand_id,i_class_id,i_category_id)
order by channel,i_brand_id,i_class_id,i_category_id
  limit 100;''', 
  
'''select  ca_zip
      ,sum(cs_sales_price)
from catalog_sales
    ,customer
    ,customer_address
    ,date_dim
where cs_bill_customer_sk = c_customer_sk
  and c_current_addr_sk = ca_address_sk 
  and ( substr(ca_zip,1,5) in ('85669', '86197','88274','83405','86475',
                                  '85392', '85460', '80348', '81792')
        or ca_state in ('CA','WA','GA')
        or cs_sales_price > 500)
  and cs_sold_date_sk = d_date_sk
  and d_qoy = 2 and d_year = 2001
group by ca_zip
order by ca_zip
  limit 100'''
]

tables=['query1','query2','query3','query4','query5']
engine_old= create_engine(URL(
    account = 'pinvdzu-ljb05593',
    user = 'gamer9797123',
    password = 'Gamer9797123)',
    database = 'SNOWFLAKE_SAMPLE_DATA',
    schema = 'TPCDS_SF10TCL',
    warehouse = 'COMPUTE_WH',
    role='ACCOUNTADMIN',
))

engine_new = create_engine(URL(
    account = 'pinvdzu-ljb05593',
    user = 'gamer9797123',
    password = 'Gamer9797123)',
    database = 'midterm',
    schema = 'public',
    warehouse = 'COMPUTE_WH',
    role='ACCOUNTADMIN',
))


for query,table_name in zip(queries,tables):
    connection = engine_old.connect()
    print(table_name,"old Connected")
    res = pd.read_sql(query, connection)
    engine_old.dispose()
    connection = engine_new.connect()
    res.to_sql(table_name,connection,index=False,if_exists='replace')
    print(table_name,"new  Connected")
    engine_new.dispose()
"""
            st.code(p1code,language='python')
        

    with col2:
        questions=['''Find customers whose increase in spending was large over the web than in stores this year compared to last 
                    year. 
                    Qualification Substitution Parameters:
                     YEAR.01 = 2001
                     SELECTONE = t_s_secyear.customer_preferred_cust_flag''',
                    '''Compute the revenue ratios across item classes: For each item in a list of given categories, during a 30 day time 
                    period, sold through the web channel compute the ratio of sales of that item to the sum of all of the sales in that 
                    item's class.
                    Qualification Substitution Parameters
                     CATEGORY.01 = Sports
                     CATEGORY.02 = Books
                     CATEGORY.03 = Home
                     SDATE.01 = 1999-02-22
                     YEAR.01 = 1999''',
                    '''
                    Calculate the average sales quantity, average sales price, average wholesale cost, total wholesale cost for store 
                    sales of different customer types (e.g., based on marital status, education status) including their household 
                    demographics, sales price and different combinations of state and sales profit for a given year.
                    Qualification Substitution Parameters:
                     STATE.01 = TX
                     STATE.02 = OH
                     STATE.03 = TX
                     STATE.04 = OR
                     STATE.05 = NM
                     STATE.06 = KY
                     STATE.07 = VA
                     STATE.08 = TX
                     STATE.09 = MS
                     ES.01 = Advanced Degree
                     ES.02 = College
                     ES.03 = 2 yr Degree
                     MS.01 = M
                     MS.02 = S
                     MS.03 = W
                    ''',
                    '''This query contains multiple iterations:
                    Iteration 1: First identify items in the same brand, class and category that are sold in all three sales channels in 
                    two consecutive years. Then compute the average sales (quantity*list price) across all sales of all three sales 
                    channels in the same three years (average sales). Finally, compute the total sales and the total number of sales 
                    rolled up for each channel, brand, class and category. Only consider sales of cross channel sales that had sales 
                    larger than the average sale.
                    Iteration 2: Based on the previous query compare December store sales.
                    Qualification Substitution Parameters:
                     DAY.01 = 11
                     YEAR.01 = 1999''',
                    '''Report the total catalog sales for customers in selected geographical regions or who made large purchases for a 
                    given year and quarter.
                    Qualification Substitution Parameters:
                     QOY.01 = 2
                     YEAR.01 = 2001
'''
                    ]
        tables=['query1','query2','query3','query4','query5']
        tab1,tab2,tab3,tab4,tab5=st.tabs(tables)
        tab_names=[tab1,tab2,tab3,tab4,tab5]
        part1_engine = create_engine(URL(
            account = 'pinvdzu-ljb05593',
            user = 'gamer9797123',
            password = 'Gamer9797123)',
            database = 'midterm',
            schema = 'public',
            warehouse = 'COMPUTE_WH',
            role='ACCOUNTADMIN',
        ))
        if sqlalchemy.inspect(part1_engine).has_table("query1"):
            for table_name,tab,question in zip(tables,tab_names,questions):
                with tab:
                    query="select * from "+str(table_name)
                    connection = part1_engine.connect()
                    res = pd.read_sql(query, connection)
                    title="Query number: "+str(table_name)
                    st.header(title)
                    st.write(question)
                    st.write(res)
                part1_engine.dispose()


def st_part2():
    part2_engine = create_engine(URL(
            account = 'pinvdzu-ljb05593',
            user = 'gamer9797123',
            password = 'Gamer9797123)',
            database = 'TPCDS_XGBOOST',
            schema = 'DEMO',
            warehouse = 'COMPUTE_WH',
            role='ACCOUNTADMIN',
        ))
    connection=part2_engine.connect()
    
    col1,col2=st.columns(2)
    with col1:
        dob_slider=st.slider('Date of Birth',1924,1992,(1924,1992))
        dob_list=list(dob_slider)

        education_option = st.multiselect(
            'Select Education Level',
            ['Advanced Degree','Secondary','2 yr Degree','4 yr Degree','College','Unknown','Primary'],
            ['Advanced Degree','Secondary','2 yr Degree'])

        gender_option=st.multiselect(
            'Select Customer Demographic by Sex',
            ['M','F'],
            ['F']
        )

        dept_option=st.multiselect(
            'Select Customer Dept',
            ['0','1'],
            ['0']
        )

        credit_option=st.multiselect(
            'Select Credit Rating',
            ['Low Risk',' Good', 'High Risk', 'Unknown'],
            ['Low Risk',' Good']
        )

        marital_option=st.multiselect(
            'Select Marital Status',
            ['S','M','W','U','D'],
            ['W','U']
        )
        st.subheader("Total Predicted vs Actual Sales")
        query,query2=run_query(list(dob_list),tuple(education_option),tuple(gender_option),tuple(dept_option),tuple(credit_option),tuple(marital_option))
        st.code(query,language='SQL')
        res2=pd.read_sql(query,connection)
        st.write('<b> Predicted Sales: </b>',round(res2.loc[0][0]/1000000000,2),'billion USD',unsafe_allow_html=True)        
        st.write("<b>Actual Sales: </b>",round(res2.loc[0][1]/1000000000,2),'billion USD',unsafe_allow_html=True)        
    with col2:   
        st.subheader("Year wise Predicted vs Actual Sales")
        st.code(query2,language='SQL')
        res3=pd.read_sql(query2,connection)
        st.write(res3)
        res4=res3
        res4=round(res4/1000000000,2)
        mse=mean_squared_error(res4.actual_sales,res4.predicted_value)
        rmse=math.sqrt(mse)
        st.write('<b>Model MSE: </b>',mse,unsafe_allow_html=True)
        st.write('<b>Model RMSE: </b>',rmse,unsafe_allow_html=True)
        res4['accu']=res4.apply(accuracy_calc,axis=1)
        st.write('<b> Model Accuracy: </b>',round((res4.accu.mean() )*100,2),unsafe_allow_html=True) 

    fig=px.line(res3,x='c_birth_year',y=['actual_sales','predicted_value'])
    fig.update_layout(autosize=True,
                      width=1600,
                      height=400)
    st.plotly_chart(fig)
    
    
if __name__=="__main__":
    main()

