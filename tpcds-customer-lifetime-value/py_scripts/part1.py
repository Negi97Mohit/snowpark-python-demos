import pandas as pd
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine

def part1():
  queries=[
  '''with year_total as (
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
    '''
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
    and d3.d_year between 1999 AND 1999 + 2) x
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
    select  * from
  (select 'store' channel, i_brand_id,i_class_id,i_category_id
          ,sum(ss_quantity*ss_list_price) sales, count(*) number_sales
  from store_sales 
      ,item
      ,date_dim
  where ss_item_sk in (select ss_item_sk from cross_items)
    and ss_item_sk = i_item_sk
    and ss_sold_date_sk = d_date_sk
    and d_week_seq = (select d_week_seq
                      from date_dim
                      where d_year = 1999 + 1
                        and d_moy = 12
                        and d_dom = 11)
  group by i_brand_id,i_class_id,i_category_id
  having sum(ss_quantity*ss_list_price) > (select average_sales from avg_sales)) this_year,
  (select 'store' channel_2, i_brand_id as i_brand_id_2, i_class_id as i_class_id_2
          ,i_category_id as i_category_id_2 , sum(ss_quantity*ss_list_price) sales_2, count(*) number_sales_2
  from store_sales
      ,item
      ,date_dim
  where ss_item_sk in (select ss_item_sk from cross_items)
    and ss_item_sk = i_item_sk
    and ss_sold_date_sk = d_date_sk
    and d_week_seq = (select d_week_seq
                      from date_dim
                      where d_year = 1999
                        and d_moy = 12
                        and d_dom = 11)
  group by i_brand_id,i_class_id,i_category_id
  having sum(ss_quantity*ss_list_price) > (select average_sales from avg_sales)) last_year
  where this_year.i_brand_id= last_year.i_brand_id_2
    and this_year.i_class_id = last_year.i_class_id_2
    and this_year.i_category_id = last_year.i_category_id_2
  order by this_year.channel, this_year.i_brand_id, this_year.i_class_id, this_year.i_category_id
    limit 100
  ''',

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
      account = 'ieimkyf-atb67363',
      user = 'MOHITSNEGI123',
      password = 'Hollyhalston97)',
      database = 'SNOWFLAKE_SAMPLE_DATA',
      schema = 'TPCDS_SF10TCL',
      warehouse = 'COMPUTE_WH',
      role='ACCOUNTADMIN',
  ))

  engine_new = create_engine(URL(
      account = 'ieimkyf-atb67363',
      user = 'MOHITSNEGI123',
      password = 'Hollyhalston97)',
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