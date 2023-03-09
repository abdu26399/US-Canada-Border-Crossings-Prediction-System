import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from pyspark.sql import SparkSession
from prophet import Prophet
import plotly.graph_objects as go


client = MongoClient()
dbname = client['BorderCrossing']
collection = dbname['collection']
documents = collection.find()
df = pd.DataFrame(documents)
mapsdocs= dbname['maps'].find()
temp_df2 = pd.DataFrame(mapsdocs)

st.title('Analysis and Exploration of US-Canada Border Tourist Data')
st.markdown('''This program gives forecasts for border crossings''')
st.header('Historical data')
with st.form('historical_chart'):
    selected_city = st.selectbox(label='City', options=df['GEO'].unique())
    submit_pressed = st.form_submit_button('Submit')
    if submit_pressed:
        dfp = df[(df.GEO==selected_city)]
        # dfp = dfp[['date','VALUE', 'city_codes']]
        dfp = dfp[dfp.date<'2019-06-01']
        hist_chart = px.line(dfp, x='date', y='VALUE', labels={"VALUE":"no of vehicles"})
        st.plotly_chart(hist_chart)


st.header('Forecast')
with st.form('forecast_chart'):
    selected_city = st.selectbox(label='City', options=df['GEO'].unique())
    months = st.selectbox('Enter months to forecast', (3,6,9,12))
    forecast_pressed = st.form_submit_button('Forecast')
    if forecast_pressed:
        df_for_training = df[df.GEO==selected_city]
        df_for_training = df[df.date<'2019-06-01']
        df_for_training = df_for_training[['date','VALUE']]
        df_for_training = df_for_training.rename(columns={'date':'ds', 'VALUE':'y'})
        model = Prophet()
        model.fit(df_for_training)
        df_for_predict = model.make_future_dataframe(
        periods=months+43,
        freq='m',
        include_history=False)
        df_for_predict = df_for_predict[df_for_predict.ds>'2022-11-30']
        df_for_predict = model.predict(df_for_predict)
        forecast_chart = px.line(df_for_predict, x='ds', y='yhat', labels={"ds":"date", "yhat":"no of vehicles"})
        st.plotly_chart(forecast_chart)


temp_df2 = temp_df2[temp_df2.Year<2020]
# year = 2019
year = int(st.text_input("Enter year", 2019))

plot_spot = st.empty()

temp_df = temp_df2[temp_df2['Year'] == year]
temp_df['text'] = temp_df['GEO'] + '\nTourists ' + (temp_df['VALUE']).astype(str)
#limits = [(0,5000),(5001,15000),(15001,40000),(40001,75000),(75001,100000)]
#colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]
#cities = []
scale = 4000

fig = go.Figure()

fig.add_trace(go.Scattergeo(
locationmode = 'USA-states',
lon = temp_df['lon'],
lat = temp_df['lat'],
text = temp_df['text'],
marker = dict(
    size = temp_df['VALUE']/scale,
    color = 'crimson',
    line_color='rgb(40,40,40)',
    line_width=0.5,
    sizemode = 'area'
),
name = 'No.of Tourists'))

fig.update_layout(
  autosize=False,
  width=700,
  height=500,
  title_text = 'Canada tourist data<br>(Click legend to toggle traces)',
  showlegend = True,
  geo = dict(
  scope = 'north america',
  landcolor = 'rgb(217, 217, 217)',
  )
)
with plot_spot:
    st.plotly_chart(fig, use_container_width=True)
#fig.show()