import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
import certifi
from prophet import Prophet

st.title('Border Crossing forecast')
st.markdown('''This program gives forecasts for border crossings''')
st.header('Historical data')
ca = certifi.where()
client = MongoClient("mongodb+srv://phoenixadt:uwindsor123@bordercrossing.f5n1clr.mongodb.net/?retryWrites=true&w=majority", tlsCAFile=ca)
dbname = client['BorderCrossing']
collection = dbname['cleanedwithcoords']
documents = collection.find()
df = pd.DataFrame(documents)
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