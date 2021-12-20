import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
#from config import DBConfig
import datetime
import click
from streamlit_autorefresh import st_autorefresh
import altair as alt


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_con():
    df=pd.read_csv('db_config.csv')
    config = {
      'user': df['user'][0],
      'password': df['password'][0],
      'host': df['host'][0],
      'database': df['database'][0],
      'raise_on_warnings': df['raise_on_warnings'][0] 
    }
    # create the connection URL
    url='mysql+pymysql://'+df['user'][0]+':'+df['password'][0]+'@'+df['host'][0]+'/'+df['database'][0]+'?charset=utf8mb4'
    # get an instance of SQL Alchemy engine for the DB connection
    return create_engine(url,convert_unicode=True)


@st.cache(allow_output_mutation=True, show_spinner=False, ttl=6)
def get_data():
    timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    df = pd.read_sql_table('tweet_sentiments', get_con())
    df = df[df.crypto!='']
    process_time=df[df['processed_at'].dt.date==datetime.datetime.now().date()].sort_values(['processed_at'],ascending=False).head(100)
    process_time=process_time['process_time'].astype(int).sum()/100
    latency=str({datetime.datetime.now():[process_time]})
    return df, timestamp

def filter_by_date(df, start_date, end_date):
    df_filtered = df.loc[(df.created_at.dt.date >= start_date) & (df.created_at.dt.date <= end_date)]
    return df_filtered

def filter_by_time(df):
    current = datetime.datetime.now()  
    right = current + datetime.timedelta(0,300) 
    left= current - datetime.timedelta(0,300) 
    df_filtered = df.loc[(df.created_at.dt.time > left.time()) &(df.created_at.dt.time<right.time())]
    return df_filtered

@st.cache(show_spinner=False)
def filter_by_subject(df, subjects):
    return df[df.crypto.isin(subjects)]


@st.cache(show_spinner=False)
def count_plot_data(df, freq):
    plot_df = df.set_index('created_at').groupby('crypto').resample(freq).tweet_id.count().unstack(level=0, fill_value=0)
    plot_df.index.rename('Date', inplace=True)
    plot_df = plot_df.rename_axis(None, axis='columns')
    return plot_df


@st.cache(show_spinner=False)
def sentiment_plot_data(df, freq):
    df['polarity']=df['polarity'].astype('double')
    #plot_df = filter_by_time(df)
    plot_df = df.set_index('created_at').groupby('crypto').resample(freq).polarity.mean().unstack(level=0, fill_value=0)
    #set_index('created_at').groupby('crypto').polarity.mean().to_frame().rename(columns={'crypto':'sentiments'}).reset_index()
    plot_df.index.rename('Date', inplace=True)
    plot_df = plot_df.rename_axis(None, axis='columns')
    #plot_df = filter_by_time(plot_df)

    return plot_df


st.set_page_config(layout="wide", page_title='Twitter Setiment Analysis')
count = st_autorefresh(interval=5000, key="fizzbuzzcounter")

data, timestamp = get_data()

st.header('Twitter Sentiment Anaysis')
Info=st.container()


with Info:
    col1,col2=st.columns(2);
    with col1:
        st.metric(label="Total tweet count:", value=str(data.shape[0]))
        st.write('Data updated at {}'.format(timestamp))
    with col2:

        #col4.write('Avg Latency {}'.format(2))
        process_time=data[data['processed_at'].dt.date==datetime.datetime.now().date()].sort_values(['processed_at'],ascending=False).head(100)
        avg_time=process_time['process_time'].astype(int).sum()/100
        st.metric(label="Avg Processing Time", value=str(round(avg_time/20,4))+' s/tweet')
        st.metric(label="Avg Latency(for 20 tweets)", value=str(data['process_time'].head(20)[0])+' s')

col3,col4 = st.columns(2) 

date_options = data.created_at.dt.date.unique()
start_date_option = st.sidebar.selectbox('Select Start Date', date_options, index=0)
end_date_option = st.sidebar.selectbox('Select End Date', date_options, index=len(date_options)-1)

keywords = data.crypto.unique()
keyword_options = st.sidebar.multiselect(label='Subjects to Include:', options=keywords.tolist(), default=keywords.tolist())

data_subjects = data[data.crypto.isin(keyword_options)]
data_daily = filter_by_date(data_subjects, start_date_option, end_date_option)

top_daily_pos_tweets = data_daily[(data_daily['sentiment']=='positive') & (data['processed_at'].dt.date==datetime.datetime.now().date())].sort_values(['polarity'], ascending=False).head(10).copy()
top_daily_neg_tweets = data_daily[(data_daily['sentiment']=='negative') & (data['processed_at'].dt.date==datetime.datetime.now().date())].sort_values(['polarity'], ascending=True).head(10).copy()

col3.subheader('Top Positive Tweets')
col3.dataframe(top_daily_pos_tweets[['tweet', 'crypto', 'created_at', 'polarity']].reset_index(drop=True), 1000, 400)

col4.subheader('Top Negative Tweets')
col4.dataframe(top_daily_neg_tweets[['tweet', 'crypto', 'created_at', 'polarity']].reset_index(drop=True), 1000, 400)
#col2.dataframe(data_daily[['tweet', 'crypto', 'created_at', 'sentiment']].sort_values(['created_at'], ascending=False).reset_index(drop=True).head(10))


col5,col6=st.columns(2)


plot_freq_options = {
    'Minute': 'T',
    'Hourly': 'H',
    'Four Hourly': '4H',
    'Daily': 'D'
}
plot_freq_box = st.sidebar.selectbox(label='Plot Frequency:', options=list(plot_freq_options.keys()), index=0)
plot_freq = plot_freq_options[plot_freq_box]


col5.subheader('Tweet Volumes')
plotdata = count_plot_data(data_daily, plot_freq)
col5.line_chart(plotdata)



with col6:
    col6.subheader('Tweet Volume per Currency')
    
    curr_df=filter_by_time(data_daily)
    line=alt.Chart(curr_df).mark_line().encode(
        y='count(tweet_id):Q',
        x='processed_at:T',
        color=alt.Color('crypto:N', scale=alt.Scale(scheme='dark2')),
        tooltip='count(tweet_id):Q'
        ).configure_range(
    category={'scheme':'lightmulti'}).interactive()
    st.altair_chart(line)


col7,col8=st.columns(2)

col7.subheader('Sentiment')
plotdata2 = sentiment_plot_data(data_daily, plot_freq)
col7.line_chart(plotdata2) 


with col8:
    data_daily['processed_at']=data_daily['processed_at'].dt.date.astype(str)
    data_daily=data_daily.groupby(['crypto','processed_at','sentiment']).size().reset_index()
    data_daily.rename(columns={0:'count'},inplace=True)
    data_daily = data_daily[data_daily.crypto!='']
    col8.subheader('Positive-Negaitve tweets per Currency')

    bar=alt.Chart(data_daily).mark_bar().encode(
        x='crypto:N',
        y='sum(count):Q',
        color=alt.Color('sentiment:N', scale=alt.Scale(scheme='dark2')),
        column='processed_at:T',
        tooltip='sum(count):Q'
    ).configure_range(
    category={'scheme':'lightmulti'})

    st.altair_chart(bar)
    
   
#col2.subheader('Sentiment')
#plotdata2 = sentiment_plot_data(data_daily, plot_freq)
#col2.line_chart(plotdata2)
