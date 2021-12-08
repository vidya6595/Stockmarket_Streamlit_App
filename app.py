import streamlit as st
from datetime import date, timedelta, datetime
import yfinance as yf 
import datetime


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy
from datetime import date
import yfinance as yf
import pandas_datareader as data
import streamlit as st
from sklearn.preprocessing import power_transform
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.callbacks import EarlyStopping 

import tensorflow as tf

import math
from sklearn.metrics import mean_squared_error


st.title('Stock Market Analysis')



user_input = st.sidebar.text_input("Enter the Stock Code of company","M&MFin.NS")
tickerSymbol = user_input    
tickerData = yf.Ticker(tickerSymbol)


# Fundamental

# Sidebar



start = '2010-01-01'
today= date.today().strftime("%Y-%m-%d")

st.sidebar.subheader('Select Date')

start = st.sidebar.date_input("Start date", datetime.date(2010, 1, 1))

end = st.sidebar.date_input("End Date",datetime.date(2021, 1, 31))

# Retrieving tickers data
#ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
#tickerSymbol = st.sidebar.selectbox('Stock ticker', ticker_list)+'.NS'

#ticker_list = pd.read_csv(r"C:\\Users\\hp\Desktop\\Latest\\Nifty1000.csv")
#tickerSymbol = st.sidebar.selectbox('Nifty Stock ticker', ticker_list )+'.NS' # Select ticker symbol
#tickerSymbol
tickerData = yf.Ticker(tickerSymbol) # Get ticker data
df = tickerData.history(period='1d', start=start, today=today)

string_logo = '<img src=%s>' % tickerData.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)

string_name = tickerData.info['longName']
st.header('*%s*' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)

import pandas_datareader as webreader
#df = webreader.DataReader(user_input, start=start, end=end, data_source="yahoo")
#df = yf.download(user_input, start=start, end=end)



st.subheader('Latest Price')
st.write(df.tail(1))

st.subheader('52 Week High & Low Price')

df['52WH'] = df.Close.rolling(256).max()
df['52WL'] = df.Close.rolling(256).min()
st.write('High',round(df['52WH'][-1],2))
st.write('Low',round(df['52WL'][-1],2))



check_box = st.sidebar.checkbox(label="Dispaly Historical Data")

if check_box:
    st.write(df)
    


#n_years = st.sidebar.slider("YEARS DATA",1,10)
#period = n_years*365

#st.subheader('Historical Data')
#st.write(df)




def plot_graph():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index,y=df['Open'],name="Stock_Open",line_color='red'))
    fig.add_trace(go.Scatter(x=df.index,y=df['Close'], name="Stock_Close", line_color='blue'))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    return fig


plot_graph()

train_df = df.sort_values(by=['Date']).copy()

df1 = df.reset_index()['Close']

scaler=MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.80)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model = Sequential()

# Model with n_neurons = inputshape Timestamps, each with x_train.shape[2] variables
n_neurons = X_test.shape[1] * X_train.shape[2]
print(n_neurons, X_train.shape[1], X_train.shape[2])
model.add(LSTM(n_neurons, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))) 
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Training the model
epochs = 2
batch_size = 64
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,validation_data=(X_test, ytest))


#model = load_model('Staked_LSTM.h5')

## Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

# plot baseline and predictions
st.title('Actual vs Prediction')
fig = plt.figure(figsize=(10,5))
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.legend(["Test",'Train',"Prediction"], loc="upper left")
st.pyplot(fig)
user_input1 = len(test_data)-100
x_input=test_data[user_input1:].reshape(1,-1)   

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# demonstrate prediction for next days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1,(len(temp_input)+100)+-n_steps-1, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        #print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        #print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

#st.write(lst_output)

df4 = scaler.inverse_transform((lst_output))

day_new=np.arange(1,101)               # As we know we taking 100 as time stamp, that's we considering (1,101)
day_pred=np.arange(101,131)            # As we planning to preditic 30days,(101+30=131)

input_data2 = (len(df1)-100)

st.title ('Stock Prediction')

st.subheader('Prediction Graph')
fig3=plt.figure(figsize=(10,5))
plt.plot(day_new,scaler.inverse_transform(df1[input_data2:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
st.pyplot(fig3)


#st.title('Prediction Graph')
#fig2 = plt.figure(figsize=(10,5))
#plt.plot(day_new,scaler.inverse_transform(df1[1624:]))
#st.pyplot(day_pred,scaler.inverse_transform(lst_output))

df3=df1.tolist()
df3.extend(lst_output)

df3 = scaler.inverse_transform(df3).tolist()

#fig4=plt.figure(figsize=(10,5))
#df3=df1.tolist()
#df3.extend(lst_output)
#plt.plot(df3[input_data2:])
#plt.legend(["Prediction"], loc="upper left")
#st.pyplot(fig4)


#st.title('Prediction Graph')
#fig2 = plt.figure(figsize=(10,5))
#plt.plot(df3)
#plt.legend(["Prediction"], loc="upper left")
#st.pyplot(fig2)

#plt.figure(figsize=(10,5))
#st.pyplot(df3)


# Range for 1 month
st.title('Prediction Range for next 30Days')
Range = df4[0],df4[-1]
st.write(Range)



# Sentiment

# Import libraries
from textblob import TextBlob
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from urllib.request import urlopen
from urllib.request import Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Parameters 
n = 100 #the # of article headlines displayed per ticker
tickers = [tickerSymbol] 


# Get Data
finviz_url = 'https://finviz.com/quote.ashx?t='
news_tables = {}

for ticker in tickers:
    url = finviz_url + ticker
    req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'}) 
    resp = urlopen(req)    
    html = BeautifulSoup(resp, features="lxml")
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

try:
    for ticker in tickers:
        df = news_tables[ticker]
        df_tr = df.findAll('tr')
    
        print ('\n')
        print ('Recent News Headlines for {}: '.format(ticker))
        
        for i, table_row in enumerate(df_tr):
            a_text = table_row.a.text
            td_text = table_row.td.text
            td_text = td_text.strip()
            print(a_text,'(',td_text,')')
            if i == n-1:
                break
except KeyError:
    pass


# Iterate through the news
parsed_news = []
for file_name, news_table in news_tables.items():
    for x in news_table.findAll('tr'):
        text = x.a.get_text() 
        date_scrape = x.td.text.split()

        if len(date_scrape) == 1:
            time = date_scrape[0]
            
        else:
            date = date_scrape[0]
            time = date_scrape[1]

        ticker = file_name.split('_')[0]
        
        parsed_news.append([ticker, date, time, text])

analyzer = SentimentIntensityAnalyzer()

columns = ['Ticker', 'Date', 'Time', 'Headline']
news = pd.DataFrame(parsed_news, columns=columns)
scores = news['Headline'].apply(analyzer.polarity_scores).tolist()

df_scores = pd.DataFrame(scores)
news = news.join(df_scores, rsuffix='_right')


# View Data 
news['Date'] = pd.to_datetime(news.Date).dt.date

unique_ticker = news['Ticker'].unique().tolist()
news_dict = {name: news.loc[news['Ticker'] == name] for name in unique_ticker}

values = []
for ticker in tickers: 
    dataframe = news_dict[ticker]
    dataframe = dataframe.set_index('Ticker')
    dataframe = dataframe.drop(columns = ['Headline'])
    print ('\n')
    print (dataframe.head())
    
    mean = round(dataframe['compound'].mean(), 2)
    values.append(mean)
    
df = pd.DataFrame(list(zip(tickers, values)), columns =['Ticker', 'Mean Sentiment']) 
df = df.set_index('Ticker')
df = df.sort_values('Mean Sentiment', ascending=False)
#print ('\n')
#print (df)

# Let's calculate subjectivity and polarity

# Subjectivity
def calc_sub(Headline):
    return TextBlob(Headline).sentiment.subjectivity

# Polarity
def calc_pola(Headline):
    return TextBlob(Headline).sentiment.polarity

news['subjectivity'] = news.Headline.apply(calc_sub)
news['polarity']     = news.Headline.apply(calc_pola)

# Classify tweets based on polarity

def sentiment(polarity):
    result = ''
    if polarity > 0:
        result = 'Positive'
    elif polarity == 0:
        result = 'Netural'
    else:
        result = 'Negative'
    return result

    

news['Sentiment'] = news.polarity.apply(sentiment)


fig, ax = plt.subplots()
news.Sentiment.value_counts().plot(kind='bar')
st.title('Sentiment Analysis Using Newspaper Headline')
st.pyplot(fig)

# Let's see the percentage of different sentiment's class

# creat
Df_Sentiment = pd.DataFrame(news.Sentiment.value_counts(normalize=True)*100)

# Calculating percentage
Df_Sentiment['Total'] = news.Sentiment.value_counts()

    
    

#Df_Sentiment

