#!/usr/bin/env python
# coding: utf-8

# In[373]:


import requests
import pandas as pd
import numpy as np
import yfinance as yf
import io
import seaborn as sns; sns.set_theme(color_codes=True)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# # Procuring Data
# First we need to extract the tickers for SnP-500

# In[14]:


# The first input contains the list of tickers
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Disable SSL verification warnings for Wikipedia
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

# Fetch the content without verifying the SSL certificate
response = requests.get(url, verify=False)

# Read the content as a CSV file with pandas
ticker_df = pd.read_html(io.StringIO(response.text))[0]
ticker_df


# #### We will rely on Yahoo Finance API for our data
# However YFinance is not always a reliable data measure since
# * Unofficial / not necessarily maintained
# * Can get you rate limited/blacklisted

# Lets verify if the intraday stock data is correctly downloading

# In[15]:


data = yf.download(tickers=['MMM', 'ACN'], start="2022-01-01", end="2022-12-31", interval='60m')
data


# Seems like API is working fine. Although please note that there can be timezone differences in datetime column so we might need to adjust times accordingly for real time trading.

# # !! Time to download all the data !! 
# The data is intraday hourly for the entire year of 2022

# In[16]:


data = yf.download(tickers=ticker_df["Symbol"].tolist(), interval='60m', start="2022-01-01", end="2022-12-31")
data


# In[17]:


# Store each stock's data in a separate DataFrame
stock_data = {}
unavailable = ["BRK.B", "BF.B", "GEHC"]
tickers = ticker_df["Symbol"].tolist()
tickers = [x for x in tickers if x not in unavailable] 
for ticker in tickers:
    stock_data[ticker] = data.xs(ticker, level=1, axis=1)
stock_data["MMM"]


# # Lets find annualized intraday volatility of each stock

# In[19]:


for ticker, df in stock_data.items():
    temp_df = df.copy().dropna()
    
    # calculate daily volatility
    temp_df["Returns"] = temp_df["Adj Close"]/temp_df["Adj Close"].shift(1)
    daily_volatility = temp_df['Returns'].groupby(pd.Grouper(freq='D')).std()

    # Convert the annualized_intraday_volatility series to a DataFrame
    daily_volatility_df = daily_volatility.to_frame(name='Daily Vol')

    # Perform an asof merge with left_index=True and right_index=True
    temp_df = pd.merge_asof(temp_df, daily_volatility_df, left_index=True, right_index=True, direction='forward')
    temp_df.fillna(method='ffill', inplace=True) # forward fill
    
    stock_data[ticker] = temp_df


# ### For our trading model, we will chose the top 50 stocks with most intraday volatility

# In[20]:


# Averaging intraday volatilites for all stocks
mean_vol = {ticker: df["Daily Vol"].mean() for ticker, df in stock_data.items()}

# Reverse sorting based on volatilities
mean_vol = sorted(mean_vol.items(), key=lambda x: x[1], reverse=True)
column_names = ['Ticker', 'Daily Vol']
trade_deck = pd.DataFrame(data=mean_vol[:50], columns=column_names)
trade_deck[:5]


# ### We also need SnP 500 weights for these stocks. Since this data is not available on any free API I will be using the crsp database provided by my university to calculate approximate daily weights for each of the top 50 volatile stock in SPX using market cap weighting

# In[21]:


df_snp = pd.read_csv("snp500_2022.csv")
df_snp["MARKETCAP"] = df_snp["PRC"] * df_snp["VOL"]
df_snp


# In[22]:


# group by date and find individual wts approximate by individual MARKETCAP/ total MARKETCAP
group = df_snp.groupby("date")
df_snp["SNPWT"] = group["MARKETCAP"].transform(lambda x: x/x.sum())
df_snp


# # Pattern Analysis

# In[23]:


# Lets look at some plots to analyze what kind of data does the volatilities exhibit
# Get 5 tickers -> lets take 1, 10, 20, 30 tickers 
ats = [trade_deck.iloc[i]["Ticker"] for i in [1,18,32,45]]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
row, col = 0,0
for ticker in ats:
    if ticker in stock_data:
        group = stock_data[ticker]
        x_values = mdates.date2num(group.index)
        scatter_kws = {'alpha':0.6}  # set marker color
        line_kws = {'color': "black"}
        sns.regplot(ax=axes[row, col], data=group, x=x_values, y='Daily Vol', scatter_kws=scatter_kws, line_kws=line_kws, ci=100)
        col = col+1
        if col == 2:
            row = row+1
            col=0


# Most of the data points are scattered with heavy noise and also have a number of outliers.

# ### Volatility Partial Autocorrelations

# In[62]:


fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(30, 20))
axs = axs.ravel()
i=0
ats = trade_deck["Ticker"].sample(n=15).values
for ticker in ats:
    #plot_acf(stock_data[ticker]["Intraday Vol"])
    groupby_obj = stock_data[ticker].groupby(pd.Grouper(freq='D'))
    y = groupby_obj.first()["Daily Vol"]
    y.fillna(y.mean(), inplace=True)
    plot_pacf(y, ax=axs[i])
    i+=1


# The first lag is kind of significant which means yesterday's volatility is our best guess for the forecast of today's volatility for most of these volatilies. A way to model intraday volatility could be ARCH(1).

# ### Stock price partial autocorrelation

# In[64]:


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
ats = trade_deck["Ticker"].sample(n=4).values
row, col = 0, 0
for ticker in ats:
    #plot_acf(stock_data[ticker]["Intraday Vol"])
    groupby_obj = stock_data[ticker].groupby(pd.Grouper(freq='D'))
    y = groupby_obj["Adj Close"].median()
    y.fillna(y.mean(), inplace=True)
    plot_pacf(y, ax=axs[row, col])
    col = col+1
    if col == 2:
        row = row+1
        col=0


# ##### Stock price differenced autocorrelation

# In[72]:


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
ats = trade_deck["Ticker"].sample(n=4).values
row, col = 0, 0
for ticker in ats:
    #plot_acf(stock_data[ticker]["Intraday Vol"])
    groupby_obj = stock_data[ticker].groupby(pd.Grouper(freq='D'))
    y = groupby_obj["Adj Close"].median()
    y.fillna(y.mean(), inplace=True)
    y = y
    y.fillna(y.mean(), inplace=True)
    plot_acf(y, ax=axs[row, col])
    col = col+1
    if col == 2:
        row = row+1
        col=0


# The daily stock prices seem to be difficult to model using time series.

# ### Frequency distribution of returns

# In[30]:


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
row, col = 0, 0
for ticker in ats:
    axs[row, col].hist(stock_data[ticker]["Returns"], bins=100, density=True, alpha=0.5, color='blue')
    col = col+1
    if col == 2:
        row = row+1
        col=0


# The return series resemble a normal distribution to a good extent.

#  # Trading strategy

# ### 1. We will use a pairs trading strategy for long short abitrage. For each high volatility stock we identify another one with highest correlation to its returns and assign them as a pair. If no such pair found we assign a pair randomly

# In[295]:


pairs = []
for stock in trade_deck["Ticker"]:
    pair = ''
    prev_c = 0
    for stock2 in trade_deck["Ticker"]:
        c = stock_data[stock]["Returns"].corr(stock_data[stock2]["Returns"])
        if stock2 != stock and c > 0.5 and c > prev_c:
                prev_c = c
                pair = stock2
    pairs.append(pair)
    
trade_deck["Pair"] = pairs
trade_deck.loc[trade_deck["Pair"]=="", "Pair"] = trade_deck["Ticker"].sample(n=1).values[0]
trade_deck


# ### 2. Since the 1st lag is the most significant we can use Exponential Moving Average (EMA) model to model volatilites with a weight of lets say 0.9

# In[377]:


def ema_forecast(train_vol, alpha, horizon):
    ema = train_vol.ewm(alpha=alpha).mean()
    return ema.iloc[-1] * np.ones(horizon)

fig, axs = plt.subplots(5, 10, figsize=(30,20))
row, col = 0, 0
vol_fc = {}
for stock in trade_deck["Ticker"]:
    
    # For June to Sept
    train = stock_data[stock].resample('D').mean()
    train.fillna(train.mean(), inplace=True)
    test = train.loc['2022-06':'2022-09'].copy()
    test = test[["Daily Vol"]]

    alpha = 0.9
    forecasts = []

    for i in test.index:
        daily_vol = train.loc[(train.index < i)]["Daily Vol"]
        forecast = ema_forecast(daily_vol, alpha, 1)
        forecasts.append(forecast[0])

    test["Forecasts"] = forecasts
    vol_fc[stock] = test["Forecasts"]
    axs[row, col].plot(test["Forecasts"], label="Forecasts")
    axs[row, col].plot(test["Daily Vol"], label="Actual")
    col += 1
    if col == 10:
        row += 1
        col = 0


# The forecasts and actual volatility plots are above for the 50 stocks in contention

# # Signal Generation

# ### 3. We will use LSTM model to forecast prices based on which we will be creating positions
# Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture designed to learn and predict sequences of data where time series cannot efficiently model. Following is our trading algorithm
# 
# 1. Identify undervalued and overvalued stocks based on trade volume, last close price, 20 day moving average using the LSTM model
# 2. If the forecasted price for the day is above 1x forecasted volatility we create a long position in the stock and a short position in its pair in ratio of their snp weights
# 3. If the forecasted price for the day is 1x forecasted volatility below we create a short position in the stock and a long position in its pair in ratio of their snp weights
# 4. We add the two positions to a position table and associate each other using a pair id.
# 5. The target price for the trade is the forecasted price. We check for this price every hour. We square off both the position when the price moves above (below for short) the forecast
# 6. We keep a stop loss of 3%. If our net pair trade value falls below 3% we square off the position and remove it from the position table
# 7. Calculate returns of portfolio each hour and sum them.
# 8. Calculate metrics - total trades, sharpe ratio, strategy capacity, win percentage, drawdown compared to SNP, information ratio, treynor ratio, profit-loss ratio, average win, turnover, annualized return, VaR, Expected Shortfall

# In[374]:


# Preprocess the data
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), :])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Format the data
def format_df(df):
    # Select columns
    model_data = df[['Adj Close', 'Volume', 'Rolling Price']].values

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    model_data = scaler.fit_transform(model_data)
    return model_data, scaler

def model_fit(df):
    # Set look_back window
    look_back = 1

    # Load your data
    # df = ...

    # Format and scale the data
    model_data, scaler = format_df(df)
    
    # Create dataset for the LSTM model
    X, Y = create_dataset(model_data, look_back)

    # Reshape the input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 3))

    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    model.fit(X, Y, epochs=100, batch_size=32, verbose=0)
    return model

def predict(model, df):
    model_data, scaler = format_df(df)
    
    # Make a prediction for the next day
    last_data = np.array([model_data[-1]])  # Take the last row of the dataset
    next_day_prediction = model.predict(last_data.reshape(1, 1, -1), verbose=0)

    # Invert scaling to get the original scale
    next_day_prediction = scaler.inverse_transform(np.hstack([next_day_prediction, last_data[:, 1:]]))[:, 0]

    return next_day_prediction[0]


# In[394]:


data = yf.download(tickers=trade_deck["Ticker"].tolist(), start="2018-01-01", end="2022-09-30", interval='1d')
data.fillna(data.mean(), inplace=True)

# Store each stock's data in a separate DataFrame
temp = {}
for ticker in trade_deck["Ticker"]:
    temp[ticker] = data.xs(ticker, level=1, axis=1)
data = temp

models = {}
window = 20
price_forecasts = []
for ticker in trade_deck["Ticker"]:
    df = data[ticker].copy()
    df["Rolling Price"] = df["Adj Close"].rolling(window=window).mean()
    df["Rolling Price"].fillna(df["Rolling Price"].mean(), inplace=True)
    df = df.drop(['Open', 'Close', 'High', 'Low'], axis=1)
    df["Market Cap"] = df["Adj Close"] * df["Volume"]
    
    # Training-Test split 
    split = df.index.get_loc(pd.to_datetime('2022-06-01 00:00:00-04:00'))
    
    # Price Forecasts
    forecasts = [0 for i in range(split)]
    model = model_fit(df[:split])
    models[ticker] = model
    for i in range(split, len(df)):
        forecasts.append(predict(model, df[:i]))    
    df["Price Forecasts"] = forecasts
    
    # Volatility Forecasts
    df = df.join(vol_fc[ticker].rename("Volatility Forecasts"))
    data[ticker] = df


# In[395]:


# Backing up the data and models
backup_dfs = data
backup_models = models
position_table = pd.DataFrame(columns=["Id", "Creation Date", "Square Off Date" "PairId", "Ticker", "Target", "Price"])


# In[393]:


test = stock_data[ticker].loc['2022-06':'2022-09'].copy()
forecast_df = data[ticker][["Price Forecasts", "Volatility Forecasts"]]
test = test.join(forecast_df.set_index(forecast_df.index.date), how='left')


# ## PS: REQUEST YOU FOR SOME TIME TO COMPLETE THIS PROJECT!! 

# In[412]:


export = pd.concat(data.values(), axis=0)
export = export.reset_index().rename(columns={'index': 'Trade date'})
export['Ticker'] = list(data.keys()) * len(data['ETSY'])
export.to_csv('snp_volatility_trained_data_2022.csv', index=False)

