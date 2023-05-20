# Long-Short-Statistical-Arbitrage
This is a pairs trading intraday strategy that uses LSTM based price forecasts and EWMA volatility forecasts.

#### We will be using git to collaborate on this project. Please go through the following details very carefully before you start working on this project

## GIT related stuff
Try using GPT to get a basic tutorial on git. We are not getting crazy with git related stuff just the basics is all that is required. The reason I am using GIT is because it will massively help with the collaboration and we can individually work on our parts without relying on others in the team. The most common commands that we will be using are git branch, git checkout branch_name, git commit -m "message", git push origin <branch-name>.
1. Clone this repository to your workplace folder and make sure that you have the master branch. Use cmd > git bash. DO NOT download as zip and extract.
2. Create a new branch for your work. For eg: if your name is Ananya branch should be feature/Ananya-feature-you-are-working-on. DO NOT forget to switch to your branch locally. DO NOT work on the master branch directly
3. Make changes in the ipynb file in your branch on your computer locally and commit them using git commit -m "message". Please make sure the message is short and sweet yet contains the most important characteristics of the change you made. 
4. Push the changes using git push to your branch. Once you are ready with your change create a pull request from your branch to master. Create a write up of the changes you made. We will do peer review with the team and after resolving comments we can merge your changes into the main project (master branch). If you run into errors or are stuck somewhere let me know.
  
## Outline of the Methodology
The goal of this project is to perform long-short statistical arbitrage using pairs trading on the most volatile stocks of SnP500 using their weights as reference for trading. As of now we have a Python script that involves procuring data, performing pattern analysis, and implementing a trading strategy using the obtained data. Let's go through the main methodology of the project step by step: 

1. Importing the necessary libraries, such as requests, pandas, numpy, yfinance, seaborn, matplotlib, statsmodels, sklearn, and tensorflow.keras. These libraries are used for data retrieval, analysis, visualization, modeling, and machine learning.

2. Procure data for the project. It extracts the tickers for the S&P 500 companies from Wikipedia and uses the Yahoo Finance API to download intraday hourly stock data for the entire year of 2022 for the identified tickers. The stock data is stored in separate DataFrames for each stock.

3. Calculates the annualized intraday volatility for each stock based on its daily returns. It applies an Exponential Moving Average (EMA) model to forecast the volatilities.

4. Select the top 50 stocks with the highest volatility and determines their respective weights using market capitalization weighting based on the S&P 500 data. The weights are used to construct trading pairs for a pairs trading strategy.

5. Pattern analysis is performed on the selected stocks, including visualizing the volatility, stock price autocorrelation, stock price differenced autocorrelation, and frequency distribution of returns.

6. The script defines a trading strategy using a combination of models. It utilizes an EMA model to forecast volatilities, LSTM (Long Short-Term Memory) model to forecast stock prices, and implements rules for generating trading signals and managing positions. The strategy involves creating long and short positions based on the forecasted prices and volatilities of the trading pairs.
	
## Components that we have completed
	
#### Identifying the right stocks
	I have already built the framework to correctly identify the most volatile stocks from SnP and assign them in pairs based on the correlation of their returns.
	
#### Generating trading signals (Partial)
	I have done the major chunk of work for us by successfully analyzing the patterns in volatility and having forecasting the prices and volatility for the stocks.

#### Boilerplate stuff done
	I have done preprocessing of the data, building the base models, setting up the framework for all of us to work within.
	
## Components that we need to work on
	
#### Incorporating something more the RNN
	Post the QAM lecture with guest speaker, I can see how Prof. Bernard wont be too much of a fan of neural networks, since its a big black box to whats actually happening. We want to incorporate some other modeling technique that we have done in the QAM assignments into the trading strategy apart from neural networks. One of us needs to think and analyze how to fit that into this project and come up with a proposal.
	
#### Generating Trading Signals (Partial)
	Currently this is the trading signal algorithm that I have come up with. 
    1. Identify undervalued and overvalued stocks based on trade volume, last close price, 20 day moving average using the LSTM model
    2. If the forecasted price for the day is above 1x forecasted volatility we create a long position in the stock and a short position in its pair in ratio of their snp weights
    3. If the forecasted price for the day is 1x forecasted volatility below we create a short position in the stock and a long position in its pair in ratio of their snp weights
    4. We add the two positions to a position table and associate each other using a pair id.
    5. The target price for the trade is the forecasted price. We check for this price every hour. We square off both the position when the price moves above (below for short) the forecast
    6. We keep a stop loss of 3%. If our net pair trade value falls below 3% we square off the position and remove it from the position table
	I want one of us (preferably someone other than me better with trading and finance) to analyze and understand whether this is the correct approach because this is something that I have pulled out of my ***. There HAS TO be a better way to use price and volatility forecasts to come up with a trading signal. Try asking GPT how to do that and come up with a better strategy.

### Calculating portfolio metrics
	I want one of us to build a very robust framework. This framework should be standalone. Try to understand how the output data from the trading strategy looks like and to calculate metrics - total trades, sharpe ratio, strategy capacity, win percentage, drawdown compared to SNP, information ratio, treynor ratio, profit-loss ratio, average win, turnover, annualized return, VaR, Expected Shortfall. Build it in such a manner that there is no dependency of you or for you on someone working with other components.
	


