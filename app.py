from flask import Flask, render_template, url_for
from pandas import Series, DataFrame
from pandas_datareader import DataReader
from datetime import datetime
from flask import Flask, make_response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from io import StringIO
import os

import random
import numpy as np
import pandas as pd
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
home = Flask(__name__)

tech_list = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'IBM', 'FB', 'BIDU', 'AVGO', 'TSLA', 'SNAP']

end = datetime.now()
start = datetime(end.year-1, end.month, end.day)

i = 1
risk = []
for stock in tech_list:
    globals()[stock] = DataReader(stock, 'morningstar', start, end)

    globals()[stock]['Close'].plot(legend=True, figsize=(17.5, 7))
    plt.style.use('ggplot')
    plt.title('Close Graph', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close', fontsize=12)
    data_path = os.path.join("static/images" + "/" + stock+ "/" +  "A.png")
    plt.savefig(data_path, transparent=True, bbox_inches='tight')
    plt.close()

    globals()[stock]['Volume'].plot(legend=True, figsize=(17.5, 7))
    plt.style.use('ggplot')
    plt.title('Volume Graph', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Volume', fontsize=12)
    data_path = os.path.join("static/images" + "/" + stock + "/" + "B.png")
    plt.savefig(data_path, transparent=True, bbox_inches='tight')
    plt.close()
    MA_day = [10, 20, 50, 100]

    for ma in MA_day:
        column_name = 'MA for %s days' % (str(ma))
        globals()[stock][column_name] = DataFrame.rolling(globals()[stock]['Close'], ma)

    globals()[stock][['Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days',
          'MA for 100 days']].plot(subplots=False, figsize=(17.5, 7))
    plt.style.use('ggplot')
    plt.title('Rolling Mean Graph', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Rolling Mean', fontsize=12)
    data_path = os.path.join("static/images" + "/" + stock + "/" + "C.png")
    plt.savefig(data_path, transparent=True, bbox_inches='tight')
    plt.close()

    globals()[stock]['Daily Return'] = globals()[stock]['Close'].pct_change()

    globals()[stock]['Daily Return'].plot(
        figsize=(17.5, 7), legend=True, linestyle='--', marker='o')
    plt.style.use('ggplot')
    plt.title('Daily Return Graph', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Daily Return', fontsize=12)
    data_path = os.path.join("static/images" + "/" + stock + "/" + "D.png")
    plt.savefig(data_path, transparent=True, bbox_inches='tight')
    plt.close()

    globals()[stock]['Daily Return'].hist(bins=100, figsize=(17.5, 7))
    
    plt.style.use('ggplot')
    plt.title('Daily Return Histogram', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Daily Return', fontsize=12)
    data_path = os.path.join("static/images" + "/" + stock + "/" + "E.png")
    plt.savefig(data_path, transparent=True, bbox_inches='tight')
    plt.close()
    plt.figure(figsize=(17.5, 7))
    sns.distplot(globals()[stock]['Daily Return'].dropna(),
                 bins=100, color='magenta')
    
    plt.style.use('ggplot')
    plt.title('Daily Return Histogram', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Daily Return', fontsize=12)
    data_path = os.path.join("static/images" + "/" + stock + "/" + "F.png")
    plt.savefig(data_path, transparent=True, bbox_inches='tight')
    plt.close()
    
    closingprice = DataReader(tech_list, 'morningstar', start, end)['Close']
    closingprice_df = closingprice.to_frame().reset_index()
    #closingprice_df.head(10)

    tech_returns = closingprice.pct_change()
    tech_returns_df = tech_returns.to_frame().reset_index()
    #tech_returns.head()

    rets = tech_returns.dropna()
    rets_df = rets.to_frame().reset_index()
    #rets.head()
    plt.figure(figsize=(17.5, 7))
    sns.distplot(globals()[stock]['Daily Return'].dropna(), bins=100, color='purple')
    
    plt.title('Daily Return Histogram', fontsize=14)
    plt.xlabel('Daily Return', fontsize=12)
    data_path = os.path.join("static/images" + "/" + stock + "/" + "G.png")
    plt.savefig(data_path, transparent=True, bbox_inches='tight')
    plt.close()
    risk.insert(i, rets[stock].quantile(0.05))
    i += 1
    #rets.head()

    days = 365

    dt = 1/days

    mu = rets.mean()

    sigma = rets.std()

    def stock_monte_carlo(start_price, days, mu, sigma):

        price = np.zeros(days)
        price[0] = start_price

        shock = np.zeros(days)
        drift = np.zeros(days)

        for x in range(1, days):
            shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
            drift[x] = mu * dt
            price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))

        return price

    start_price = 100.05
    plt.figure(figsize=(17.5, 7))
    for run in range(100):
        plt.plot(stock_monte_carlo(start_price, days, mu, sigma))

    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.title('Monte Carlo Analysis for morningstar')
    data_path = os.path.join("static/images" + "/" + stock + "/" + "H.png")
    plt.savefig(data_path, transparent=True, bbox_inches='tight')
    plt.close()    
    runs = 100
    simulations = np.zeros(runs)

    for run in range(runs):
        simulations[run] = stock_monte_carlo(start_price, days, mu, sigma)[days-1]

    q = np.percentile(simulations, 1)
    plt.figure(figsize=(17.5, 7))
    plt.hist(simulations, bins=200)

    plt.figtext(0.6, 0.8, s='Start Price: $%.2f' % start_price)

    plt.figtext(0.6, 0.7, s='Mean Final Price: $%.2f' % simulations.mean())

    plt.figtext(0.6, 0.6, s='VaR(0.99): $%.2f' % (start_price - q))

    plt.figtext(0.15, 0.6, s="q(0.99): $%.2f" % q)

    # Plot a line at the 1% quantile result
    plt.axvline(x=q, linewidth=4, color='r')

    # For plot title
    plt.title(s="Final price distribution for morningstar "+stock+ " after %s days" %
            days, fontsize=16)
    plt.xlabel('Price', fontsize=14)
    plt.ylabel('Risk', fontsize=14)
    data_path = os.path.join("static/images" + "/" + stock + "/" + "I.png")
    plt.savefig(data_path, transparent=True, bbox_inches='tight')
    plt.close()
@home.route('/')
def index():
    return render_template('home.html', AAPL = AAPL, GOOGL = GOOGL, MSFT = MSFT, AMZN = AMZN, 
                                        IBM = IBM, FB = FB, BIDU = BIDU, AVGO = AVGO, TSLA = TSLA, SNAP = SNAP, tech_list=tech_list, end = end, risk = risk)

if __name__ == '__main__':
    home.run(debug=True)
