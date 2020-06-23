import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import pandas as pd
import datetime

# Get Ticker Data
aapl = yf.Ticker('AAPL')

# Get Historical Data
time_period = '5y'
hist = aapl.history(period=time_period)

# Get Close Data
close_prc = hist['Close']
time_len = len(close_prc)

# Close Price Change Data
diff_close = np.zeros((1, time_len))
diff_close = diff_close[0]

for i in range(0, time_len - 1):
    diff_close[i] = close_prc[i + 1] - close_prc[i]

avg_growth_rate = mean(diff_close)

# Get Other Financial Data
net_income = aapl.info['netIncomeToCommon']

trail_eps = aapl.info['trailingEps']

price_to_book = aapl.info['priceToBook']

reg_prc = aapl.info['regularMarketPrice']

p_e = reg_prc/trail_eps

peg = aapl.info['pegRatio']     # Price-to-Earnings/ EPS Growth Rate

eps_growth = p_e/peg    # EPS Growth Rate [percentage]

book_to_earn = p_e/price_to_book    # Book value/ Earnings

# Best Fit Line
time = np.linspace(0, len(close_prc), len(close_prc))

poly_coeff = np.polyfit(time, close_prc, 2)


def best_fit_quad(t, a):
    return a[0]*(t**2) + a[1]*t + a[2]


# TODO: Make this more dynamic (i.e able to check any future date and plot it)
next_pos = best_fit_quad(max(time) + 30, poly_coeff)
next_date = datetime.datetime(2020, 7, 22)


# Make Plots
plt.subplot(211)
plt.tight_layout()
plt.plot(close_prc)     # Closing Price Plot
plt.plot(hist.index, best_fit_quad(time, poly_coeff))
plt.plot(next_date, next_pos,'ro')
plt.xlabel('Date')
plt.ylabel('Price [$]')
plt.title('Historical Close Prices')
plt.legend(('Close Price', 'Best-Fit Line', 'Predicted Value'))

plt.subplot(223)
plt.tight_layout()
plt.plot(hist.index, diff_close)    # Close Price Change Plot
plt.axhline(avg_growth_rate, color='r')
plt.xlabel('Date')
plt.ylabel('Price Change [$]')
plt.title('Change in Closing Price')

plt.subplot(224)
plt.tight_layout()
plt.plot(aapl.dividends)    # Dividends Plot
plt.xlabel('Date')
plt.ylabel('Dividend Yield [Percentage]')
plt.title('Dividend Yield')

plt.show()

# Print Out Results
print('-------------------------------------------------------------------')
print(f'Net Income: ${net_income}')
print(f'Trailing EPS (Earning Per Share): ${trail_eps:0.3f}')
print(f'Price-To-Book Ratio: {price_to_book:0.3f}')
print(f'Market Price: ${reg_prc}')
print(f'PE Ratio: {p_e:0.3f}')
print(f'PE-To-Growth Ratio: {peg:0.3f}')
print(f'EPS Growth Rate: {p_e/peg:0.3f}%')
print(f'Average Close Price Growth (Monthy): ${avg_growth_rate*30:0.3f}')
print(f'Book Value/Earnings Ratio: {p_e/price_to_book:0.3f}')
print(f'Estimated Value For Specified Time: ${next_pos:0.3f}')
print('-------------------------------------------------------------------')

