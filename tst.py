import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean


# Get Data
aapl = yf.Ticker('AAPL')

# Historical Data
time_period = '5y'
hist = aapl.history(period=time_period)

# Plot Hist Data
plt.figure()
# plt.plot(hist['Open'])
plt.plot(hist['Close'])
plt.xlabel('Date')
plt.ylabel('Price [$]')
plt.title('Historical Close Prices')
plt.show()


close_prc = hist['Close']
time_len = len(close_prc)

diff_cls = np.zeros((1, time_len))
diff_cls = diff_cls[0]

#  Closing Price Change
for i in range(0, time_len - 1):
    diff_cls[i] = close_prc[i + 1] - close_prc[i]

avg_growth_rate = mean(diff_cls)
print(avg_growth_rate)

plt.figure()
plt.plot(hist.index, diff_cls)
plt.axhline(avg_growth_rate, color='r')
plt.xlabel('Date')
plt.ylabel('Price Change [$]')
plt.title('Change in Closing Price')
plt.show()


net_income = aapl.info['netIncomeToCommon']

trail_eps = aapl.info['trailingEps']

price_to_book = aapl.info['priceToBook']

reg_prc = aapl.info['regularMarketPrice']

p_e = reg_prc/trail_eps

peg = aapl.info['pegRatio']     # Price-to-Earnings/ EPS Growth Rate

print(net_income, trail_eps, price_to_book, reg_prc, p_e, peg)

print(p_e/peg)  # EPS Growth Rate (in percentages)

print(p_e/price_to_book)    # Should be around 1?  (Book value/Earnings)

# Plot Dividends
plt.figure()
plt.plot(aapl.dividends)
plt.xlabel('Date')
plt.ylabel('Dividend Yield [Percentage]')
plt.title('Dividend Yield')
plt.show()
