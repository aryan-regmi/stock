import yfinance as yf
import matplotlib.pyplot as plt


# Get Data
aapl = yf.Ticker('AAPL')

# Historical Data
time_period = 'max'
hist = aapl.history(period=time_period)

# Plot Hist Data
plt.figure(1)
# plt.plot(hist['Open'])
plt.plot(hist['Close'])
plt.xlabel('Date')
plt.ylabel('Price [$]')
plt.title('Historical Close Prices')
plt.show()

# Diff btwn open and close
open_prc = hist['Open']
close_prc = hist['Close']
sell_diff = open_prc - close_prc

# Diff btwn high and low
high_prc = hist['High']
low_prc = hist['Low']
prc_diff = high_prc - low_prc

# plt.figure(4)
# plt.plot(prc_diff)
# plt.plot(sell_diff)
# plt.axhline(y=0, xmin=0, xmax=1, color='k')
# plt.legend(('High vs Low', 'Open vs Close'))
# plt.show()

print(hist.tail(5))

plt.figure(5)
plt.plot(aapl.dividends)
plt.show()


# TODO: find average growth rate for closing prices

# TODO: analyze other metrics (net income, volume, etc)

# TODO: dividends vs eps

# TODO: monte carlo sims on expected closing prices
