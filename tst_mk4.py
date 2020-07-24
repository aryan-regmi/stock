import yfinance as yf
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import altair as alt
# alt.renderers.enable('altair_viewer')

# -------------------------------------------------------------------------------------------------------------------- #

# Testing Altair Functions
symbol = 'AAPL'
time_period = '5y'

tick_data = yf.Ticker(symbol)
hist_data = tick_data.history(time_period).reset_index()

fig1 = alt.Chart(hist_data).mark_line().encode(
    x=alt.X('Date', axis=alt.Axis(title='Date')),
    y=alt.Y('Close:Q', axis=alt.Axis(title='Price [$]')),
    tooltip=['Close', 'Date']
).properties(
    width=800,
    height=400,
    title='Historical Close Price Data'
).interactive()

fig1.save('plot_file.html')
