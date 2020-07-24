import yfinance as yf
import numpy as np
from statistics import mean
from dateutil.relativedelta import relativedelta
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime

# import plotly.io as pio
# print(pio.renderers)
# pio.renderers.default = 'png'


# TODO: Replace PolyVec implementation with numpy array
class PolyVec:

    def __init__(self, size: int):
        self.poly_vec = []
        self.err_vec = np.zeros([1, size])[0]


class Stock:

    def __init__(self, symbol: str, time_period: str = '5y'):
        self._ticker = yf.Ticker(symbol)
        self._time_period = time_period
        self.__valid_symb_flag: int = 1
        self._FIT_DEG: int = -1

        if self._ticker.history() is None:
            self.__valid_symb_flag = 0
            raise print('Symbol not found. Please try a different symbol.')

        if self.__valid_symb_flag:
            # Initialize Variables
            self._hist = self._ticker.history(self._time_period)
            self._net_income = self._ticker.info['netIncomeToCommon']
            self._trail_eps = self._ticker.info['trailingEps']
            self._price_to_book = self._ticker.info['priceToBook']
            self._reg_price = self._ticker.info['regularMarketPrice']
            self._p_e = self._reg_price / self._trail_eps
            self._peg = self._ticker.info['pegRatio']  # Price-to-Earnings/ EPS Growth Rate
            self._last_date = self._hist.index[-1]
            self._eps_growth = self._p_e / self._peg  # EPS Growth Rate [percentage]
            self._book_to_earn = self._p_e / self._price_to_book  # Book value/ Earnings
            self._dividends = self._ticker.dividends
            self._data_size = None
            self._time_vector = None
            self._close_price, self._diff_close, self._avg_close_growth = None, None, None
            self._poly, self._close_error = None, None

            # Call Necessary Functions
            self._close_price, self._diff_close, self._avg_close_growth = self._close_data_analysis()
            self._find_best_est()
            self._poly, self._close_error = self._close_price_best_fit(self._FIT_DEG)

    def _close_data_analysis(self):
        """
        Gets Close Price data and the close price growth rate
        :return:
            close_price: Historical Close Price
            diff_close: Change in Close Price
            avg_close_growth: Average Close Price Growth
        """
        close_price = self._hist['Close']
        time_len = len(close_price)
        self._data_size = time_len

        # Change in Close Price
        diff_close = np.diff(close_price, n=1, axis=0)  # [$]
        avg_close_growth = mean(diff_close)  # Average Close Price Growth [$]

        return close_price, diff_close, avg_close_growth

    def _close_price_best_fit(self, fit_deg: int):
        """
        Returns Best-Fit Polynomial for Close Price Data
        :param fit_deg: Degree of polynomial to fit
        :return:
            poly: The polynomial representing the best-fit line
            close_err: The error between best-fit and close price data
        """
        self._time_vector = np.linspace(0, self._data_size, self._data_size)
        poly_coeff = np.polyfit(self._time_vector, self._hist['Close'], fit_deg)
        poly = np.poly1d(poly_coeff)

        # Error Calcs
        close_err = (self._hist['Close'] - poly(self._time_vector)) / self._hist['Close']

        return poly, close_err

    def _predict_close_price(self, desc_poly, period: int = 30):
        """
        Returns predicted close price for specified period/date
        :param desc_poly: Polynomial describing the best-fit close price line
        :param period: Days to predict for [Default to 30 days]
        :return:
            next_date: Date corresponding to 'next_price'
            next_price: Predicted Close Price [$]
        """
        next_price = desc_poly(max(self._time_vector) + period)
        next_date = self._last_date + relativedelta(days=period)
        return next_date, next_price

    def _find_best_est(self):
        """
        Finds polynomial with smallest error and sets _FIT_DEG to that degree
        """
        result_vec = PolyVec(10)
        for i in range(1, 11):
            _, err = self._close_price_best_fit(i)
            max_err = max(err)
            result_vec.err_vec[i - 1] = max_err

        print(f"SUM: {result_vec.err_vec.cumsum(axis=0)}")
        abs_err_vec = abs(result_vec.err_vec)
        min_error: float = min(abs_err_vec)
        min_idx = (abs_err_vec <= min_error).nonzero()
        self._FIT_DEG = min_idx[0]

    def plot_close(self):
        """
        Plots historical Close Price along with various analysis plots
        """
        plot_titles = ('<b>Historical Close Prices</b>', '<b>Close Price Growth</b>',
                       '<b>Best-Fit Line Error</b>', '<b>Dividend Yield</b>')
        fig = make_subplots(rows=2, cols=2, subplot_titles=plot_titles)

        # Historical Close Price
        fig.add_trace(
            go.Scatter(x=self._hist.index, y=self._close_price,
                       mode='lines',
                       name='Close Price',
                       hovertemplate='<br><b>Price</b>: $%{y:.2f}<br>',
                       showlegend=False
                       ),
            row=1, col=1
        )

        # Best-Fit Line
        fig.add_trace(
            go.Scatter(x=self._hist.index, y=self._poly(self._time_vector),
                       mode='lines',
                       name='Best-Fit',
                       hovertemplate='<br><b>Price</b>: $%{y:.2f}<br>',
                       showlegend=False,
                       ),
            row=1, col=1
        )

        # Error Between Best-Fit and Historical Data
        fig.add_trace(
            go.Scatter(x=self._hist.index, y=self._close_error * 100,
                       mode='lines',
                       name='Close Price Error',
                       hovertemplate=
                       '<br><b>% Error</b>: %{y:.2f}<br>' +
                       '<b>Date</b>: %{x}',
                       showlegend=False
                       ),
            row=2, col=1
        )

        # Change in Close Price
        fig.add_trace(
            go.Scatter(x=self._hist.index, y=self._diff_close,
                       mode='lines',
                       name='Close Price Growth',
                       hovertemplate=
                       '<br><b>Price</b>: $%{y:.2f}<br>' +
                       '<b>Date</b>: %{x}',
                       showlegend=False
                       ),
            row=1, col=2
        )

        # Dividends
        fig.add_trace(
            go.Scatter(x=self._dividends.index, y=self._dividends,
                       mode='lines',
                       name='Dividends',
                       hovertemplate=
                       '<br><b>Dividends</b>: %{y:.2f}<br>' +
                       '<b>Date</b>: %{x}',
                       showlegend=False
                       ),
            row=2, col=2
        )

        # Update Axes Titles
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Price [$]", row=1, col=1)
        fig.update_yaxes(title_text="Percent Error", row=2, col=1)
        fig.update_yaxes(title_text="Price [$]", row=1, col=2)
        fig.update_yaxes(title_text="Dividend Yield [%]", row=2, col=2)

        # Change Background Color
        fig.update_layout(plot_bgcolor='rgba(75, 0, 0, 0.20)', hovermode='x unified')
        fig.show()

    def plot_predicted_price(self, period=30):
        """
        Plots predicted price using best-fit line
        :param period: Days to predict price for [Default to 30 days]
        """
        next_date, next_price = self._predict_close_price(self._poly, period)

        # Make Plot
        plot_titles = '<b>Predicted Close Price</b>'
        fig = make_subplots(rows=1, cols=1)

        # Historical Close Price
        fig.add_trace(
            go.Scatter(x=self._hist.index, y=self._close_price,
                       mode='lines',
                       name='Close Price',
                       hovertemplate='<br><b>Price</b>: $%{y:.2f}<br>',
                       showlegend=False
                       ),
            row=1, col=1
        )

        my_date = pd.DataFrame([next_date])
        my_prc = pd.DataFrame([next_price])

        # Predicted Point
        fig.add_trace(
            go.Scatter(x=my_date[0], y=my_prc[0],
                       name='Predicted Point',
                       hovertemplate='<br><b>Price</b>: $%{y:.2f}<br>',
                       showlegend=False,
                       marker=dict(
                           color='red',
                           size=10)
                       ),
            row=1, col=1
        )

        # Update Axes titles
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Price [$]", row=1, col=1)

        # Change Background Color
        fig.update_layout(plot_bgcolor='rgba(75, 0, 0, 0.20)',
                          hovermode='x unified',
                          title_text=plot_titles,
                          title_x=0.5,
                          title_font_size=30)
        # fig.show()


v1 = Stock('AAPL')
# v1.plot_close()
v1.plot_predicted_price()
