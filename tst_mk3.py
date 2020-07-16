import yfinance as yf
import numpy as np
from statistics import mean
from dateutil.relativedelta import relativedelta
import plotly.express as px


class PolyVec:

    def __init__(self):
        self.poly_vec = np.array([])
        self.err_vec = np.array([])


class Stock:

    def __init__(self, symbol: str, time_period: str = '5y'):
        self._ticker = yf.Ticker(symbol)
        self._time_period = time_period
        self.__valid_symb_flag: int = 1
        self._FIT_DEG: int = -1

        if not self._ticker.history():
            self.__valid_symb_flag = 0
            raise print('Symbol not found. Please try a different symbol.')

        if self.__valid_symb_flag:
            self._hist = self._ticker.history(self._time_period)
            self._net_income = self._ticker.info['netIncomeToCommon']
            self._trail_eps = self._ticker.info['trailingEps']
            self._price_to_book = self._ticker.info['priceToBook']
            self._reg_price = self._ticker.info['regularMarketPrice']
            self._p_e = self._reg_price / self._trail_eps
            self._peg = self._ticker['pegRatio']    # Price-to-Earnings/ EPS Growth Rate
            self._last_date = self._hist.index[-1]
            self._eps_growth = self._p_e / self._peg  # EPS Growth Rate [percentage]
            self._book_to_earn = self._p_e / self._price_to_book    # Book value/ Earnings
            self._dividends = self._ticker.dividends
            self._data_size = None
            self._time_vector = None

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
        avg_close_growth = mean(diff_close)     # Average Close Price Growth [$]

        return close_price, diff_close, avg_close_growth

    def _close_price_best_fit(self, fit_deg: int):
        """
        Returns Best-Fit Polynomial for Close Price Data
        :param fit_deg: Degree of polynomial to fit
        :return:
            poly: The polynomial representing the best-fit line
            close_err: The error between best-fit and close price data
        """
        self._time_vector = np.linspace(self._data_size, self._data_size)
        poly_coeff = np.polyfit(self._time_vector, self._hist['Close'], fit_deg)
        poly = np.poly1d(poly_coeff)

        # Error Calcs
        close_err = (self._hist['Close'] - poly(self._time_vector))/self._hist['Close']

        return poly, close_err

    def predict_close_price(self, desc_poly, period: int = 30):
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

        result_vec = PolyVec()
        for i in range(1, 11):
            poly, err = self._close_price_best_fit(i)
            result_vec.poly_vec[i-1] = poly
            result_vec.err_vec[i-1] = err

        abs_err_vec = abs(result_vec.err_vec)
        min_error: float = min(abs_err_vec)
        min_idx = (abs_err_vec < min_error).nonzero()
        








