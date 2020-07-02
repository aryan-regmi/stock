import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from dateutil.relativedelta import relativedelta
import pickle
import os.path
import time

# clear = lambda: os.system('clear')
# clear()


# ------------------------------------------------------------------------------------------------------------------- #
# -----------------------------------------------Initialize Variables------------------------------------------------ #


class Stocks:

    def __init__(self, ticker: str, time_period: str):
        self._ticker = ticker
        self._period = time_period
        self.hist, self.net_income, self.trail_eps, self.price_to_book, self.reg_prc, self.p_e, self.peg, \
            self.eps_growth, self.book_to_earn, self.dividends = self.get_data()
        self.last_date = self.hist.index[-1]
        self.data_size = 0
        self.time_vector = None

    def get_data(self):
        fid = self._ticker + '.p'

        # Check if Data Exists
        if not os.path.isfile(fid):
            print("Downloading Stock Data...")
            try:
                # Open File
                f = open(fid, 'wb')

                # Get Ticker Data
                stock_data = yf.Ticker(self._ticker)

                # Get Historical Data
                hist = stock_data.history(period=self._period)

                # Get Other Financial Data
                net_income = stock_data.info['netIncomeToCommon']
                trail_eps = stock_data.info['trailingEps']
                price_to_book = stock_data.info['priceToBook']
                reg_prc = stock_data.info['regularMarketPrice']
                p_e = reg_prc / trail_eps
                peg = stock_data.info['pegRatio']  # Price-to-Earnings/ EPS Growth Rate
                eps_growth = p_e / peg  # EPS Growth Rate [percentage]
                book_to_earn = p_e / price_to_book  # Book value/ Earnings
                dividends = stock_data.dividends

                # Pickle Data and Save
                pickle.dump(hist, f)
                pickle.dump(net_income, f)
                pickle.dump(trail_eps, f)
                pickle.dump(price_to_book, f)
                pickle.dump(reg_prc, f)
                pickle.dump(p_e, f)
                pickle.dump(peg, f)
                pickle.dump(eps_growth, f)
                pickle.dump(book_to_earn, f)
                pickle.dump(dividends, f)

                # Close File
                f.close()

                # Print Status to Console
                print('Data has been saved.')
                # time.sleep(2)
                # clear()

            except:
                print('An error occured while saving the data.')

        # Load Data from Pickle File
        f = open(fid, 'rb')

        hist = pickle.load(f)  # Historical data

        # Get Other Financial Data
        net_income = pickle.load(f)
        trail_eps = pickle.load(f)
        price_to_book = pickle.load(f)
        reg_prc = pickle.load(f)
        p_e = pickle.load(f)
        peg = pickle.load(f)
        eps_growth = pickle.load(f)
        book_to_earn = pickle.load(f)
        dividends = pickle.load(f)

        # Close File
        f.close()

        return hist, net_income, trail_eps, price_to_book, reg_prc, p_e, peg, eps_growth, book_to_earn, dividends

    def close_data_analysis(self):
        # Get Close Data
        close_prc = self.hist['Close']
        time_len = len(close_prc)
        self.data_size = time_len

        # Change in Close Price
        diff_close = np.diff(close_prc, n=1, axis=0)    # [$]
        avg_close_growth = mean(diff_close)     # Average Close Price Growth [$]

        return close_prc, diff_close, avg_close_growth

    def close_prc_best_fit(self, fitting_data, fit: int = 2):
        """
        Returns Best-Fit Polynomial for specified data
        :param fitting_data: Data to fit
        :param fit: The degree to fit to
        :return: The polynomial representing the best-ft line
        """
        time_vector = np.linspace(0, self.data_size, self.data_size)
        self.time_vector = time_vector
        poly_coeff = np.polyfit(time_vector, fitting_data, fit)
        poly = np.poly1d(poly_coeff)

        return poly

    def predict_close_prc(self, desc_poly, period=30):
        """
        Returns predicted close price for specified period
        :param desc_poly: Polynomial describing the best-fit close price line
        :param period: Days to predict for [Default to 30 days]
        :return:
            next_date: Date corresponding to 'next_prc'
            next_prc: Predicted Close Price
        """
        next_prc = desc_poly(max(self.time_vector) + period)
        next_date = self.last_date + relativedelta(days=period)
        return next_date, next_prc

    def plot_data(self):
        # Get Close Price data
        close_prc, diff_close, avg_close_growth = self.close_data_analysis()

        # Get Polynomial for Best-Fit
        poly = self.close_prc_best_fit(close_prc, 5)    # 5th degree polynomial
        # TODO: Find the fit dynamically and update '5'

        plt.figure()

        # Close Price Plot
        plt.subplot(221)
        plt.tight_layout()
        plt.plot(close_prc)     # Close Price
        plt.plot(self.hist.index, poly(self.time_vector))   # Best-Fit
        plt.xlabel('Date')
        plt.ylabel('Price [$]')
        plt.title('Historical Close Prices')
        plt.legend(('Close Price', 'Best-Fit Line'))

        # Errors in Best-Fit Line
        close_err = (close_prc - poly(self.time_vector))/close_prc

        # Error Plot
        plt.subplot(223)
        plt.tight_layout()
        plt.plot(close_err)
        plt.xlabel('Date')
        plt.ylabel('Error')
        plt.title('Best-Fit Line Error')

        # Close Price Change Plot
        plt.subplot(222)
        plt.tight_layout()
        plt.plot(self.hist.index[1:len(close_prc)], diff_close)
        plt.axhline(avg_close_growth, color='r')
        plt.xlabel('Date')
        plt.ylabel('Price Change [$]')
        plt.legend(('Close Price Change', 'Average Close Price Growth'))
        plt.title('Change in Closing Price')

        # Plot Dividends
        plt.subplot(224)
        plt.tight_layout()
        plt.plot(self.dividends)
        plt.xlabel('Date')
        plt.ylabel('Dividend Yield [Percentage]')
        plt.title('Dividend Yield')

        plt.show()

    def plot_predicted_prc(self, period=30):
        # Get Close Price data
        close_prc, _, _ = self.close_data_analysis()

        # Get Polynomial for Best-Fit
        poly = self.close_prc_best_fit(close_prc)    # 5th degree polynomial

        next_date, next_prc = self.predict_close_prc(poly, period)

        # Make Plot
        plt.figure()
        plt.plot(close_prc)
        plt.plot(next_date, next_prc, 'ro')
        plt.axvline(next_date, color='k', ls='--')
        plt.xlabel('Date')
        plt.ylabel('Price [$]')
        plt.legend(('Close Price', f'Predicted Price: ${next_prc: 0.2f}', f'{next_date.date()}'))
        plt.title('Predicted Close Price Data')
        plt.show()

    # TODO: Find avg error/ cumulative sum of error
    # TODO: Show avg monthly price growth in 'Change in Closing Price' plot
    # TODO: Find fair value price from EPS growth and P/E Ratio
    # TODO: Use regression to minimize errors between best-fit (w/ varying degrees) and actual prices


st = Stocks('AAPL', '5y')

st.plot_data()
st.plot_predicted_prc(365)
