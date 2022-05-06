"""anomaly service"""
#from pandas.tseries.offsets import *
import numpy as np


class AnomalyDetector:
    """AnomalyDetector class"""

    def __init__(self, data, **kwargs):
        self.data = data
        self.kwargs = kwargs

    def detect_anomaly(self, metric_col_name,
                       time_series_col_name,
                       past_n=8):
        """function to detect anomaly"""

        time_series_data = self.data[[time_series_col_name, metric_col_name]]
        sma = time_series_data.rolling(window=10).mean().dropna()
        rstd = time_series_data.rolling(window=10).std().dropna()
        upper_band = sma + 1.0 * rstd
        lower_band = sma - 1.0 * rstd
        sma_band = sma
        # find upper and lower bound
        upper_band = upper_band.rename(
            columns={metric_col_name: 'upper_bound'})
        lower_band = lower_band.rename(
            columns={metric_col_name: 'lower_bound'})
        sma_band = sma_band.rename(columns={metric_col_name: 'sma_bound'})
        # find bollinger
        bollinger = time_series_data.join(
            upper_band).join(lower_band).join(sma_band)
        bollinger = bollinger.dropna()
        # find lower and upper anomaly
        lower_anomaly = bollinger[bollinger[metric_col_name]
                                  < bollinger['lower_bound']]
        upper_anomaly = bollinger[bollinger[metric_col_name]
                                  > bollinger['upper_bound']]
        results = bollinger.copy()
        results['rnk'] = results[time_series_col_name].rank(ascending=False)
        results = results[results['rnk'] <= past_n][
            [time_series_col_name, metric_col_name, 'upper_bound',
             'lower_bound', 'sma_bound']].reset_index(drop=True)

        results.rename(columns={metric_col_name: 'value'}, inplace=True)
        results[time_series_col_name] = results[time_series_col_name].astype(
            str)
        results['value'] = results['value'].apply(lambda x: round(x, 3))
        results['upper_bound'] = results['upper_bound'].apply(
            lambda x: round(x, 3))
        results['lower_bound'] = results['lower_bound'].apply(
            lambda x: round(x, 3))
        results['sma_bound'] = results['sma_bound'].apply(
            lambda x: round(x, 3))

        anamoly_dates = lower_anomaly[time_series_col_name].tolist() + \
            upper_anomaly[time_series_col_name].tolist()
        results['is_anomaly'] = np.where(
            results[time_series_col_name].isin(anamoly_dates), True, False)

        results = results.to_dict('records')

        return results
