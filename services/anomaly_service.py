import string
import random
import numpy as np
from pandas.tseries.offsets import *

# Plotting
import plotly.graph_objects as go
import plotly.graph_objs as go


class AnomalyDetector(object):

    def __init__(self, data, past_n=8, **kwargs):
        self.data = data
        self.past_n= past_n

    def detect_anomaly(self, metric, plot=True):

        gp = self.data[[metric, 'week_start_date']]
        sma = gp.rolling(window=10).mean().dropna()
        rstd = gp.rolling(window=10).std().dropna()
        
        upper_band = sma + 1 * rstd
        lower_band = sma - 1 * rstd
        avg_band = sma
        
        upper_band = upper_band.rename(columns={metric: 'upper_bound'})
        lower_band = lower_band.rename(columns={metric: 'lower_bound'})
        avg_band = avg_band.rename(columns={metric: 'avg_bound'})
        bb = gp.join(upper_band).join(lower_band).join(avg_band)
        bb = bb.dropna()
        
        buyers = bb[bb[metric] < bb['lower_bound']]
        sellers = bb[bb[metric] > bb['upper_bound']]

        gp1 = bb.copy()
        gp1['rnk'] = gp1["week_start_date"].rank(ascending=False)
        
        gp1 = gp1[gp1['rnk'] <= self.past_n][
            ['week_start_date', metric, 'upper_bound', 
             'lower_bound', 'avg_bound']].reset_index(drop=True)
        gp1.rename(columns = {metric:'value'}, inplace=True)
        gp1['week_start_date'] = gp1['week_start_date'].astype(str)
        gp1['value'] = gp1['value'].apply(lambda x: round(x,3))
        gp1['upper_bound'] = gp1['upper_bound'].apply(lambda x: round(x,3))
        gp1['lower_bound'] = gp1['lower_bound'].apply(lambda x: round(x,3))
        gp1['avg_bound'] = gp1['avg_bound'].apply(lambda x: round(x,3))

        anamoly_dates = buyers['week_start_date'].tolist() + sellers['week_start_date'].tolist() 
        gp1['is_anomaly'] = np.where(gp1['week_start_date'].isin(anamoly_dates), True, False)

        if plot: self.plot_anomaly(metric, lower_band, upper_band, gp, sma, buyers, sellers)

        results = gp1.to_dict('records')

        return results
    

    def plot_anomaly(self, metric, lower_band, upper_band, df, sma, buyers, sellers):

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lower_band.index, y=lower_band['lower_bound'], name='Lower Band', line_color='rgba(173,204,255,0.2)'))
        fig.add_trace(go.Scatter(x=upper_band.index, y=upper_band['upper_bound'], name='Upper Band', fill='tonexty', fillcolor='rgba(173,204,255,0.2)', line_color='rgba(173,204,255,0.2)'))
        fig.add_trace(go.Scatter(x=df.index, y=df[metric], name=metric, line_color='#636EFA'))
        fig.add_trace(go.Scatter(x=sma.index, y=sma[metric], name='SMA', line_color='#FECB52'))
        fig.add_trace(go.Scatter(x=buyers.index, y=buyers[metric], name='-- Anamoly', mode='markers', marker=dict(color='#00CC96', size=10,)))
        fig.add_trace(go.Scatter(x=sellers.index, y=sellers[metric], name='+ Anamoly', mode='markers', marker=dict(color='#EF553B', size=10,),))
        fig.update_layout(title=f"Metric: {metric}",)

        name = ''.join(
                random.choice(string.ascii_lowercase) for i in range(8))
        save_path = "static/" + name + ".png"
        fig.write_image(save_path)
