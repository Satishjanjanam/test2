import os
from glob import glob 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from pandas.tseries.offsets import *
from datetime import datetime, timedelta
import gc
#import datetime

pd.set_option('display.max_columns', 55)
pd.set_option("display.width", 185)
pd.set_option("display.max_rows",200)
pd.set_option("display.max_colwidth",100)
pd.set_option('max_columns', 100)
#plt.style.use('bmh')

# Plotting
import plotly.io as pio
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
import sys
# Set notebook mode to work in offline
pyo.init_notebook_mode()
pio.templates.default = "plotly_dark"
#-------------------------------------------------------------------------------------------------------------------------

os.chdir("set input directory here if required")

###################################################################################################################################################################
def plot_anamoly(metric, lower_band, upper_band, df, sma, buyers, sellers):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lower_band.index, y=lower_band['lower'], name='Lower Band', line_color='rgba(173,204,255,0.2)'))
    
    fig.add_trace(go.Scatter(x=upper_band.index, y=upper_band['upper'], name='Upper Band', fill='tonexty', fillcolor='rgba(173,204,255,0.2)', line_color='rgba(173,204,255,0.2)'))

    fig.add_trace(go.Scatter(x=df.index, y=df[metric], name=metric, line_color='#636EFA'))

    fig.add_trace(go.Scatter(x=sma.index, y=sma[metric], name='SMA', line_color='#FECB52'))

    fig.add_trace(go.Scatter(x=buyers.index, y=buyers[metric], name='-- Anamoly', mode='markers', marker=dict(color='#00CC96', size=10,)))

    fig.add_trace(go.Scatter(x=sellers.index, y=sellers[metric], name='+ Anamoly', mode='markers', marker=dict(color='#EF553B', size=10,),))
    fig.update_layout(title=f"Metric: {metric}",)
    #fig.show()
    plot(fig)


#######################################################################################################################################################################
# latest 12 weeks

def anamoly_model(df, metric, plot=False):


    gp = df.groupby('week_start_date')[[metric]].mean().reset_index(drop=False)
#    df = df[df['geolocation_state']==geo_state][[metric, 'week_start_date']].reset_index(drop=True)
    #df.rename(columns = {'unique_visitors':'Close'}, inplace=True)
    sma = gp.rolling(window=20).mean().dropna()
    rstd = gp.rolling(window=20).std().dropna()
    
    upper_band = sma + 2 * rstd
    lower_band = sma - 2 * rstd
    
    upper_band = upper_band.rename(columns={metric: 'upper'})
    lower_band = lower_band.rename(columns={metric: 'lower'})
    bb = gp.join(upper_band).join(lower_band)
    bb = bb.dropna()
    
    buyers = bb[bb[metric] <= bb['lower']]
    sellers = bb[bb[metric] >= bb['upper']]

#    end_date = gp['week_start_date'].max()
#    start_date = end_date - timedelta(weeks = 12)
#    gp[gp['week_start_date']>=start_date]

    gp1 = gp.copy()
    gp1['rnk'] = gp1["week_start_date"].rank(ascending=False)
    gp1 = gp1[gp1['rnk']<=12][['week_start_date', metric]].reset_index(drop=True)
    gp1.rename(columns = {metric:'value'}, inplace=True)
    gp1['week_start_date'] = gp1['week_start_date'].astype(str)
    gp1['value'] = gp1['value'].apply(lambda x: round(x,3))

    anamoly_dates = buyers['week_start_date'].tolist() + sellers['week_start_date'].tolist() 

    gp1['is_anomaly'] = np.where(gp1['week_start_date'].isin(anamoly_dates), True, False)

    if plot: plot_anamoly(metric, lower_band, upper_band, gp, sma, buyers, sellers)

    out = gp1.to_dict('records')

    return out

###################################################################################################################################


metric_list = [
 'unique_visitors',
 'visits',
 'page_views',
 'carts',
 'cart_additions',
 'checkouts',
 'orders',
 'units',
 'revenue',
 'aov',
 'cart_open_to_checkout_initiation',
 'checkout_initiation_to_orders',
 'upt',
 'visit_depth',
 'avg_session_duration',
 'abandoned',
 'abandonment_rate',
 'buyer_conversion',
 'abandoned_revenue',
 'page_views_per_visitor',
 'click_throughs',
 'internal_searches',
 'shop_visits',
 'visit_duration']

#zerodf = (((df == 0).astype(int).sum(axis=0).sort_values(ascending=False)/df.shape[0])*100).reset_index(drop=False)
#zerodf.columns = ['zero_column', 'zero_percent']
#zero_columns = zerodf[zerodf['zero_percent']>75]['zero_column'].tolist()
zero_columns = ['shop_visits', 'visit_duration', 'buyer_conversion']
metric_list = [x for x in metric_list if x not in zero_columns]

#remove limit and set None for complete output

def anamoly_model_f(df, limit=2):
    final_output = {}
    for metric in metric_list[:limit]:
        anamoly_list = anamoly_model(df.copy(), metric, plot=False)
        final_output[metric] = anamoly_list
    return final_output

df = pd.read_csv('weekly_geo_state_data.csv')
df['week_start_date'] = pd.to_datetime(df['week_start_date'])
df['week_start_date'] = pd.to_datetime(df['week_start_date'], format = '%Y-%m-%d')

final_output = anamoly_model_f(df)
#print(final_output)
#print(final_output.keys())
#---------------------------------------------------------------------------------------------------------------------------------------------
