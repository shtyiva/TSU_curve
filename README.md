Parser functionality: 
Yield Data Retrieval, Forward Curve calculation, Spreads, Correlations and Recession probability indicator
Short overview:
U.S. Treasury yield curve reflects market expectations for interest rates, inflation and growth. The 2s10s spread is a widely used recession indicator, often signalling downturns 12–18 months ahead, while the 5s30s spread reflects long-term economic expectations.
Forward rates are exceptionally important for understanding the structure of the yield curve and for identifying relatively cheap or rich segments along the curve. The forward curve serves as a benchmark for market expectations, reflecting both hopes and fears. Implied one-year forward spot rates represent the future spot rates that equalize holding-period returns across government bonds over the coming years. Forward rates indicate how much yields on longer-term bonds must change to offset returns available on shorter-term maturities. Forwards tell how much yield along term bond need to change to offset yield over short term. 
Generally, positive carry (causing against rising rates) / negative carry (future flattening of spot curve) offset negative carry.
(HINT) The Forward curve is benchmark for hopes & fears.
Assuming annual compounding, the one-year forward rate is easily computed:
 
One-year forward rates measure reward for lengthening the maturity of investment by one year, while spot rates measure an investment average reward from today to maturity n.

Parser Solution Idea & Technik: 
Recently I found out very interesting Python library : pandas_datareader.data (imported as web) is a used to retrieve financial and economic time series data from online sources such as the Federal Reserve Economic Data (FRED) directly into pandas DataFrames.
It allows to create powerful and absolutely free parser for Fixed income portfolio managers and traders. The parser built on Streamlit library, which allows user to run tool directly from Web browser and make necessary inputs in windows designed.  Streamlit is a Python framework for quickly building interactive web applications, especially for data analysis and visualization.




Following imports are necessary to run the code
import streamlit as st
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
