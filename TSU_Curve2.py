import streamlit as st
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

# Corrected Treasury symbols
TREASURY_SYMBOLS = {
    '3M':'DGS3MO',       # 13-week Treasury Bill (approximated)
    '2Y':'DGS2',     # Corrected 2-Year Treasury Note
    '5Y':'DGS5',       # 5-Year Treasury Note
    '10Y':'DGS10',      # 10-Year Treasury Note
    '30Y':'DGS30'       # 30-Year Treasury Bond
}

def fetch_yield_data(start_date=None):
    """
    Fetch yield curve data from FRED (Federal Reserve Economic Data).
    Returns a DataFrame with maturities as columns in % yields.
    """
    if start_date is None:
        start_date = datetime.today() - datetime.timedelta(days=365)

    end_date = datetime.today()
    series = []
    for maturity, symbol in TREASURY_SYMBOLS.items():
        try:
            df = web.DataReader(symbol, "fred", start=start_date, end=end_date)
            # df has one column named by the symbol (e.g., 'DGS10')
            s = df[symbol].rename(maturity)  # rename to '10Y', '2Y', etc.
            series.append(s)
        except Exception as e:
            print(f"Could not fetch data for {maturity} ({symbol}): {e}")
            series.append(pd.Series(name=maturity, dtype="float64"))

    # Align on the date index and combine
    df = pd.concat(series, axis=1)

    # Fill small gaps (FRED has occasional missing days/holidays)
    df = df.ffill().bfill()

    return df

def plot_yield_curve(df, selected_date):
    available_dates = df.index
    closest_date = available_dates[available_dates <= pd.Timestamp(selected_date)].max()

    if pd.isnull(closest_date):
        st.error("No data available for the selected date range")
        return None

    maturities = {'3M': 0.25, '2Y': 2, '5Y': 5, '10Y': 10, '30Y': 30}

    fig = go.Figure()

    current_yields = df.loc[closest_date]
    valid_maturities = []
    valid_yields = []

    for mat, year in maturities.items():
        if pd.notna(current_yields[mat]):
            valid_maturities.append(year)
            valid_yields.append(current_yields[mat])

    fig.add_trace(go.Scatter(
        x=valid_maturities,
        y=valid_yields,
        name=closest_date.strftime('%Y-%m-%d'),
        line=dict(color='blue', width=3)
    ))

    month_ago = closest_date - pd.DateOffset(months=1)
    month_ago = available_dates[available_dates <= month_ago].max()

    if pd.notna(month_ago) and month_ago in df.index:
        historical_yields = df.loc[month_ago]
        valid_hist_maturities = []
        valid_hist_yields = []

        for mat, year in maturities.items():
            if pd.notna(historical_yields[mat]):
                valid_hist_maturities.append(year)
                valid_hist_yields.append(historical_yields[mat])

        fig.add_trace(go.Scatter(
            x=valid_hist_maturities,
            y=valid_hist_yields,
            name='1 Month Ago',
            line=dict(color='gray', dash='dash')
        ))

    fig.update_layout(
        title='U.S. Treasury Yield Curve',
        xaxis_title='Years to Maturity',
        yaxis_title='Yield (%)',
        template='plotly_dark',
        showlegend=True
    )

    return fig

def calculate_spreads(df):
    spreads = pd.DataFrame()
    if '2Y' in df.columns and '10Y' in df.columns:
        spreads['2s10s'] = df['10Y'] - df['2Y']
    if '3M' in df.columns and '10Y' in df.columns:
        spreads['3m10y'] = df['10Y'] - df['3M']
    if '5Y' in df.columns and '30Y' in df.columns:
        spreads['5s30s'] = df['30Y'] - df['5Y']
    return spreads

def plot_spreads(spreads):
    fig = go.Figure()
    for col in spreads.columns:
        if not spreads[col].isna().all():
            fig.add_trace(go.Scatter(
                x=spreads.index,
                y=spreads[col],
                name=col,
                line=dict(width=2)
            ))
    fig.add_hline(y=0, line_color='red', line_dash='dash')
    fig.update_layout(
        title='Treasury Yield Spreads',
        xaxis_title='Date',
        yaxis_title='Spread (bps)',
        template='plotly_dark'
    )
    return fig

def calculate_forward_rates(df):
    forwards = pd.DataFrame()
    maturities = {'3M': 0.25, '2Y': 2, '5Y': 5, '10Y': 10, '30Y': 30}
    for short_term, long_term in [('3M', '2Y'), ('2Y', '5Y'), ('5Y', '10Y'), ('10Y', '30Y')]:
        if short_term in df.columns and long_term in df.columns:
            r1 = maturities[short_term]
            r2 = maturities[long_term]
            try:
                forwards[f'{short_term}-{long_term}'] = (
                    ((1 + df[long_term] / 100) ** r2 / (1 + df[short_term] / 100) ** r1) ** (1 / (r2 - r1)) - 1
                ) * 100
            except Exception as e:
                print(f"Could not calculate forward rate for {short_term}-{long_term}: {e}")
    return forwards

def plot_forward_rates(forwards):
    fig = go.Figure()
    for col in forwards.columns:
        if not forwards[col].isna().all():
            fig.add_trace(go.Scatter(
                x=forwards.index,
                y=forwards[col],
                name=col,
                line=dict(width=2)
            ))
    fig.update_layout(
        title='Implied Forward Rates',
        xaxis_title='Date',
        yaxis_title='Rate (%)',
        template='plotly_dark',
        showlegend=True
    )
    return fig

def main():
    st.set_page_config(layout="wide", page_title="Bond Yield Curve Analysis")
    st.title('📈 Bond Yield Curve Analysis Dashboard')

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        default_end_date = datetime.now()
        default_start_date = default_end_date - timedelta(days=365)
        date_range = st.date_input('Select Date Range',
                                   [default_start_date, default_end_date],
                                   max_value=default_end_date)

    if len(date_range) == 2:
        start_date, end_date = date_range

        with st.spinner('Fetching yield data...'):
            df = fetch_yield_data(start_date)

        if not df.empty:
            fig = plot_yield_curve(df, end_date)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

            spreads = calculate_spreads(df)
            if not spreads.empty:
                st.subheader('Yield Spread Analysis')
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(plot_spreads(spreads), use_container_width=True)
                with col2:
                    current_spreads = spreads.iloc[-1]
                    for spread, value in current_spreads.items():
                        if pd.notna(value):
                            st.metric(f'{spread} Spread', f'{value:.2f} bps')

            forwards = calculate_forward_rates(df)
            if not forwards.empty:
                st.subheader('Forward Rate Analysis')
                st.plotly_chart(plot_forward_rates(forwards), use_container_width=True)

            if '2s10s' in spreads.columns:
                current_2s10s = spreads['2s10s'].iloc[-1]
                if pd.notna(current_2s10s):
                    st.subheader('Recession Probability Indicator')
                    recession_prob = 1 / (1 + np.exp(-(-current_2s10s / 100)))
                    st.metric('Recession Probability (Based on 2s10s spread)', f'{recession_prob:.1%}')

            st.subheader('Yield Curve Statistics')
            col1, col2 = st.columns(2)
            with col1:
                st.write('Summary Statistics')
                st.dataframe(df.describe())
            with col2:
                st.write('Correlation Matrix')
                st.dataframe(df.corr())
    else:
        st.error("Please select both start and end dates.")

if __name__ == "__main__":
    main()
