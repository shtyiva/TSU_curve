import streamlit as st
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Corrected Treasury symbols
TREASURY_SYMBOLS = {
    "1M": "DGS1MO",  # 1 Month
    "3M": "DGS3MO",  # 3 Months
    "6M": "DGS6MO",  # 6 Months
    "1Y": "DGS1",  # 1 Year
    "2Y": "DGS2",  # 2 Years
    "3Y": "DGS3",  # 3 Years
    "5Y": "DGS5",  # 5 Years
    "7Y": "DGS7",  # 7 Years
    "10Y": "DGS10",  # 10 Years
    "20Y": "DGS20",  # 20 Years
    "30Y": "DGS30",  # 30 Years
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


def plot_yield_curve(df, selected_date, maturities, one_year_forward_curve=None):
    """
    Plot the spot yield curve and optionally the 1Y forward curve.
    """
    if df.empty:
        st.error("No data available.")
        return None

    # If no forward curve provided, compute it
    if one_year_forward_curve is None:
        one_year_forward_curve = calculate_one_year_forward_curve(
            df=df,
            maturities=maturities,
            max_year=30,
        )

    # --- your existing code from here on ---
    selected_ts = pd.Timestamp(selected_date)
    available_dates = df.index[df.index <= selected_ts]
    if len(available_dates) == 0:
        st.error("No data available for the selected date range")
        return None
    closest_date = available_dates.max()

    # Build spot curve vectors
    current = df.loc[closest_date]
    xs_spot, ys_spot = [], []
    for label, year in maturities.items():
        if (label in df.columns) and pd.notna(current.get(label)):
            xs_spot.append(year)
            ys_spot.append(float(current[label]))

    fig = go.Figure()

    # Current spot curve
    fig.add_trace(go.Scatter(
        x=xs_spot,
        y=ys_spot,
        name=closest_date.strftime('%Y-%m-%d'),
        mode='lines+markers',
        line=dict(color='blue', width=2)
    ))

    # One month ago comparison (unchanged)
    month_ago_candidate = closest_date - pd.DateOffset(months=1)
    prev_dates = df.index[(df.index <= month_ago_candidate)]
    if len(prev_dates) > 0:
        month_ago = prev_dates.max()
        prev = df.loc[month_ago]
        xs_prev, ys_prev = [], []
        for label, year in maturities.items():
            if (label in df.columns) and pd.notna(prev.get(label)):
                xs_prev.append(year)
                ys_prev.append(float(prev[label]))

        if xs_prev:
            fig.add_trace(go.Scatter(
                x=xs_prev,
                y=ys_prev,
                name=f'1 Month Ago ({month_ago.strftime("%Y-%m-%d")})',
                mode='lines+markers',
                line=dict(color='gray', width=2, dash='dash')
            ))

    # Overlay 1Y forward curve
    if (closest_date in one_year_forward_curve.index):
        row = one_year_forward_curve.loc[closest_date]
        xs_fwd, ys_fwd = [], []
        for col in row.index:
            try:
                left, right = col.split('-')  # "0Y-1Y"
                t = int(left.replace('Y', ''))
                xs_fwd.append(t + 0.5)       # midpoint
                ys_fwd.append(float(row[col]))
            except Exception:
                continue

        if xs_fwd:
            fig.add_trace(go.Scatter(
                x=xs_fwd,
                y=ys_fwd,
                name='1Y Forwards (cross-section)',
                mode='lines+markers',
                line=dict(color='red', width=2,dash='dash')
            ))

    fig.update_layout(
        title='US Treasury Yield Curve',
        xaxis_title='Years to Maturity',
        yaxis_title='Yield / Forward (%)',
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
                                                                ((1 + df[long_term] / 100) ** r2 / (
                                                                            1 + df[short_term] / 100) ** r1) ** (
                                                                            1 / (r2 - r1)) - 1
                                                        ) * 100
            except Exception as e:
                print(f"Could not calculate forward rate for {short_term}-{long_term}: {e}")
    return forwards


def calculate_one_year_forward_curve(df: pd.DataFrame, maturities: dict, max_year: int = 30) -> pd.DataFrame:
    """
        Compute 1Y forward rates f(t,t+1) from spot yields (in %) for integer maturities t=0..max_year-1.
        Returns a DataFrame (index = dates, columns like '0Y-1Y', '1Y-2Y', ...), values in %.
    """
    cols_present = [c for c in df.columns if c in maturities]
    if not cols_present:
        return pd.DataFrame(index=df.index)

    maturity_years = np.array([float(maturities[c]) for c in cols_present], dtype=float)
    order = np.argsort(maturity_years)
    maturity_years = maturity_years[order]
    spot_df = df[[cols_present[i] for i in order]].astype(float)

    target_years = np.arange(0, max_year + 1, dtype=float)

    interp_list = []
    for date, row in spot_df.iterrows():
        yi = np.interp(x=target_years, xp=maturity_years, fp=row.values.astype(float))
        interp_list.append(pd.Series(yi, index=target_years, name=date))

    interp_spot = pd.DataFrame(interp_list)
    interp_spot.index = spot_df.index
    interp_spot[0.0] = 0.0
    interp_spot = interp_spot.reindex(sorted(interp_spot.columns), axis=1)

    fwd_cols = []
    fwd_data = {}
    for t in range(0, max_year):
        t0, t1 = float(t), float(t + 1)
        if t0 in interp_spot.columns and t1 in interp_spot.columns:
            z_t = interp_spot[t0] / 100.0
            z_tp1 = interp_spot[t1] / 100.0
            fwd = ((1.0 + z_tp1) ** (t + 1)) / ((1.0 + z_t) ** t) - 1.0
            col_name = f"{t}Y-{t + 1}Y"
            fwd_data[col_name] = fwd * 100.0
            fwd_cols.append(col_name)

    one_year_forward_curve = pd.DataFrame(fwd_data, index=interp_spot.index)[fwd_cols]
    return one_year_forward_curve


def plot_forward_rates(one_year_forward_curve):
    fig = go.Figure()
    for col in one_year_forward_curve.columns:
        if not one_year_forward_curve[col].isna().all():
            fig.add_trace(go.Scatter(
                x=one_year_forward_curve.index,
                y=one_year_forward_curve[col],
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
    st.title('📈 Treasuries Yield Curve Analysis Dashboard')

    col1, col2, col3 = st.columns([1, 2, 3])
    with col2:
        default_end_date = datetime.now()
        default_start_date = default_end_date - timedelta(days=365)
        date_range = st.date_input(
            'Select Date Range',
            [default_start_date, default_end_date],
            max_value=default_end_date
        )

    if len(date_range) == 2:
        start_date, end_date = date_range

        with st.spinner('Fetching yield data...'):
            # Your existing data fetch
            df = fetch_yield_data(start_date)

            # Existing forward rate calc (if you still use it)
            forwards = calculate_forward_rates(df)

            # 👉 Add new 1Y forward curve calc
            maturities = {
                '1M': 1 / 12, '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2,
                '3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30
            }
            one_year_forward_curve = calculate_one_year_forward_curve(
                df=df,
                maturities=maturities,
                max_year=29
            )

        if not df.empty:
            fig = plot_yield_curve(df, end_date, maturities)
            st.plotly_chart(fig, use_container_width=True)

                # Optionally, display data in an expander
            with st.expander("Show 1-Year Forward Curve Data"):
                st.dataframe(one_year_forward_curve.tail(10))

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

            # Forward rate section — no need to recalculate
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
