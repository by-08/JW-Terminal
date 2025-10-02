import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
from io import StringIO
import json
import os

# Cache file
CACHE_FILE = 'tickers_cache.json'

@st.cache_data(ttl=86400)  # Cache for 1 day
def get_tickers():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        sp_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        sp_response = requests.get(sp_url, headers=headers)
        sp_table = pd.read_html(StringIO(sp_response.text))[0]
        sp500 = sp_table['Symbol'].tolist()
        
        nasdaq_url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        nasdaq_response = requests.get(nasdaq_url, headers=headers)
        nasdaq_tables = pd.read_html(StringIO(nasdaq_response.text))
        nasdaq100 = []
        for table in nasdaq_tables:
            if 'Ticker' in table.columns:
                nasdaq100 = table['Ticker'].tolist()
                break
        
        all_tickers = list(set(sp500 + nasdaq100))
        all_tickers = [t for t in all_tickers if t not in ['BF.B', 'BRK.B']]
        
        # Save to cache file
        with open(CACHE_FILE, 'w') as f:
            json.dump(all_tickers, f)
        return all_tickers
    except Exception as e:
        st.error(f"Error fetching tickers: {e}")
        return []

def load_cached_tickers():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return []

def process_data(date_str, percentage, filter_mode, min_avg_vol, min_rel_vol, use_cache=False):
    if use_cache:
        tickers = load_cached_tickers()
        if not tickers:
            st.warning("No cached tickers. Run 'New Pull' first.")
            return None
    else:
        with st.spinner("Fetching fresh ticker list..."):
            tickers = get_tickers()
        if not tickers:
            return None
    
    date = datetime.strptime(date_str, '%Y-%m-%d').date()
    end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    start_30d = (date - timedelta(days=45)).strftime('%Y-%m-%d')
    
    def process_mode(mode):
        with st.spinner(f"Downloading data for {mode}..."):
            # Batch download for efficiency
            batch_size = 100
            batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
            all_hist_single = {}
            progress_bar = st.progress(0)
            for i, chunk in enumerate(batches):
                hist_chunk = yf.download(chunk, start=date_str, end=end_date, group_by='ticker', threads=True, progress=False)
                for ticker in chunk:
                    if ticker in hist_chunk.columns.get_level_values(0):
                        all_hist_single[ticker] = hist_chunk[ticker]
                progress_bar.progress((i + 1) / len(batches))
        
        all_data = []
        progress_bar = st.progress(0)
        for i, ticker in enumerate(tickers):
            if ticker in all_hist_single and not all_hist_single[ticker].empty:
                single_data = all_hist_single[ticker]
                o, h, l, c, v = single_data['Open'].iloc[0], single_data['High'].iloc[0], single_data['Low'].iloc[0], single_data['Close'].iloc[0], single_data['Volume'].iloc[0]
                
                range_pct = ((h - l) / c * 100) if c != 0 else 0
                range_val = h - l
                if range_val == 0:
                    close_pct, signal = 0, 'No'
                else:
                    if mode == 'Bullish':
                        close_pct = ((h - c) / range_val * 100)
                        percentage_val = percentage / 100
                        signal = 'Yes' if (o > h - range_val * percentage_val) and (c > h - range_val * percentage_val) else 'No'
                    else:  # Bearish
                        close_pct = ((c - l) / range_val * 100)
                        percentage_val = percentage / 100
                        signal = 'Yes' if (o < l + range_val * percentage_val) and (c < l + range_val * percentage_val) else 'No'
                
                all_data.append({
                    'Ticker': ticker, 'Open': round(o, 2), 'High': round(h, 2), 'Low': round(l, 2),
                    'Close': round(c, 2), 'Volume': int(v), 'Range %': round(range_pct, 2),
                    'Close %': round(close_pct, 2), 'Signal': signal, 'JW Mode': mode
                })
            progress_bar.progress((i + 1) / len(tickers))
        
        df_mode = pd.DataFrame(all_data)
        df_mode = df_mode[df_mode['Signal'] == 'Yes']
        if df_mode.empty:
            return None
        
        yes_tickers = df_mode['Ticker'].tolist()
        with st.spinner(f"Fetching 30D history for {len(yes_tickers)} stocks ({mode})..."):
            hist_30d = yf.download(yes_tickers, start=start_30d, end=end_date, group_by='ticker', threads=True, progress=False)
        
        for idx, row in df_mode.iterrows():
            ticker = row['Ticker']
            if ticker in hist_30d.columns.get_level_values(0) and not hist_30d[ticker].empty:
                full_data = hist_30d[ticker]
                v = row['Volume']
                volumes = full_data['Volume'].dropna()
                single_idx = pd.to_datetime(date_str)
                volumes_before = volumes[volumes.index < single_idx]
                avg_vol = volumes_before.tail(30).mean() if len(volumes_before) >= 30 else (volumes_before.mean() if len(volumes_before) > 0 else 0)
                rel_vol = v / avg_vol if avg_vol > 0 else 0
                df_mode.at[idx, '30D Avg Vol'] = int(round(avg_vol, 0)) if avg_vol > 0 else 0
                df_mode.at[idx, 'Relative Vol'] = round(rel_vol, 2)
        
        df_mode = df_mode[df_mode['30D Avg Vol'] > min_avg_vol * 1000000]
        df_mode = df_mode[df_mode['Relative Vol'] > min_rel_vol]
        if df_mode.empty:
            return None
        return df_mode
    
    if filter_mode == 'All':
        df_bull = process_mode('Bullish')
        df_bear = process_mode('Bearish')
        if df_bull is None and df_bear is None:
            st.warning("No stocks match the criteria.")
            return None
        df = pd.concat([df_bull, df_bear]) if df_bull is not None and df_bear is not None else (df_bull or df_bear)
    else:
        df = process_mode(filter_mode)
        if df is None:
            st.warning("No stocks match the criteria.")
            return None
    
    # Compute Strength
    def range_score(x):
        if x <= 0.5: return 0
        elif x <= 1.5: return 5 * (x - 0.5) / 1.0
        elif x <= 3: return 5 + 2.5 * (x - 1.5) / 1.5
        elif x <= 5: return 7.5 + 2.5 * (x - 3) / 2.0
        else: return 10

    def rel_vol_score(x):
        if x <= 0.5: return 0
        elif x <= 0.8: return 2 * (x - 0.5) / 0.3
        elif x <= 1.0: return 2 + 2 * (x - 0.8) / 0.2
        elif x <= 1.25: return 4 + 2 * (x - 1.0) / 0.25
        elif x <= 1.5: return 6 + 1.5 * (x - 1.25) / 0.25
        elif x <= 2.5: return 7.5 + 2.5 * (x - 1.5) / 1.0
        else: return 10

    df['Range Score'] = df['Range %'].apply(lambda x: min(10, max(0, range_score(x))))
    df['Rel Vol Score'] = df['Relative Vol'].apply(lambda x: min(10, max(0, rel_vol_score(x))))
    df['Close Score'] = 10 * (1 - (df['Close %'] / percentage)).clip(0, 10)
    df['Strength'] = round((df['Range Score'] + df['Close Score'] + df['Rel Vol Score']) / 3, 1)
    return df.sort_values('Strength', ascending=False)

# Streamlit UI
st.set_page_config(layout="wide", page_title="John Wick Terminal")
st.markdown("""
    <style>
    .main {background-color: black;}
    .stApp {background-color: black;}
    .stTextInput > div > div > input {background-color: black; color: lime;}
    .stNumberInput > div > div > input {background-color: black; color: lime;}
    .stSelectbox > div > div > select {background-color: black; color: lime;}
    .stDataFrame {background-color: black; color: lime;}
    </style>
""", unsafe_allow_html=True)

st.title("🔫 JOHN WICK TERMINAL")
st.markdown("*Si vis pacem, para bellum.*", unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.header("Parameters")
    date = st.date_input("Date", value=datetime.now())
    date_str = date.strftime('%Y-%m-%d')
    percentage = st.number_input("JW %", value=20.0, min_value=0.0)
    min_avg_vol = st.number_input("Min Avg Vol (M)", value=5.0, min_value=0.0)
    min_rel_vol = st.number_input("Min Rel Vol", value=0.9, min_value=0.0)
    filter_mode = st.selectbox("JW Mode", ['All', 'Bullish', 'Bearish'])
    minimalist = st.checkbox("Minimalist View")

# Main content
col1, col2 = st.columns(2)
with col1:
    if st.button("New Pull", type="primary"):
        st.session_state.df = process_data(date_str, percentage, filter_mode, min_avg_vol, min_rel_vol, use_cache=False)
with col2:
    if st.button("Fetch Data"):
        st.session_state.df = process_data(date_str, percentage, filter_mode, min_avg_vol, min_rel_vol, use_cache=True)

if 'df' in st.session_state and st.session_state.df is not None and not st.session_state.df.empty:
    df = st.session_state.df
    
    # Filters (simulate table filter)
    table_filter = st.selectbox("Table Filter", ['All', 'Bullish', 'Bearish'], key='table_filter')
    if table_filter != 'All':
        df = df[df['JW Mode'] == table_filter]
    
    # Columns
    if minimalist:
        display_cols = ['Ticker', 'Close', 'Volume', 'Relative Vol', 'Range %', 'Close %', 'JW Mode', 'Strength']
    else:
        display_cols = ['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', '30D Avg Vol', 'Relative Vol', 'Range %', 'Close %', 'JW Mode', 'Signal', 'Strength']
    
    # Format columns
    df_display = df[display_cols].copy()
    df_display['Volume'] = df_display['Volume'].apply(lambda x: f"{x/1000000:.1f} M")
    if '30D Avg Vol' in df_display:
        df_display['30D Avg Vol'] = df_display['30D Avg Vol'].apply(lambda x: f"{x/1000000:.1f} M")
    
    st.dataframe(df_display, use_container_width=True, height=500)
    
    # Export
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Export CSV", csv, "john_wick_data.csv", "text/csv")
else:
    st.info("Click a button to fetch data.")