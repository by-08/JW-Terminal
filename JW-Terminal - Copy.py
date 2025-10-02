import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
from io import StringIO
import json
import os

# Page config
st.set_page_config(page_title="John Wick Terminal", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for cyberpunk theme
st.markdown("""
<style>
    .main {
        background-color: black;
        color: lime;
        font-family: 'Courier New', monospace;
    }
    .stApp {
        background-color: black;
    }
    .css-1d391kg {
        background-color: black;
    }
    .stSidebar {
        background-color: black;
    }
    [data-testid="stSidebar"] {
        background-color: black;
    }
    .stTextInput > div > div > input {
        background-color: black;
        color: lime;
        border: 1px solid lime;
    }
    .stNumberInput > div > div > input {
        background-color: black;
        color: lime;
        border: 1px solid lime;
    }
    .stSelectbox > div > div > select {
        background-color: black;
        color: lime;
        border: 1px solid lime;
    }
    .stCheckbox > div {
        color: lime;
    }
    .stButton > button {
        background-color: black;
        color: cyan;
        border: 2px solid cyan;
        font-family: 'Courier New', monospace;
        font-size: 10px;
    }
    .stButton > button:hover {
        background-color: #00FF00;
        color: black;
        border: 2px solid #00FF00;
    }
    .stDownloadButton > button {
        background-color: black;
        color: cyan;
        border: 2px solid cyan;
        font-family: 'Courier New', monospace;
        font-size: 10px;
    }
    .stDownloadButton > button:hover {
        background-color: #00FF00;
        color: black;
        border: 2px solid #00FF00;
    }
    .stDataFrame {
        background-color: black;
        color: lime;
        font-family: 'Courier New', monospace;
        font-size: 9px;
    }
    .dataframe {
        background-color: black;
        color: lime;
        font-family: 'Courier New', monospace;
        font-size: 9px;
    }
    .dataframe thead th {
        background-color: darkgreen;
        color: lime;
        font-weight: bold;
        font-size: 10px;
        font-family: 'Courier New', monospace;
    }
    .dataframe tbody td {
        background-color: black;
        color: lime;
        border: none;
    }
    .stTabs {
        background-color: black;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: black;
        border-bottom: 1px solid darkgreen;
    }
    .stTabs [data-baseweb="tab"] {
        color: cyan;
        font-family: 'Courier New', monospace;
        font-size: 10px;
        padding: 8px 12px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: lime;
        background-color: black;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: black;
        background-color: darkgreen;
        border: 1px solid darkgreen;
    }
    .stProgress > div > div > div {
        background-color: lime;
    }
    .stAlert {
        background-color: black;
        color: lime;
        border: 1px solid lime;
    }
    .stInfo {
        background-color: black;
        color: lime;
        border: 1px solid lime;
    }
    .stWarning {
        background-color: black;
        color: #FF6B6B;
        border: 1px solid #FF6B6B;
    }
    .stSuccess {
        background-color: black;
        color: #51CF66;
        border: 1px solid #51CF66;
    }
    .stSelectbox > label {
        color: lime;
    }
    .stTextInput > label {
        color: lime;
    }
    .stNumberInput > label {
        color: lime;
    }
    .input-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
        margin-bottom: 20px;
    }
    .input-item {
        background: rgba(0, 255, 0, 0.1);
        padding: 10px;
        border: 1px solid lime;
        border-radius: 5px;
    }
    .filter-section {
        background: rgba(0, 0, 139, 0.2);
        padding: 15px;
        border: 1px solid cyan;
        border-radius: 5px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Title and subheader
st.markdown("<h1 style='text-align: center; color: lime; font-family: \"Courier New\", monospace; font-size: 16px; font-weight: bold; margin-bottom: 0;'>JOHN WICK TERMINAL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: cyan; font-family: \"Courier New\", monospace; font-size: 12px; font-style: italic; margin-top: 0;'>Si vis pacem, para bellum.</p>", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_df' not in st.session_state:
    st.session_state.last_df = pd.DataFrame()
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False

# Cache files for tickers
@st.cache_data
def load_cached_tickers():
    cache_file = 'tickers_cache.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return []

def save_tickers(tickers):
    cache_file = 'tickers_cache.json'
    with open(cache_file, 'w') as f:
        json.dump(tickers, f)

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
        return all_tickers
    except Exception as e:
        st.error(f"Error fetching tickers: {e}")
        return []

def process_data(date_str, percentage, filter_mode, min_avg_vol, min_rel_vol, tickers=None, use_cache=True):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if tickers is None or not use_cache:
        status_text.text('Fetching fresh ticker list...')
        progress_bar.progress(0.05)
        tickers = get_tickers()
        if not tickers:
            st.warning("Error: Could not fetch ticker list.")
            return pd.DataFrame()
        save_tickers(tickers)
        st.success(f'Fetched and cached {len(tickers)} tickers.')
        progress_bar.progress(0.1)

    date = datetime.strptime(date_str, '%Y-%m-%d').date()
    end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    start_30d = (date - timedelta(days=45)).strftime('%Y-%m-%d')

    def process_mode(mode, mode_progress_start, mode_progress_end):
        local_progress = 0.0
        total = len(tickers)
        status_text.text(f'Downloading single-day data for {total} tickers ({mode})...')
        batch_size = 100
        batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
        all_hist_single = {}
        download_processed = 0
        total_batches = len(batches)
        batch_num = 0
        for chunk in batches:
            batch_num += 1
            status_text.text(f'Downloading single-day batch {batch_num}/{total_batches} ({mode})...')
            hist_chunk = yf.download(chunk, start=date_str, end=end_date, group_by='ticker', threads=True, progress=False)
            for ticker in chunk:
                if ticker in hist_chunk.columns.get_level_values(0) and not hist_chunk[ticker].empty:
                    all_hist_single[ticker] = hist_chunk[ticker]
            download_processed += len(chunk)
            local_progress = (download_processed / total) * 0.3  # 30% for download
            overall_progress = mode_progress_start + (local_progress * (mode_progress_end - mode_progress_start))
            progress_bar.progress(overall_progress)

        status_text.text(f'Calculating signals... 0% ({mode})')
        all_data = []
        calc_processed = 0
        for ticker in tickers:
            try:
                if ticker in all_hist_single:
                    single_data = all_hist_single[ticker]
                    if not single_data.empty:
                        o = single_data['Open'].iloc[0]
                        h = single_data['High'].iloc[0]
                        l = single_data['Low'].iloc[0]
                        c = single_data['Close'].iloc[0]
                        v = single_data['Volume'].iloc[0]
                        
                        range_pct = ((h - l) / c * 100) if c != 0 else 0
                        
                        range_val = (h - l)
                        if range_val == 0:
                            close_pct = 0
                            signal = 'No'
                        else:
                            if mode == 'Bullish':
                                close_pct = ((h - c) / range_val * 100)
                            else:  # Bearish
                                close_pct = ((c - l) / range_val * 100)
                            
                            percentage_val = percentage / 100
                            if mode == 'Bullish':
                                signal = 'Yes' if (o > h - range_val * percentage_val) and (c > h - range_val * percentage_val) else 'No'
                            else:  # Bearish
                                signal = 'Yes' if (o < l + range_val * percentage_val) and (c < l + range_val * percentage_val) else 'No'
                        
                        all_data.append({
                            'Ticker': ticker,
                            'Open': round(o, 2),
                            'High': round(h, 2),
                            'Low': round(l, 2),
                            'Close': round(c, 2),
                            'Volume': int(v),
                            'Range %': round(range_pct, 2),
                            'Close %': round(close_pct, 2),
                            'Signal': signal,
                            'JW Mode': mode
                        })
            except Exception as e:
                print(f"Error for {ticker}: {e}")
                continue
            
            calc_processed += 1
            local_progress = 0.3 + (calc_processed / total) * 0.2  # 20% for calc
            overall_progress = mode_progress_start + (local_progress * (mode_progress_end - mode_progress_start))
            progress_bar.progress(overall_progress)

        if not all_data:
            return pd.DataFrame()

        df_mode = pd.DataFrame(all_data)
        df_mode = df_mode[df_mode['Signal'] == 'Yes']
        
        if df_mode.empty:
            return pd.DataFrame()

        yes_tickers_mode = df_mode['Ticker'].tolist()
        status_text.text(f'Fetching 30D history for {len(yes_tickers_mode)} matching stocks... ({mode})')

        yes_batch_size = 50
        yes_batches = [yes_tickers_mode[i:i+yes_batch_size] for i in range(0, len(yes_tickers_mode), yes_batch_size)]
        all_hist_30d = {}
        fetch_processed = 0
        yes_total_batches = len(yes_batches)
        yes_batch_num = 0
        for yes_chunk in yes_batches:
            yes_batch_num += 1
            status_text.text(f'Fetching 30D batch {yes_batch_num}/{yes_total_batches} ({mode})...')
            hist_30d_chunk = yf.download(yes_chunk, start=start_30d, end=end_date, group_by='ticker', threads=True, progress=False)
            for ticker in yes_chunk:
                if ticker in hist_30d_chunk.columns.get_level_values(0) and not hist_30d_chunk[ticker].empty:
                    all_hist_30d[ticker] = hist_30d_chunk[ticker]
            fetch_processed += len(yes_chunk)
            local_progress = 0.5 + (fetch_processed / len(yes_tickers_mode)) * 0.2  # 20% for fetch
            overall_progress = mode_progress_start + (local_progress * (mode_progress_end - mode_progress_start))
            progress_bar.progress(overall_progress)

        status_text.text(f'Computing volumes and strength... 0% ({mode})')

        comp_total = len(df_mode)
        comp_processed = 0
        for idx, row in df_mode.iterrows():
            ticker = row['Ticker']
            try:
                if ticker in all_hist_30d:
                    full_data = all_hist_30d[ticker]
                    if not full_data.empty:
                        v = row['Volume']
                        
                        volumes = full_data['Volume'].dropna()
                        single_idx = pd.to_datetime(date_str)
                        volumes_before = volumes[volumes.index < single_idx]
                        if len(volumes_before) >= 30:
                            avg_vol = volumes_before.tail(30).mean()
                        elif len(volumes_before) > 0:
                            avg_vol = volumes_before.mean()
                        else:
                            avg_vol = 0
                        
                        rel_vol = v / avg_vol if avg_vol > 0 else 0
                        
                        df_mode.at[idx, '30D Avg Vol'] = int(round(avg_vol, 0)) if avg_vol > 0 else 0
                        df_mode.at[idx, 'Relative Vol'] = round(rel_vol, 2)
            except Exception as e:
                print(f"Error for {ticker} volume: {e}")
                continue
            
            comp_processed += 1
            local_progress = 0.7 + (comp_processed / comp_total) * 0.3  # 30% for compute
            overall_progress = mode_progress_start + (local_progress * (mode_progress_end - mode_progress_start))
            progress_bar.progress(overall_progress)

        df_mode = df_mode[df_mode['30D Avg Vol'] > min_avg_vol * 1000000]
        df_mode = df_mode[df_mode['Relative Vol'] > min_rel_vol]
        
        if df_mode.empty:
            return pd.DataFrame()
        
        return df_mode

    if filter_mode == 'All':
        df_bull = process_mode('Bullish', 0.1, 0.55)
        if not df_bull.empty:
            df_bear = process_mode('Bearish', 0.55, 1.0)
        else:
            df_bear = pd.DataFrame()
        if df_bull.empty and df_bear.empty:
            st.warning('Warning: No stocks match the criteria.')
            return pd.DataFrame()
        df = pd.concat([df_bull, df_bear]) if not df_bear.empty else df_bull
    else:
        df = process_mode(filter_mode, 0.1, 1.0)
        if df.empty:
            st.warning('Warning: No stocks match the criteria.')
            return pd.DataFrame()

    if not df.empty:
        def range_score(x):
            if x <= 0.5:
                return 0
            elif x <= 1.5:
                return 5 * (x - 0.5) / 1.0
            elif x <= 3:
                return 5 + 2.5 * (x - 1.5) / 1.5
            elif x <= 5:
                return 7.5 + 2.5 * (x - 3) / 2.0
            else:
                return 10
        df['Range Score'] = df['Range %'].apply(lambda x: min(10, max(0, range_score(x))))
        
        def rel_vol_score(x):
            if x <= 0.5:
                return 0
            elif x <= 0.8:
                return 2 * (x - 0.5) / 0.3
            elif x <= 1.0:
                return 2 + 2 * (x - 0.8) / 0.2
            elif x <= 1.25:
                return 4 + 2 * (x - 1.0) / 0.25
            elif x <= 1.5:
                return 6 + 1.5 * (x - 1.25) / 0.25
            elif x <= 2.5:
                return 7.5 + 2.5 * (x - 1.5) / 1.0
            else:
                return 10
        df['Rel Vol Score'] = df['Relative Vol'].apply(lambda x: min(10, max(0, rel_vol_score(x))))
        
        df['Close Score'] = 10 * (1 - (df['Close %'] / percentage))
        df['Close Score'] = df['Close Score'].clip(lower=0, upper=10)
        
        df['Strength'] = round((df['Range Score'] + df['Close Score'] + df['Rel Vol Score']) / 3, 1)

    df = df.sort_values('Strength', ascending=False)
    
    # Save to session state history
    if not df.empty:
        new_records = []
        for _, row in df.iterrows():
            rec = row.to_dict()
            rec['Query_Date'] = date_str
            new_records.append(rec)
        st.session_state.history.extend(new_records)
        if len(st.session_state.history) > 1000:
            st.session_state.history = st.session_state.history[-1000:]
    
    return df

def get_color(strength):
    # Interpolate from red to green
    factor = strength / 10.0
    r = int(255 * (1 - factor))
    g = int(255 * factor)
    b = 0
    return f'rgb({r},{g},{b})'

def style_df(df, minimalist, table_filter):
    def highlight_strength(val):
        if isinstance(val, (int, float)):
            color = get_color(val)
            return f'color: {color}'
        return ''

    subset = df.copy()
    if table_filter != 'All':
        subset = subset[subset['JW Mode'] == table_filter]

    subset['Volume'] = subset['Volume'].apply(lambda v: f"{v/1000000:.1f} M" if isinstance(v, (int, float)) and v > 0 else "0.0 M")
    if '30D Avg Vol' in subset.columns:
        subset['30D Avg Vol'] = subset['30D Avg Vol'].apply(lambda v: f"{v/1000000:.1f} M" if isinstance(v, (int, float)) and v > 0 else "0.0 M")

    if minimalist:
        display_columns = ['Ticker', 'Close', 'Volume', 'Relative Vol', 'Range %', 'Close %', 'JW Mode', 'Strength']
        subset = subset[display_columns]
    else:
        display_columns = ['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', '30D Avg Vol', 'Relative Vol', 'Range %', 'Close %', 'JW Mode', 'Signal', 'Strength']
        subset = subset.reindex(columns=[c for c in display_columns if c in subset.columns])

    # Apply styling to Strength column
    subset = subset.style.applymap(highlight_strength, subset=pd.IndexSlice[:, ['Strength']])

    return subset

# Inputs in a modern grid layout
st.markdown('<div class="filter-section"><h3 style="color: cyan; margin-bottom: 15px;">Configuration</h3>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    date = st.date_input("Date", value=datetime.now().date(), key="date")
    date_str = date.strftime('%Y-%m-%d')
with col2:
    jw_percent = st.number_input("JW %", value=20.0, min_value=0.0, step=1.0, key="jw_percent")
    min_avg_vol = st.number_input("Min Avg Vol (M)", value=5.0, min_value=0.0, step=0.5, key="min_avg_vol")
with col3:
    jw_mode = st.selectbox("JW Mode", ['All', 'Bullish', 'Bearish'], index=0, key="jw_mode")
    min_rel_vol = st.number_input("Min Rel Vol", value=0.9, min_value=0.0, step=0.1, key="min_rel_vol")

st.markdown('</div>', unsafe_allow_html=True)

minimalist = st.checkbox("Minimalist View", key="minimalist")

# Buttons
col_btn1, col_btn2, col_btn3 = st.columns(3)
use_cache = True

if col_btn1.button("New Pull", key="new_pull"):
    use_cache = False
    with st.spinner("Running analysis..."):
        st.session_state.last_df = process_data(date_str, jw_percent, jw_mode, min_avg_vol, min_rel_vol, None, use_cache)
        st.session_state.analysis_run = True
        st.rerun()

if col_btn2.button("Fetch Data", key="fetch_data"):
    use_cache = True
    with st.spinner("Running analysis..."):
        st.session_state.last_df = process_data(date_str, jw_percent, jw_mode, min_avg_vol, min_rel_vol, load_cached_tickers(), use_cache)
        st.session_state.analysis_run = True
        st.rerun()

col_btn3.button("CANCEL", key="cancel", disabled=True)

# Status
status = st.empty()

# Tabs
tab1, tab2 = st.tabs(["Main", "History"])

with tab1:
    if st.session_state.analysis_run and not st.session_state.last_df.empty:
        status.success(f"Loaded {len(st.session_state.last_df)} stocks successfully ({jw_mode}).")
        table_filter = st.selectbox("Table Filter", ['All', 'Bullish', 'Bearish'], key="table_filter")
        styled_df = style_df(st.session_state.last_df, minimalist, table_filter)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Export
        csv = st.session_state.last_df.to_csv(index=False).encode('utf-8')
        st.download_button("EXPORT CSV", csv, "jw_terminal.csv", "text/csv", key="export")
    else:
        status.warning("No data available.")
        st.info("No data to export.")

with tab2:
    history = st.session_state.history
    if history:
        df_hist = pd.DataFrame(history)
        df_hist['Query_Date'] = pd.to_datetime(df_hist['Query_Date'])
        df_hist = df_hist.sort_values(['Query_Date', 'Strength'], ascending=[False, False])
        df_hist['Volume'] = df_hist['Volume'].apply(lambda v: f"{v/1000000:.1f} M" if isinstance(v, (int, float)) and v > 0 else "0.0 M")
        display_cols = ['Query_Date', 'Ticker', 'Close', 'Volume', 'Relative Vol', 'Range %', 'Close %', 'JW Mode', 'Strength']
        df_hist_display = df_hist[display_cols].copy()
        # Apply color to Strength
        def hist_highlight_strength(val):
            if isinstance(val, (int, float)):
                color = get_color(val)
                return f'color: {color}'
            return ''
        styled_hist = df_hist_display.style.applymap(hist_highlight_strength, subset=pd.IndexSlice[:, ['Strength']])
        st.dataframe(styled_hist, use_container_width=True, hide_index=True)
    else:
        st.info("No history available.")