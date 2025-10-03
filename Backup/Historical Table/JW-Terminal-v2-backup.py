import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import io
import subprocess

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
        border-radius: 0;
    }
    .stNumberInput > div > div > input {
        background-color: black;
        color: lime;
        border: 1px solid lime;
        border-radius: 0;
    }
    .stSelectbox > div > div > select {
        background-color: black;
        color: lime;
        border: 1px solid lime;
        border-radius: 0;
    }
    .stDateInput > div > div > input {
        background-color: black;
        color: lime;
        border: 1px solid lime;
        border-radius: 0;
    }
    .stCheckbox > div {
        color: lime;
    }
    .stButton > button {
        background-color: black;
        color: lime;
        border: 2px solid lime;
        font-family: 'Courier New', monospace;
        font-size: 10px;
        border-radius: 0;
        display: block;
        margin: 0 auto;
    }
    .stButton {
        text-align: center;
    }
    .stButton > button:hover {
        background-color: #00FF00;
        color: black;
        border: 2px solid #00FF00;
    }
    .stDownloadButton > button {
        background-color: black;
        color: lime;
        border: 2px solid lime;
        font-family: 'Courier New', monospace;
        font-size: 10px;
        border-radius: 0;
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
        border-radius: 0;
    }
    .dataframe {
        background-color: black;
        color: lime;
        font-family: 'Courier New', monospace;
        font-size: 9px;
        border-radius: 0;
    }
    .dataframe thead th {
        background-color: darkgreen;
        color: lime;
        font-weight: bold;
        font-size: 10px;
        font-family: 'Courier New', monospace;
        border-radius: 0;
        text-align: center !important;
    }
    .dataframe tbody td {
        background-color: black;
        color: lime;
        border: none;
        text-align: center !important;
    }
    .stTabs {
        background-color: black;
        border-radius: 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: black;
        border-bottom: 1px solid darkgreen;
        border-radius: 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: lime;
        font-family: 'Courier New', monospace;
        font-size: 10px;
        padding: 8px 12px;
        border-radius: 0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00FF00;
        background-color: black;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: black;
        background-color: darkgreen;
        border: 1px solid darkgreen;
        border-radius: 0;
    }
    .stAlert {
        background-color: black;
        color: lime;
        border: 1px solid lime;
        border-radius: 0;
    }
    .stInfo {
        background-color: black;
        color: lime;
        border: 1px solid lime;
        border-radius: 0;
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }
    .stWarning {
        background-color: black;
        color: #FF6B6B;
        border: 1px solid #FF6B6B;
        border-radius: 0;
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }
    .stSuccess {
        background-color: black;
        color: #51CF66;
        border: 1px solid #51CF66;
        border-radius: 0;
    }
    .stSelectbox > label {
        color: lime;
    }
    .stTextInputs > label {
        color: lime;
    }
    .stNumberInput > label {
        color: lime;
    }
    .stDateInput > label {
        color: lime;
    }
    .css-1aumxhk {
        text-align: center;
    }
    /* Ensure all numerical cells are centered */
    .dataframe td.numeric {
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and subheader
st.markdown("<h1 style='text-align: center; color: lime; font-family: \"Courier New\", monospace; font-size: 16px; font-weight: bold; margin-bottom: 0;'>JOHN WICK TERMINAL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: lime; font-family: \"Courier New\", monospace; font-size: 12px; font-style: italic; margin-top: 0;'>Si vis pacem, para bellum.</p>", unsafe_allow_html=True)

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

def save_historical_signals():
    cache_file = 'historical_signals.json'
    with open(cache_file, 'w') as f:
        json.dump(st.session_state.history, f, default=str)

# Top right controls
col_ctrls1, col_ctrls2, col_history, col_ctrls3 = st.columns([17, 2, 1, 1])
with col_ctrls2:
    minimalist = st.checkbox("Minimalist View", key="minimalist")
with col_history:
    if st.button("⚡", key="history_icon", help="Historical Signals"):
        st.session_state.show_history = not st.session_state.show_history
with col_ctrls3:
    if st.button("⚙️", key="settings_icon", help="Settings"):
        st.session_state.show_settings = not st.session_state.show_settings

# Initialize session state
if 'history' not in st.session_state:
    hist_file = 'historical_signals.json'
    if os.path.exists(hist_file):
        with open(hist_file, 'r') as f:
            st.session_state.history = json.load(f)
    else:
        st.session_state.history = []
if 'last_df' not in st.session_state:
    st.session_state.last_df = pd.DataFrame()
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False
if 'custom_tickers' not in st.session_state:
    st.session_state.custom_tickers = {}
if 'ticker_lists' not in st.session_state:
    # Load from file if exists
    ticker_lists_file = 'ticker_lists.json'
    if os.path.exists(ticker_lists_file):
        with open(ticker_lists_file, 'r') as f:
            st.session_state.ticker_lists = json.load(f)
    else:
        st.session_state.ticker_lists = {}
if 'selected_ticker_list' not in st.session_state:
    st.session_state.selected_ticker_list = 'Default'
if 'show_settings' not in st.session_state:
    st.session_state.show_settings = False
if 'show_history' not in st.session_state:
    st.session_state.show_history = False
if 'default_tickers' not in st.session_state:
    st.session_state.default_tickers = load_cached_tickers()

# Function to save ticker lists
def save_ticker_lists():
    with open('ticker_lists.json', 'w') as f:
        json.dump(st.session_state.ticker_lists, f)

def push_to_github(files, show_success=True):
    try:
        subprocess.run(['git', 'add'] + files, check=True)
        subprocess.run(['git', 'commit', '-m', f'Update {" and ".join(files)}'], check=True)
        subprocess.run(['git', 'push'], check=True)
        if show_success:
            st.success(f"Pushed updated {' and '.join(files)} to GitHub.")
    except Exception as e:
        if show_success:
            st.warning(f"Could not push to GitHub: {e}. Ensure git is configured with credentials.")

def get_tickers():
    all_tickers = []
    
    # S&P 500 from Wikipedia
    try:
        sp_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        all_tickers.extend(sp_table['Symbol'].tolist())
    except:
        pass
    
    # S&P 500 from stockanalysis.com
    try:
        sp_table = pd.read_html('https://stockanalysis.com/list/sp-500-stocks/')[0]
        all_tickers.extend(sp_table['Symbol'].tolist())
    except:
        pass
    
    # Nasdaq-100 from Wikipedia
    try:
        nasdaq_tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        for table in nasdaq_tables:
            if 'Ticker' in table.columns:
                all_tickers.extend(table['Ticker'].tolist())
                break
    except:
        pass
    
    # Nasdaq-100 from slickcharts
    try:
        tables = pd.read_html('https://www.slickcharts.com/nasdaq100')
        for table in tables:
            if 'Symbol' in table.columns:
                all_tickers.extend(table['Symbol'].tolist())
                break
    except:
        pass
    
    # High volume ETFs from etfdb.com
    try:
        tables = pd.read_html('https://etfdb.com/compare/volume/')
        for table in tables:
            if 'Symbol' in table.columns:
                # Take top 50 high volume ETFs
                top_etfs = table['Symbol'].head(50).tolist()
                all_tickers.extend(top_etfs)
                break
    except:
        # Fallback hardcoded high volume ETFs
        hardcoded_etfs = ['SPY', 'QQQ', 'IWM', 'EEM', 'TLT', 'GLD', 'USO', 'XLF', 'XLE', 'XLI', 'XLB', 'XLY', 'XLP', 'XLU', 'XLK', 'XLV', 'XLI', 'XLRE', 'XLC', 'VXX', 'UVXY', 'TQQQ', 'SQQQ']
        all_tickers.extend(hardcoded_etfs)
    
    # Sector SPDRs hardcoded
    spdr_sectors = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLK']
    all_tickers.extend(spdr_sectors)
    
    all_tickers = list(set(all_tickers))
    all_tickers = [t for t in all_tickers if t not in ['BF.B', 'BRK.B']]
    return all_tickers

def custom_progress(container, value, text):
    text_color = "lime" if value < 0.5 else "black"
    html = f"""
    <div style="
        width: 100%;
        height: 25px;
        background-color: black;
        border: 2px solid lime;
        position: relative;
        border-radius: 0;
        font-family: 'Courier New', monospace;
        font-size: 11px;
    ">
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background-color: lime;
            width: {int(value * 100)}%;
            transition: width 0.2s ease;
        "></div>
        <span style="
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: {text_color};
            z-index: 1;
            white-space: nowrap;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        ">{text}</span>
    </div>
    """
    container.markdown(html, unsafe_allow_html=True)

def process_mode(mode, mode_progress_start, mode_progress_end, progress_container, tickers, date_str, percentage, start_30d, end_date, min_avg_vol, min_rel_vol, min_rvolat):
    def process_local_progress(local_prog):
        overall = mode_progress_start + (local_prog * (mode_progress_end - mode_progress_start))
        return overall

    total = len(tickers)
    custom_progress(progress_container, process_local_progress(0), f'Downloading single-day data for {total} tickers ({mode})...')
    batch_size = 100
    batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
    all_hist_single = {}
    download_processed = 0
    total_batches = len(batches)
    batch_num = 0
    for chunk in batches:
        batch_num += 1
        custom_progress(progress_container, process_local_progress((download_processed / total) * 0.3), f'Downloading single-day batch {batch_num}/{total_batches} ({mode})...')
        hist_chunk = yf.download(chunk, start=date_str, end=end_date, group_by='ticker', threads=True, progress=False)
        for ticker in chunk:
            if ticker in hist_chunk.columns.get_level_values(0) and not hist_chunk[ticker].empty:
                all_hist_single[ticker] = hist_chunk[ticker]
        download_processed += len(chunk)

    custom_progress(progress_container, process_local_progress(0.3), f'Calculating signals... 0% ({mode})')
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
                        'JW %': round(close_pct, 2),
                        'Signal': signal,
                        'JW Signal': mode
                    })
        except Exception as e:
            print(f"Error for {ticker}: {e}")
            continue
        
        calc_processed += 1
        local_progress = 0.3 + (calc_processed / total) * 0.2  # 20% for calc
        custom_progress(progress_container, process_local_progress(local_progress), f'Calculating signals... {int((calc_processed / total) * 100)}% ({mode})')

    if not all_data:
        return pd.DataFrame()

    df_mode = pd.DataFrame(all_data)
    df_mode = df_mode[df_mode['Signal'] == 'Yes']
    
    if df_mode.empty:
        return pd.DataFrame()

    yes_tickers_mode = df_mode['Ticker'].tolist()
    custom_progress(progress_container, process_local_progress(0.5), f'Fetching 30D history for {len(yes_tickers_mode)} matching stocks... ({mode})')

    yes_batch_size = 50
    yes_batches = [yes_tickers_mode[i:i+yes_batch_size] for i in range(0, len(yes_tickers_mode), yes_batch_size)]
    all_hist_30d = {}
    fetch_processed = 0
    yes_total_batches = len(yes_batches)
    yes_batch_num = 0
    for yes_chunk in yes_batches:
        yes_batch_num += 1
        custom_progress(progress_container, process_local_progress(0.5 + (fetch_processed / len(yes_tickers_mode)) * 0.2), f'Fetching 30D batch {yes_batch_num}/{yes_total_batches} ({mode})...')
        hist_30d_chunk = yf.download(yes_chunk, start=start_30d, end=end_date, group_by='ticker', threads=True, progress=False)
        for ticker in yes_chunk:
            if ticker in hist_30d_chunk.columns.get_level_values(0) and not hist_30d_chunk[ticker].empty:
                all_hist_30d[ticker] = hist_30d_chunk[ticker]
        fetch_processed += len(yes_chunk)

    custom_progress(progress_container, process_local_progress(0.7), f'Computing volumes and strength... 0% ({mode})')

    # Initialize columns
    df_mode['30D Avg Vol'] = 0
    df_mode['rVolume'] = 0.0
    df_mode['rVolatility'] = 0.0

    comp_total = len(df_mode)
    comp_processed = 0
    single_idx = pd.to_datetime(date_str)
    for idx, row in df_mode.iterrows():
        ticker = row['Ticker']
        try:
            if ticker in all_hist_30d:
                full_data = all_hist_30d[ticker]
                if not full_data.empty:
                    v = row['Volume']
                    
                    volumes = full_data['Volume'].dropna()
                    volumes_before = volumes[volumes.index < single_idx]
                    if len(volumes_before) >= 30:
                        avg_vol = volumes_before.tail(30).mean()
                    elif len(volumes_before) > 0:
                        avg_vol = volumes_before.mean()
                    else:
                        avg_vol = 0
                    
                    rel_vol = v / avg_vol if avg_vol > 0 else 0
                    
                    df_mode.at[idx, '30D Avg Vol'] = int(round(avg_vol, 0)) if avg_vol > 0 else 0
                    df_mode.at[idx, 'rVolume'] = round(rel_vol, 2)
                    
                    # Compute rVolatility
                    current_range = (row['High'] - row['Low']) / row['Close'] if row['Close'] != 0 else 0
                    full_data_before = full_data[full_data.index < single_idx]
                    if not full_data_before.empty:
                        daily_ranges = ((full_data_before['High'] - full_data_before['Low']) / full_data_before['Close']).dropna()
                        if len(daily_ranges) >= 30:
                            avg_range = daily_ranges.tail(30).mean()
                        elif len(daily_ranges) > 0:
                            avg_range = daily_ranges.mean()
                        else:
                            avg_range = 0
                    else:
                        avg_range = 0
                    rvolat = current_range / avg_range if avg_range > 0 else 0
                    df_mode.at[idx, 'rVolatility'] = round(rvolat, 2)
        except Exception as e:
            print(f"Error for {ticker} volume: {e}")
            continue
        
        comp_processed += 1
        local_progress = 0.7 + (comp_processed / comp_total) * 0.3  # 30% for compute
        custom_progress(progress_container, process_local_progress(local_progress), f'Computing volumes and strength... {int((comp_processed / comp_total) * 100)}% ({mode})')

    df_mode = df_mode[df_mode['30D Avg Vol'] > min_avg_vol * 1000000]
    df_mode = df_mode[df_mode['rVolume'] > min_rel_vol]
    df_mode = df_mode[df_mode['rVolatility'] > min_rvolat]
    
    if df_mode.empty:
        return pd.DataFrame()
    
    return df_mode

def process_data(date_str, percentage, filter_mode, min_avg_vol, min_rel_vol, min_rvolat, tickers=None, progress_container=None):
    if tickers is None or len(tickers) == 0:
        if progress_container:
            custom_progress(progress_container, 1.0, "John Wicks Not Identified.")
        return pd.DataFrame()
    
    if progress_container is None:
        progress_container = st.empty()
    
    custom_progress(progress_container, 0, "Initializing analysis...")
    
    date = datetime.strptime(date_str, '%Y-%m-%d').date()
    end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    start_30d = (date - timedelta(days=45)).strftime('%Y-%m-%d')

    if filter_mode == 'All':
        df_bull = process_mode('Bullish', 0.0, 0.5, progress_container, tickers, date_str, percentage, start_30d, end_date, min_avg_vol, min_rel_vol, min_rvolat)
        df_bear = process_mode('Bearish', 0.5, 1.0, progress_container, tickers, date_str, percentage, start_30d, end_date, min_avg_vol, min_rel_vol, min_rvolat)
        df = pd.concat([df_bull, df_bear]) if not df_bear.empty else df_bull
    else:
        df = process_mode(filter_mode, 0.0, 1.0, progress_container, tickers, date_str, percentage, start_30d, end_date, min_avg_vol, min_rel_vol, min_rvolat)

    if df.empty:
        custom_progress(progress_container, 1.0, "John Wicks Not Identified.")
        return pd.DataFrame()

    custom_progress(progress_container, 1.0, "John Wicks Identified.")

    def rvolat_score(x):
        if x <= 0.7:
            return 0
        elif x <= 1.2:
            return 5 * (x - 0.7) / 0.5
        elif x <= 2.0:
            return 5 + 5 * (x - 1.2) / 0.8
        else:
            return 10
    df['rVolat Score'] = df['rVolatility'].apply(lambda x: min(10, max(0, rvolat_score(x))))
    
    def rvolume_score(x):
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
    df['rVolume Score'] = df['rVolume'].apply(lambda x: min(10, max(0, rvolume_score(x))))
    
    df['Close Score'] = 10 * (1 - (df['JW %'] / percentage))
    df['Close Score'] = df['Close Score'].clip(lower=0, upper=10)
    
    df['Strength'] = round((df['rVolat Score'] + df['Close Score'] + df['rVolume Score']) / 3, 1)

    df = df.sort_values('Strength', ascending=False)
    
    # Save to session state history
    if not df.empty:
        new_records = []
        filter_settings = f"{percentage} | {min_avg_vol} | {min_rel_vol} | {min_rvolat}"
        for _, row in df.iterrows():
            rec = row.to_dict()
            rec['Query_Date'] = date_str
            rec['Filter_Settings'] = filter_settings
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

def highlight_mode(val):
    if val == 'Bullish':
        return 'color: lime'
    elif val == 'Bearish':
        return 'color: red'
    return ''

def highlight_t_returns(row):
    styles = [''] * len(row)
    signal = row['JW Signal']
    if pd.isna(signal):
        signal = ''
    if isinstance(signal, str):
        for i, col in enumerate(row.index):
            if col in ['T+1', 'T+2', 'T+3', 'T+7'] and pd.notna(row[col]):
                ret_val = float(row[col])
                if signal == 'Bullish':
                    color = 'color: lime' if ret_val > 0 else 'color: red'
                elif signal == 'Bearish':
                    color = 'color: lime' if ret_val < 0 else 'color: red'
                else:
                    color = ''
                styles[i] = color
    return styles

def style_df(df, minimalist):
    def highlight_strength(val):
        if isinstance(val, (int, float)):
            color = get_color(val)
            return f'color: {color}'
        return ''

    subset = df.copy()

    # Format Volume to M before styling
    if 'Volume' in subset.columns:
        subset['Volume'] = subset['Volume'].apply(lambda v: f"{v/1000000:.2f} M" if isinstance(v, (int, float)) and v > 0 else "0.00 M")
    if '30D Avg Vol' in subset.columns:
        subset['30D Avg Vol'] = subset['30D Avg Vol'].apply(lambda v: f"{v/1000000:.2f} M" if isinstance(v, (int, float)) and v > 0 else "0.00 M")

    if minimalist:
        display_columns = ['Ticker', 'Close', 'Volume', 'rVolume', 'rVolatility', 'JW %', 'JW Signal', 'Strength']
        subset = subset[display_columns]
    else:
        display_columns = ['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', '30D Avg Vol', 'rVolume', 'rVolatility', 'JW %', 'JW Signal', 'Strength']
        subset = subset.reindex(columns=[c for c in display_columns if c in subset.columns])

    # Apply styling to Strength and JW Signal columns
    subset = subset.style.applymap(highlight_strength, subset=pd.IndexSlice[:, ['Strength']]).applymap(highlight_mode, subset=pd.IndexSlice[:, ['JW Signal']])

    # Format numerics to 2 decimals (skip Volume since string)
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'rVolume', 'rVolatility', 'JW %', 'Strength']
    format_dict = {col: '{:.2f}' for col in numeric_cols if col in subset.columns}

    subset = subset.format(format_dict)

    return subset

# Template CSV for tickers
template_df = pd.DataFrame({'Ticker': ['AAPL', 'GOOGL']})
template_csv = template_df.to_csv(index=False).encode('utf-8')

# Historical Signals popup
if st.session_state.show_history:
    st.markdown("")
    hist_df = pd.DataFrame(st.session_state.history)
    if not hist_df.empty:
        hist_df['Date'] = pd.to_datetime(hist_df['Query_Date']).dt.date
        hist_df['Filter Settings'] = hist_df['Filter_Settings']

        # Migrate old columns
        if 'Close %' in hist_df.columns:
            hist_df['JW %'] = hist_df['Close %']
            hist_df.drop(columns=['Close %'], inplace=True)
        if 'Relative Vol' in hist_df.columns:
            hist_df['rVolume'] = hist_df['Relative Vol']
            hist_df.drop(columns=['Relative Vol'], inplace=True)
        if 'Range %' in hist_df.columns:
            hist_df.drop(columns=['Range %'], inplace=True)
        for col in ['rVolume', 'rVolatility', 'JW %']:
            if col not in hist_df.columns:
                hist_df[col] = float('nan')

        # Ensure T+ columns exist in history dicts
        for rec in st.session_state.history:
            for days in [1, 2, 3, 7]:
                if f'T+{days}' not in rec:
                    rec[f'T+{days}'] = None

        # Refresh hist_df
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df['Date'] = pd.to_datetime(hist_df['Query_Date']).dt.date
        hist_df['Filter Settings'] = hist_df['Filter_Settings']

        # Migrate again after refresh
        if 'Close %' in hist_df.columns:
            hist_df['JW %'] = hist_df['Close %']
            hist_df.drop(columns=['Close %'], inplace=True)
        if 'Relative Vol' in hist_df.columns:
            hist_df['rVolume'] = hist_df['Relative Vol']
            hist_df.drop(columns=['Relative Vol'], inplace=True)
        if 'Range %' in hist_df.columns:
            hist_df.drop(columns=['Range %'], inplace=True)
        for col in ['rVolume', 'rVolatility', 'JW %']:
            if col not in hist_df.columns:
                hist_df[col] = float('nan')

        # Ensure columns are numeric
        for col in ['T+1', 'T+2', 'T+3', 'T+7']:
            hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')

        # Buttons
        col_btn_hist = st.columns([3, 1])
        with col_btn_hist[0]:
            compute_clicked = st.button("Run Backtest")
        with col_btn_hist[1]:
            full_compute_clicked = st.button("Full Refresh")

        progress_hist = st.empty()

        if compute_clicked or full_compute_clicked:
            today = datetime.now().date()
            if full_compute_clicked:
                tickers_to_compute = hist_df['Ticker'].unique()
                button_text = "Full Refresh"
            else:
                tickers_with_none_t7 = hist_df[hist_df['T+7'].isna()]['Ticker'].unique()
                if len(tickers_with_none_t7) == 0:
                    pass
                else:
                    tickers_to_compute = tickers_with_none_t7
                    button_text = "Backtest"

            if 'tickers_to_compute' in locals():
                computed = 0
                total_tickers = len(tickers_to_compute)
                custom_progress(progress_hist, 0, f"Starting {button_text}...")
                for ticker in tickers_to_compute:
                    print(f"Processing ticker: {ticker}")
                    ticker_rows = hist_df[hist_df['Ticker'] == ticker]
                    if ticker_rows.empty:
                        continue
                    earliest_date = min(ticker_rows['Date']) - timedelta(days=3)
                    start_str = earliest_date.strftime('%Y-%m-%d')
                    end_str = (today + timedelta(days=10)).strftime('%Y-%m-%d')  # Extend to ensure T+7 data
                    try:
                        hist_all = yf.download(ticker, start=start_str, end=end_str, progress=False, threads=False)
                        print(f"Downloaded data for {ticker}, shape: {hist_all.shape}, type: {type(hist_all)}")
                        if not hist_all.empty:
                            hist_all = hist_all.tz_localize(None) if hist_all.index.tz else hist_all  # Ensure no timezone
                            for _, row in ticker_rows.iterrows():
                                # Find the corresponding record index in history
                                rec_idx = None
                                for k, rec in enumerate(st.session_state.history):
                                    if (rec['Ticker'] == ticker and 
                                        rec['Query_Date'] == row['Query_Date']):
                                        rec_idx = k
                                        break
                                if rec_idx is None:
                                    print(f"Could not find record for {ticker} on {row['Query_Date']}")
                                    continue

                                t_date = row['Date']
                                t_close_stored = row['Close']
                                t_idx = pd.to_datetime(t_date)
                                print(f"Checking date {t_date} for {ticker}, t_idx: {t_idx}")
                                if t_idx in hist_all.index:
                                    pos = hist_all.index.get_loc(t_idx)
                                    print(f"Found position {pos} for {t_date}")
                                    if isinstance(pos, slice):
                                        pos = pos.start
                                        print(f"Warning: slice returned for {t_date}, using start {pos}")
                                    t_close_verify = hist_all.iloc[pos]['Close']
                                    print(f"Stored close: {t_close_stored}, downloaded: {t_close_verify}")
                                    if abs(float(t_close_verify) - t_close_stored) > 0.01:
                                        print(f"Warning: Close price mismatch for {ticker} on {t_date}: stored {t_close_stored}, downloaded {t_close_verify}")
                                        # Update stored close to match downloaded
                                        st.session_state.history[rec_idx]['Close'] = round(float(t_close_verify), 2)
                                    t_close = float(t_close_verify)  # Use verified close for calculation
                                    for days in [1, 2, 3, 7]:
                                        if pos + days < len(hist_all):
                                            f_pos = pos + days
                                            f_idx = hist_all.index[f_pos]
                                            close_f = hist_all.iloc[f_pos]['Close']
                                            ret = (float(close_f) / t_close - 1) * 100
                                            st.session_state.history[rec_idx][f'T+{days}'] = round(ret, 2)
                                            print(f"  T+{days}: {f_idx.date()} return {ret:.2f}% (using position {f_pos})")
                                        else:
                                            print(f"  Not enough future data for T+{days} on {t_date}")
                                            st.session_state.history[rec_idx][f'T+{days}'] = None
                                else:
                                    print(f"Signal date {t_date} not in downloaded data for {ticker}. Available dates: {hist_all.index[:5].tolist()} ... {hist_all.index[-5:].tolist()}")
                        else:
                            print(f"No data downloaded for {ticker}")
                    except Exception as e:
                        print(f"Error fetching {ticker}: {e}")
                        import traceback
                        print(traceback.format_exc())
                        continue
                    computed += 1
                    custom_progress(progress_hist, computed / total_tickers, f"Processed {computed}/{total_tickers} tickers")
                custom_progress(progress_hist, 1.0, "Computation complete.")

                save_historical_signals()
                push_to_github(['historical_signals.json'], show_success=False)

                # Refresh hist_df after updates
                hist_df = pd.DataFrame(st.session_state.history)
                hist_df['Date'] = pd.to_datetime(hist_df['Query_Date']).dt.date
                hist_df['Filter Settings'] = hist_df['Filter_Settings']

                # Migrate again
                if 'Close %' in hist_df.columns:
                    hist_df['JW %'] = hist_df['Close %']
                    hist_df.drop(columns=['Close %'], inplace=True)
                if 'Relative Vol' in hist_df.columns:
                    hist_df['rVolume'] = hist_df['Relative Vol']
                    hist_df.drop(columns=['Relative Vol'], inplace=True)
                if 'Range %' in hist_df.columns:
                    hist_df.drop(columns=['Range %'], inplace=True)
                for col in ['rVolume', 'rVolatility', 'JW %']:
                    if col not in hist_df.columns:
                        hist_df[col] = float('nan')

                # Ensure columns are numeric
                for col in ['T+1', 'T+2', 'T+3', 'T+7']:
                    hist_df[col] = pd.to_numeric(hist_df[col], errors='coerce')

        # Handle JW Signal column to avoid duplicates
        if 'JW Mode' in hist_df.columns:
            if 'JW Signal' not in hist_df.columns:
                hist_df = hist_df.rename(columns={'JW Mode': 'JW Signal'})
            else:
                hist_df['JW Signal'] = hist_df['JW Signal'].fillna(hist_df['JW Mode'])
                hist_df.drop(columns=['JW Mode'], inplace=True)

        display_cols = ['Date', 'Ticker', 'Close', 'rVolume', 'rVolatility', 'JW %', 'JW Signal', 'Strength', 'Filter Settings', 'T+1', 'T+2', 'T+3', 'T+7']
        hist_df = hist_df[display_cols].sort_values('Date', ascending=False)
        def format_return(x):
            return f"{x:.2f}%" if pd.notna(x) else "N/A"
        styled_hist = (hist_df.style
                       .apply(highlight_t_returns, axis=1)
                       .applymap(highlight_mode, subset=pd.IndexSlice[:, ['JW Signal']])
                       .format({
                           'Close': '{:.2f}',
                           'rVolume': '{:.2f}',
                           'rVolatility': '{:.2f}',
                           'JW %': '{:.2f}',
                           'Strength': '{:.2f}',
                           'Volume': '{:.2f}',
                           'T+1': format_return,
                           'T+2': format_return,
                           'T+3': format_return,
                           'T+7': format_return,
                       }))
        st.dataframe(styled_hist, use_container_width=True, hide_index=True)
    else:
        st.info("No historical signals yet.")
    if st.button("Close"):
        st.session_state.show_history = False
        st.rerun()

# Settings popup
if st.session_state.show_settings:
    st.markdown("## Settings")
    ticker_list_options = ['Default'] + list(st.session_state.ticker_lists.keys())
    current_list = st.selectbox("Select Ticker List", ticker_list_options, index=ticker_list_options.index(st.session_state.selected_ticker_list) if st.session_state.selected_ticker_list in ticker_list_options else 0)
    
    st.subheader("Default Tickers Cache")
    col_fetch1, col_fetch2 = st.columns(2)
    with col_fetch1:
        if st.button("Fetch Tickers"):
            with st.spinner("Fetching new tickers..."):
                new_tickers = get_tickers()
                old_len = len(st.session_state.default_tickers)
                st.session_state.default_tickers = list(set(st.session_state.default_tickers + new_tickers))
                save_tickers(st.session_state.default_tickers)
                if len(st.session_state.default_tickers) > old_len:
                    st.success(f"Added {len(st.session_state.default_tickers) - old_len} new tickers. Total: {len(st.session_state.default_tickers)}")
                    push_to_github(['tickers_cache.json'])
                else:
                    st.info("No new tickers added.")
    
    with col_fetch2:
        st.info(f"Current cached tickers: {len(st.session_state.default_tickers)}")
    
    st.subheader("Create New List")
    st.download_button("Download Template", template_csv, "ticker_template.csv", "text/csv")
    with st.form("new_list_form"):
        list_name = st.text_input("List Name")
        uploaded_file = st.file_uploader("Upload CSV (one column: Ticker)", type="csv")
        save_new_btn = st.form_submit_button("Save New List")
        if save_new_btn and uploaded_file and list_name:
            df_upload = pd.read_csv(uploaded_file)
            if 'Ticker' in df_upload.columns:
                tickers = df_upload['Ticker'].dropna().astype(str).tolist()
                st.session_state.ticker_lists[list_name] = tickers
                save_ticker_lists()
                st.rerun()
            else:
                pass

    if st.button("Save Settings & Close"):
        st.session_state.selected_ticker_list = current_list
        st.session_state.show_settings = False
        st.rerun()

# Inputs
col1, col2, col3 = st.columns(3)
with col1:
    date = st.date_input("Date", value=datetime.now().date(), key="date")
    date_str = date.strftime('%Y-%m-%d')
    min_rvolat = st.number_input("Min rVolatility", value=1.0, min_value=0.0, step=0.1, key="min_rvolat")
with col2:
    jw_percent = st.number_input("JW %", value=20.0, min_value=0.0, step=1.0, key="jw_percent")
    min_avg_vol = st.number_input("Min aVolume (M)", value=5.0, min_value=0.0, step=0.5, key="min_avg_vol")
with col3:
    jw_mode = st.selectbox("JW Mode", ['All', 'Bullish', 'Bearish'], index=0, key="jw_mode")
    min_rel_vol = st.number_input("Min rVolume", value=0.9, min_value=0.0, step=0.1, key="min_rel_vol")

progress_container = st.empty()

if st.session_state.analysis_run:
    if st.session_state.last_df.empty:
        custom_progress(progress_container, 1.0, "John Wicks Not Identified.")
    else:
        custom_progress(progress_container, 1.0, "John Wicks Identified.")
else:
    custom_progress(progress_container, 0, "Find John Wicks.")

# Analysis button
col_btn = st.columns([1])
selected = st.session_state.selected_ticker_list
tickers_to_use = st.session_state.default_tickers if selected == 'Default' else st.session_state.ticker_lists.get(selected, [])

with col_btn[0]:
    old_hist_len = len(st.session_state.history)
    if st.button("Fortis Fortuna Adiuvat", key="run_analysis"):
        st.session_state.last_df = process_data(date_str, jw_percent, jw_mode, min_avg_vol, min_rel_vol, min_rvolat, tickers_to_use, progress_container)
        st.session_state.analysis_run = True
        if len(st.session_state.history) > old_hist_len:
            save_historical_signals()
            push_to_github(['historical_signals.json'], show_success=False)
        st.rerun()

# Results table
if st.session_state.analysis_run:
    if not st.session_state.last_df.empty:
        styled_df = style_df(st.session_state.last_df, minimalist)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Export
        csv = st.session_state.last_df.to_csv(index=False).encode('utf-8')
        st.download_button("EXPORT CSV", csv, "jw_terminal.csv", "text/csv", key="export")