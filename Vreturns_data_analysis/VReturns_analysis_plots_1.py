import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import warnings
import yfinance as yf

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='invalid value encountered in log')
warnings.filterwarnings('ignore', message='invalid value encountered in sqrt')
warnings.filterwarnings('ignore', message='divide by zero encountered in divide')
warnings.filterwarnings('ignore', message='Optimal parameters not found')

# Set matplotlib style and font parameters
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})

# --- Data Loading Functions ---

def load_financial_data(indices, data_folder='financial_data'):
    """Load financial data from pickle files."""
    data = {}
    all_prices_raw = {}
    for index_name in indices:
        close_filename = os.path.join(data_folder, f'{index_name}_close.pkl')
        full_filename = os.path.join(data_folder, f'{index_name}_full.pkl')
        if os.path.exists(close_filename):
            with open(close_filename, 'rb') as f:
                data[index_name] = pickle.load(f)
        if os.path.exists(full_filename):
            with open(full_filename, 'rb') as f:
                all_prices_raw[index_name] = pickle.load(f)
    return data, all_prices_raw

def filter_data_by_period(data, all_prices_raw, start_date='1992-01-02', end_date='2023-12-29'):
    """Filter data to specific period with index-specific start dates.
    
    Start dates are adjusted to ensure returns begin on the target dates:
    - For returns to start on 1992-01-02, prices must start on 1992-01-01
    - For Nikkei returns to start on 1992-01-06, prices must start on 1992-01-05
    """
    # Define specific start dates for each index
    index_start_dates = {
        'DJIA': '1992-01-02',  
        'DAX': '1992-01-02',    
        'IPC': '1992-01-02',    
        'Nikkei': '1992-01-06'  
    }
    
    end_date = pd.to_datetime(end_date)
    
    print(f"Filtering data with index-specific start dates:")
    
    filtered_data = {}
    filtered_all_prices_raw = {}
    
    for index_name, price_series in data.items():
        # Use index-specific start date
        index_start = pd.to_datetime(index_start_dates.get(index_name, start_date))
        
        print(f"  {index_name}: {index_start.date()} to {end_date.date()}")
        
        if not price_series.empty:
            mask = (price_series.index >= index_start) & (price_series.index <= end_date)
            filtered_data[index_name] = price_series[mask]
        else:
            filtered_data[index_name] = price_series
            
        if index_name in all_prices_raw:
            price_df = all_prices_raw[index_name]
            if not price_df.empty:
                mask_df = (price_df.index >= index_start) & (price_df.index <= end_date)
                filtered_all_prices_raw[index_name] = price_df[mask_df]
            else:
                filtered_all_prices_raw[index_name] = price_df
    
    
    return filtered_data, filtered_all_prices_raw

def download_financial_data():
    """Download or load financial data for analysis."""
    indices = ['DJIA', 'DAX', 'IPC', 'Nikkei']
    
    # Try to load existing data first
    data, all_prices_raw = load_financial_data(indices)
    
    if not data or any(data[idx].empty for idx in indices if idx in data):
        print("Downloading financial data from Yahoo Finance...")
        
        # Yahoo Finance tickers
        tickers = {
            'DJIA': '^DJI',
            'DAX': '^GDAXI', 
            'IPC': '^MXX',
            'Nikkei': '^N225'
        }
        
        data = {}
        all_prices_raw = {}
        
        for index_name, ticker in tickers.items():
            try:
                stock_data = yf.download(ticker, start='1990-01-01', end='2024-01-01', progress=False)
                if not stock_data.empty:
                    all_prices_raw[index_name] = stock_data
                    data[index_name] = stock_data['Close']
                    print(f"Downloaded {index_name}: {len(stock_data)} records")
                else:
                    print(f"Warning: No data downloaded for {index_name}")
                    data[index_name] = pd.Series(dtype=float)
                    all_prices_raw[index_name] = pd.DataFrame()
            except Exception as e:
                print(f"Error downloading {index_name}: {e}")
                data[index_name] = pd.Series(dtype=float)
                all_prices_raw[index_name] = pd.DataFrame()
    
    # Filter data to the specified period
    data, all_prices_raw = filter_data_by_period(data, all_prices_raw)
    
    return data, all_prices_raw

# --- Trend Analysis Functions ---

def identify_trends(prices):
    """Identify trends in price series."""
    trends = []
    if len(prices) < 3:
        return trends
    
    current_direction = None
    trend_start = 0
    
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            direction = 'up'
        elif prices[i] < prices[i-1]:
            direction = 'down'
        else:
            continue
        
        if current_direction != direction:
            if current_direction is not None:
                trends.append({
                    'start': trend_start,
                    'end': i-1,
                    'direction': current_direction
                })
            trend_start = i-1
            current_direction = direction
    
    if current_direction is not None:
        trends.append({
            'start': trend_start,
            'end': len(prices)-1,
            'direction': current_direction
        })
    
    return trends

def calculate_VReturns_by_duration(prices, trends, target_durations=[1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """Calculate VReturns grouped by trend duration.
    
    For each trend of duration n days, calculate:
    (log(P_2)-log(P_1))/1, (log(P_3)-log(P_1))/2, ..., (log(P_n+1)-log(P_1))/n
    
    Args:
        prices: Array of price values
        trends: List of trend dictionaries
        target_durations: List of trend durations to extract (default: [1, 2, 3, 4])
    
    Returns:
        Dictionary with duration as key and list of VReturns as value
    """
    VReturns_by_duration = {duration: [] for duration in target_durations}
    
    for trend in trends:
        start_idx = trend['start']
        end_idx = trend['end']
        duration = end_idx - start_idx
        
        if duration < 1:  # Need at least 1 day duration
            continue
            
        p_start = prices[start_idx]
        if p_start <= 0:
            continue
            
        log_p_start = np.log(p_start)
        
        # Calculate VReturns for this trend
        for i in range(1, duration + 1):
            price_idx = start_idx + i
            if price_idx < len(prices):
                p_i = prices[price_idx]
                if p_i > 0:
                    log_p_i = np.log(p_i)
                    tvreturn = (log_p_i - log_p_start) / i
                    
                    # Only store VReturns for trends of target durations
                    if duration in target_durations:
                        VReturns_by_duration[duration].append(tvreturn)
    
    # Convert lists to numpy arrays
    for duration in target_durations:
        VReturns_by_duration[duration] = np.array(VReturns_by_duration[duration])
    
    return VReturns_by_duration

def extract_VReturns_by_duration_and_index(all_prices_data, target_durations=[1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """Extract VReturns grouped by duration for all indices."""
    VReturns_by_duration_and_index = {}
    
    for index_name, price_series in all_prices_data.items():
        if price_series.empty:
            VReturns_by_duration_and_index[index_name] = {duration: np.array([]) for duration in target_durations}
            continue
            
        prices = price_series.values
        trends = identify_trends(prices)
        VReturns_by_duration = calculate_VReturns_by_duration(prices, trends, target_durations)
        
        VReturns_by_duration_and_index[index_name] = VReturns_by_duration
    
    return VReturns_by_duration_and_index

def calculate_trend_VReturns_by_duration(prices, trends, target_durations=[1, 2, 3, 4]):
    """Calculate TVReturns grouped by trend duration.
    
    For each trend of duration k days:
    TSₘᵏ := (log(Pₘ₊ₖ) - log(Pₘ)) / k
    
    Args:
        prices: Array of price values
        trends: List of trend dictionaries
        target_durations: List of trend durations to extract (default: [1, 2, 3, 4])
    
    Returns:
        Dictionary with duration as key and list of TVReturns as value
    """
    trend_VReturns_by_duration = {duration: [] for duration in target_durations}
    
    for trend in trends:
        start_idx = trend['start']  # m
        end_idx = trend['end']      # m + k
        duration = end_idx - start_idx  # k
        
        if duration < 1:  # Need at least 1 day duration
            continue
            
        # Only store TVReturns for trends of target durations
        if duration in target_durations:
            p_start = prices[start_idx]    # Pₘ
            p_end = prices[end_idx]        # Pₘ₊ₖ
            
            if p_start <= 0 or p_end <= 0:
                continue
                
            # Calculate TSₘᵏ := (log(Pₘ₊ₖ) - log(Pₘ)) / k
            log_p_start = np.log(p_start)
            log_p_end = np.log(p_end)
            trend_tvreturn = (log_p_end - log_p_start) / duration
            trend_VReturns_by_duration[duration].append(trend_tvreturn)
    
    # Convert lists to numpy arrays
    for duration in target_durations:
        trend_VReturns_by_duration[duration] = np.array(trend_VReturns_by_duration[duration])
    
    return trend_VReturns_by_duration

def extract_trend_VReturns_by_duration_and_index(all_prices_data, target_durations=[1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """Extract TVReturns grouped by duration for all indices."""
    trend_VReturns_by_duration_and_index = {}
    
    for index_name, price_series in all_prices_data.items():
        if price_series.empty:
            trend_VReturns_by_duration_and_index[index_name] = {duration: np.array([]) for duration in target_durations}
            continue
            
        prices = price_series.values
        trends = identify_trends(prices)
        trend_VReturns_by_duration = calculate_trend_VReturns_by_duration(prices, trends, target_durations)
        
        trend_VReturns_by_duration_and_index[index_name] = trend_VReturns_by_duration
    
    return trend_VReturns_by_duration_and_index

def calculate_treturns_by_duration(prices, trends, target_durations=[1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """Calculate TReturns grouped by trend duration.
    
    Based on the formula from the image: TRet_m(t) = log R_m(R) - log R_m(L)
    Where R_m(R) is the right price and R_m(L) is the left price of trend m.
    
    Args:
        prices: Array of price values
        trends: List of trend dictionaries
        target_durations: List of trend durations to extract
    
    Returns:
        Dictionary with duration as key and list of TReturns as value
    """
    treturns_by_duration = {duration: [] for duration in target_durations}
    
    for trend in trends:
        start_idx = trend['start']  # m (left)
        end_idx = trend['end']      # m + k (right)
        duration = end_idx - start_idx  # k
        
        if duration < 1:  # Need at least 1 day duration
            continue
            
        # Only store TReturns for trends of target durations
        if duration in target_durations:
            p_left = prices[start_idx]    # R_m(L)
            p_right = prices[end_idx]     # R_m(R)
            
            if p_left <= 0 or p_right <= 0:
                continue
                
            # Calculate TRet_m(t) := log(R_m(R)) - log(R_m(L))
            log_p_left = np.log(p_left)
            log_p_right = np.log(p_right)
            treturn = log_p_right - log_p_left
            treturns_by_duration[duration].append(treturn)
    
    # Convert lists to numpy arrays
    for duration in target_durations:
        treturns_by_duration[duration] = np.array(treturns_by_duration[duration])
    
    return treturns_by_duration

def extract_treturns_by_duration_and_index(all_prices_data, target_durations=[1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """Extract TReturns grouped by duration for all indices."""
    treturns_by_duration_and_index = {}
    
    for index_name, price_series in all_prices_data.items():
        if price_series.empty:
            treturns_by_duration_and_index[index_name] = {duration: np.array([]) for duration in target_durations}
            continue
            
        prices = price_series.values
        trends = identify_trends(prices)
        
        # Calculate TReturns grouped by duration
        treturns_by_duration_and_index[index_name] = calculate_treturns_by_duration(prices, trends, target_durations)
    
    return treturns_by_duration_and_index


def plot_combined_histograms_by_duration_4x3_log_y(VReturns_by_duration_and_index, trend_VReturns_by_duration_and_index, 
                                        treturns_by_duration_and_index,
                                        all_VReturns_by_index, all_trend_VReturns_by_index, all_treturns_by_index,
                                        save_path='TVReturns_Analysis_Plots', target_durations=[1, 2, 3, 4, 5, 6, 7, 8, 9]):
    """Generate a combined plot with 4 rows x 3 columns showing histograms of VReturns, TVReturns, and TReturns by duration.
    This version uses logarithmic scale on y-axis and linear scale on x-axis.
    
    Left column: Regular VReturns histograms
    Middle column: TVReturns histograms
    Right column: TReturns histograms
    Rows: DJIA, DAX, IPC, Nikkei (top to bottom)
    """
    
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Define colors for different durations
    duration_colors = {
        1: '#1f77b4',  # Blue
        2: '#ff7f0e',  # Orange  
        3: '#2ca02c',  # Green
        4: '#d62728',  # Red
        5: '#9467bd',  # Purple
        6: '#8c564b',  # Brown
        7: '#e377c2',  # Pink
        8: '#7f7f7f',  # Gray
        9: '#bcbd22'   # Olive
    }
    
    # Color for all data (no duration filtering)
    all_data_color = '#000000'  # Black
    
    index_names = ['DJIA', 'DAX', 'IPC', 'Nikkei']  # Order specified by user
    target_durations = [1, 3, 6, 9]
    
    # Create figure with 4 rows x 3 columns
    fig, axes = plt.subplots(4, 3, figsize=(24, 20))
    
    for i, index_name in enumerate(index_names):
        # Left column: Regular VReturns
        ax_left = axes[i, 0]
        # Middle column: TVReturns  
        ax_middle = axes[i, 1]
        # Right column: TReturns
        ax_right = axes[i, 2]
        
        # Process Regular VReturns (left column)
        if index_name in VReturns_by_duration_and_index:
            VReturns_by_duration = VReturns_by_duration_and_index[index_name]
            
            # Get all VReturns for this index to determine common x-axis range
            all_VReturns_for_index = []
            for duration in target_durations:
                if len(VReturns_by_duration[duration]) > 0:
                    all_VReturns_for_index.extend(VReturns_by_duration[duration])
            
            # Add all data (no duration filtering) to range calculation
            if index_name in all_VReturns_by_index and len(all_VReturns_by_index[index_name]) > 0:
                all_VReturns_for_index.extend(all_VReturns_by_index[index_name])
            
            if all_VReturns_for_index:
                # Calculate common bins for all histograms - show ALL data
                x_min, x_max = np.min(all_VReturns_for_index), np.max(all_VReturns_for_index)
                bins = np.linspace(x_min, x_max, 100)
                
                # Plot histogram for all data (no duration filtering) first
                if index_name in all_VReturns_by_index and len(all_VReturns_by_index[index_name]) > 0:
                    all_VReturns_data = all_VReturns_by_index[index_name]
                    ax_left.hist(all_VReturns_data, bins=bins, 
                               histtype='step',
                               color=all_data_color, 
                               linewidth=2.5, 
                               alpha=0.9,
                               label=f'All data (n={len(all_VReturns_data)})')
                
                # Plot histogram for each duration
                for duration in target_durations:
                    VReturns = VReturns_by_duration[duration]
                    
                    if len(VReturns) > 0:
                        ax_left.hist(VReturns, bins=bins, 
                                   histtype='step',
                                   color=duration_colors[duration], 
                                   linewidth=2.0, 
                                   alpha=0.8,
                                   label=f'{duration} day (n={len(VReturns)})')
                
                ax_left.set_xlabel('VReturns')
                ax_left.set_ylabel('Frequency (log)')
                ax_left.set_yscale('log')  # Set logarithmic scale on y-axis
                ax_left.set_title(f'{index_name} - VReturns')
                ax_left.legend(loc='upper right', fontsize=8)
                ax_left.grid(True, alpha=0.3)
            else:
                ax_left.set_title(f'{index_name} - VReturns')
                ax_left.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax_left.transAxes)
        else:
            ax_left.set_title(f'{index_name} - VReturns')
            ax_left.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax_left.transAxes)
        
        # Process TVReturns (middle column)
        if index_name in trend_VReturns_by_duration_and_index:
            trend_VReturns_by_duration = trend_VReturns_by_duration_and_index[index_name]
            
            # Get all TVReturns for this index to determine common x-axis range
            all_trend_VReturns_for_index = []
            for duration in target_durations:
                if len(trend_VReturns_by_duration[duration]) > 0:
                    all_trend_VReturns_for_index.extend(trend_VReturns_by_duration[duration])
            
            # Add all data (no duration filtering) to range calculation
            if index_name in all_trend_VReturns_by_index and len(all_trend_VReturns_by_index[index_name]) > 0:
                all_trend_VReturns_for_index.extend(all_trend_VReturns_by_index[index_name])
            
            if all_trend_VReturns_for_index:
                # Calculate common bins for all histograms - show ALL data
                x_min, x_max = np.min(all_trend_VReturns_for_index), np.max(all_trend_VReturns_for_index)
                bins = np.linspace(x_min, x_max, 100)
                
                # Plot histogram for all data (no duration filtering) first
                if index_name in all_trend_VReturns_by_index and len(all_trend_VReturns_by_index[index_name]) > 0:
                    all_trend_VReturns_data = all_trend_VReturns_by_index[index_name]
                    ax_middle.hist(all_trend_VReturns_data, bins=bins, 
                                 histtype='step',
                                 color=all_data_color, 
                                 linewidth=2.5, 
                                 alpha=0.9,
                                 label=f'All data (n={len(all_trend_VReturns_data)})')
                
                # Plot histogram for each duration
                for duration in target_durations:
                    trend_VReturns = trend_VReturns_by_duration[duration]
                    
                    if len(trend_VReturns) > 0:
                        ax_middle.hist(trend_VReturns, bins=bins, 
                                    histtype='step',
                                    color=duration_colors[duration], 
                                    linewidth=2.0, 
                                    alpha=0.8,
                                    label=f'{duration} day (n={len(trend_VReturns)})')
                
                ax_middle.set_xlabel('TVReturns')
                ax_middle.set_ylabel('Frequency (log)')
                ax_middle.set_yscale('log')  # Set logarithmic scale on y-axis
                ax_middle.set_title(f'{index_name} - TVReturns')
                ax_middle.legend(loc='upper right', fontsize=8)
                ax_middle.grid(True, alpha=0.3)
            else:
                ax_middle.set_title(f'{index_name} - TVReturns')
                ax_middle.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax_middle.transAxes)
        else:
            ax_middle.set_title(f'{index_name} - TVReturns')
            ax_middle.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax_middle.transAxes)
        
        # Process TReturns (right column)
        if index_name in treturns_by_duration_and_index:
            treturns_by_duration = treturns_by_duration_and_index[index_name]
            
            # Get all TReturns for this index to determine common x-axis range
            all_treturns_for_index = []
            for duration in target_durations:
                if len(treturns_by_duration[duration]) > 0:
                    all_treturns_for_index.extend(treturns_by_duration[duration])
            
            # Add all data (no duration filtering) to range calculation
            if index_name in all_treturns_by_index and len(all_treturns_by_index[index_name]) > 0:
                all_treturns_for_index.extend(all_treturns_by_index[index_name])
            
            if all_treturns_for_index:
                # Calculate common bins for all histograms - show ALL data
                x_min, x_max = np.min(all_treturns_for_index), np.max(all_treturns_for_index)
                bins = np.linspace(x_min, x_max, 100)
                
                # Plot histogram for all data (no duration filtering) first
                if index_name in all_treturns_by_index and len(all_treturns_by_index[index_name]) > 0:
                    all_treturns_data = all_treturns_by_index[index_name]
                    ax_right.hist(all_treturns_data, bins=bins, 
                                 histtype='step',
                                 color=all_data_color, 
                                 linewidth=2.5, 
                                 alpha=0.9,
                                 label=f'All data (n={len(all_treturns_data)})')
                
                # Plot histogram for each duration
                for duration in target_durations:
                    treturns = treturns_by_duration[duration]
                    
                    if len(treturns) > 0:
                        ax_right.hist(treturns, bins=bins, 
                                    histtype='step',
                                    color=duration_colors[duration], 
                                    linewidth=2.0, 
                                    alpha=0.8,
                                    label=f'{duration} day (n={len(treturns)})')
                
                ax_right.set_xlabel('TReturns')
                ax_right.set_ylabel('Frequency (log)')
                ax_right.set_yscale('log')  # Set logarithmic scale on y-axis
                ax_right.set_title(f'{index_name} - TReturns')
                ax_right.legend(loc='upper right', fontsize=8)
                ax_right.grid(True, alpha=0.3)
            else:
                ax_right.set_title(f'{index_name} - TReturns')
                ax_right.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax_right.transAxes)
        else:
            ax_right.set_title(f'{index_name} - TReturns')
            ax_right.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax_right.transAxes)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_filename = os.path.join(save_path, 'combined_histograms_by_duration_4x3_log_y.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nCombined histograms plot (Log Y) saved: {plot_filename}")
    print("=" * 80)
    
    return plot_filename


def calculate_all_VReturns(prices, trends):
    """Calculate all VReturns without duration filtering."""
    VReturns = []
    
    for trend in trends:
        start_idx = trend['start']
        end_idx = trend['end']
        duration = end_idx - start_idx
        
        if duration < 1:  # Need at least 1 day duration
            continue
            
        p_start = prices[start_idx]
        if p_start <= 0:
            continue
            
        log_p_start = np.log(p_start)
        
        # Calculate VReturns for this trend
        for i in range(1, duration + 1):
            price_idx = start_idx + i
            if price_idx < len(prices):
                p_i = prices[price_idx]
                if p_i > 0:
                    log_p_i = np.log(p_i)
                    tvreturn = (log_p_i - log_p_start) / i
                    VReturns.append(tvreturn)
    
    return np.array(VReturns)

def calculate_all_trend_VReturns(prices, trends):
    """Calculate all TVReturns without duration filtering."""
    trend_VReturns = []
    
    for trend in trends:
        start_idx = trend['start']
        end_idx = trend['end']
        duration = end_idx - start_idx
        
        if duration < 1:  # Need at least 1 day duration
            continue
            
        p_start = prices[start_idx]
        p_end = prices[end_idx]
        
        if p_start <= 0 or p_end <= 0:
            continue
            
        log_p_start = np.log(p_start)
        log_p_end = np.log(p_end)
        
        # Calculate TVReturn for this trend
        tvreturn = (log_p_end - log_p_start) / duration
        trend_VReturns.append(tvreturn)
    
    return np.array(trend_VReturns)

def calculate_all_treturns(prices, trends):
    """Calculate all TReturns without duration filtering."""
    treturns = []
    
    for trend in trends:
        start_idx = trend['start']
        end_idx = trend['end']
        
        p_start = prices[start_idx]
        p_end = prices[end_idx]
        
        if p_start <= 0 or p_end <= 0:
            continue
            
        log_p_start = np.log(p_start)
        log_p_end = np.log(p_end)
        
        # Calculate TReturn for this trend
        treturn = log_p_end - log_p_start
        treturns.append(treturn)
    
    return np.array(treturns)

def extract_all_data_by_index(all_prices_data):
    """Extract all VReturns, TVReturns, and TReturns without duration filtering for all indices."""
    all_VReturns_by_index = {}
    all_trend_VReturns_by_index = {}
    all_treturns_by_index = {}
    
    for index_name, price_series in all_prices_data.items():
        if price_series.empty:
            all_VReturns_by_index[index_name] = np.array([])
            all_trend_VReturns_by_index[index_name] = np.array([])
            all_treturns_by_index[index_name] = np.array([])
            continue
            
        prices = price_series.values
        trends = identify_trends(prices)
        
        all_VReturns = calculate_all_VReturns(prices, trends)
        all_trend_VReturns = calculate_all_trend_VReturns(prices, trends)
        all_treturns = calculate_all_treturns(prices, trends)
        
        all_VReturns_by_index[index_name] = all_VReturns
        all_trend_VReturns_by_index[index_name] = all_trend_VReturns
        all_treturns_by_index[index_name] = all_treturns
    
    return all_VReturns_by_index, all_trend_VReturns_by_index, all_treturns_by_index

def main():
    """Main function to execute the VReturns analysis by duration."""
    print("Loading financial data...")
    
    # Load financial data
    all_prices_data, all_prices_raw = download_financial_data()
    
    if not all_prices_data:
        print("Error: No financial data available.")
        return
    
    target_durations = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Up to 9 days
    
    # Extract all data (without duration filtering) for all indices
    print("Calculating returns...")
    all_VReturns_by_index, all_trend_VReturns_by_index, all_treturns_by_index = extract_all_data_by_index(all_prices_data)
    
    # Extract VReturns grouped by duration for all indices
    VReturns_by_duration_and_index = extract_VReturns_by_duration_and_index(all_prices_data, target_durations)
    
    # Extract TVReturns grouped by duration for all indices
    trend_VReturns_by_duration_and_index = extract_trend_VReturns_by_duration_and_index(all_prices_data, target_durations)
    
    # Extract TReturns grouped by duration for all indices
    treturns_by_duration_and_index = extract_treturns_by_duration_and_index(all_prices_data, target_durations)
    
    # Generate combined histograms plot with logarithmic y-axis
    print("Generating plot...")
    combined_plot_log_y_filename = plot_combined_histograms_by_duration_4x3_log_y(VReturns_by_duration_and_index, trend_VReturns_by_duration_and_index, treturns_by_duration_and_index, all_VReturns_by_index, all_trend_VReturns_by_index, all_treturns_by_index)
    
    print(f"Done! Plot saved: {combined_plot_log_y_filename}")

if __name__ == "__main__":
    main()
