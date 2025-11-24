import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis
import seaborn as sns
import warnings
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

warnings.filterwarnings('ignore')

# Global list to collect all statistics
GLOBAL_STATISTICS = []

# Plot style configuration
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
})

# ===== TREND ANALYSIS FUNCTIONS =====

def identify_trends(prices):
    """Identify ascending and descending trends in price series."""
    trends = []
    i = 0
    while i < len(prices) - 1:
        j = i + 1
        while j < len(prices) and prices[j] >= prices[j-1]:
            j += 1
        if j - 1 > i:
            trends.append({'start': i, 'end': j - 1, 'direction': 1})
            i = j - 1
            continue
        k = i + 1
        while k < len(prices) and prices[k] <= prices[k-1]:
            k += 1
        if k - 1 > i:
            trends.append({'start': i, 'end': k - 1, 'direction': -1})
            i = k - 1
            continue
        i += 1
    return trends

def enrich_trend_data(prices, trends_list):
    """Enrich trend data with duration, returns, and velocity metrics."""
    enriched_trends = []
    for trend in trends_list:
        start_idx, end_idx = trend['start'], trend['end']
        duration = end_idx - start_idx
        if duration == 0:
            continue
        p_start, p_end = prices[start_idx], prices[end_idx]
        if p_start > 1e-9 and p_end > 1e-9:
            log_r = np.log(p_end) - np.log(p_start)
            abs_r = abs(log_r)
            velocity = log_r / duration
            abs_velocity = abs(velocity)
            trend_info = {
                'start': start_idx, 'end': end_idx,
                'duration': float(duration), 'direction': trend['direction'],
                'log_return': float(log_r), 'abs_return': float(abs_r),
                'velocity': float(velocity), 'abs_velocity': float(abs_velocity)
            }
            enriched_trends.append(trend_info)
    return enriched_trends

def load_experiment_repetitions(experiment_folder):
    """
    Load all repetitions for a single experiment.
    
    Args:
        experiment_folder (str): Path to experiment folder containing rep_X subfolders
        
    Returns:
        list: List of DataFrames, one per repetition (full DataFrame, not just price series)
    """
    repetitions_data = []
    
    # Look for rep_X folders
    rep_folders = []
    for item in os.listdir(experiment_folder):
        if item.startswith('rep_') and os.path.isdir(os.path.join(experiment_folder, item)):
            rep_folders.append(item)
    
    # Sort to ensure consistent ordering
    rep_folders.sort(key=lambda x: int(x.split('_')[1]))
    
    for rep_folder in rep_folders:
        csv_path = os.path.join(experiment_folder, rep_folder, 'price_series.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, index_col=0)
                repetitions_data.append(df)  # Store full DataFrame instead of just price series
            except Exception as e:
                return None
    
    if len(repetitions_data) != 20:
        return None
    
    return repetitions_data

def calculate_log_returns(price_series):
    """Calculate logarithmic returns from price series."""
    return np.log(price_series[1:].values / price_series[:-1].values)

def calculate_statistics_with_errors(data, bootstrap_samples=1000):
    """
    Calculate comprehensive statistics with bootstrap error estimates.
    
    Args:
        data (array): Data to analyze
        bootstrap_samples (int): Number of bootstrap samples for error estimation
        
    Returns:
        dict: Dictionary with statistics and their errors
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if len(data) == 0:
        return {
            'mean': np.nan, 'mean_error': np.nan,
            'median': np.nan, 'median_error': np.nan,
            'std': np.nan, 'std_error': np.nan,
            'skewness': np.nan, 'skewness_error': np.nan,
            'kurtosis': np.nan, 'kurtosis_error': np.nan,
            'count': 0
        }
    
    n = len(data)
    
    # Original statistics
    mean_val = np.mean(data)
    median_val = np.median(data)
    std_val = np.std(data, ddof=1) if n > 1 else np.nan
    skew_val = skew(data)
    kurtosis_val = kurtosis(data)
    
    # Bootstrap for error estimation
    bootstrap_means = []
    bootstrap_medians = []
    bootstrap_stds = []
    bootstrap_skews = []
    bootstrap_kurtosis = []
    
    for _ in range(bootstrap_samples):
        try:
            # Generate a bootstrap sample with replacement
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            if len(bootstrap_sample) > 0:
                bootstrap_means.append(np.mean(bootstrap_sample))
                bootstrap_medians.append(np.median(bootstrap_sample))
                bootstrap_stds.append(np.std(bootstrap_sample, ddof=1) if n > 1 else np.nan)
                bootstrap_skews.append(skew(bootstrap_sample))
                bootstrap_kurtosis.append(kurtosis(bootstrap_sample))
        except:
            continue
    
    # Calculate errors as standard deviations of bootstrap distributions
    mean_error = np.std(bootstrap_means) if bootstrap_means else np.nan
    median_error = np.std(bootstrap_medians) if bootstrap_medians else np.nan
    std_error = np.std(bootstrap_stds) if bootstrap_stds and len(np.unique(bootstrap_stds)) > 1 else np.nan
    skew_error = np.std(bootstrap_skews) if bootstrap_skews else np.nan
    kurtosis_error = np.std(bootstrap_kurtosis) if bootstrap_kurtosis else np.nan
    
    return {
        'mean': mean_val, 'mean_error': mean_error,
        'median': median_val, 'median_error': median_error,
        'std': std_val, 'std_error': std_error,
        'skewness': skew_val, 'skewness_error': skew_error,
        'kurtosis': kurtosis_val, 'kurtosis_error': kurtosis_error,
        'count': n
    }

def generate_and_save_stats_tables(output_folder, experiment_name):
    """
    Generate and save a CSV and LaTeX table from the collected statistics.
    """
    global GLOBAL_STATISTICS
    
    if not GLOBAL_STATISTICS:
        return
        
    df_stats = pd.DataFrame(GLOBAL_STATISTICS)
    
    # Create the filename
    csv_filename = os.path.join(output_folder, f'{experiment_name}_statistics.csv')
    
    # Ensure value and error columns are numeric, handle any string values
    def safe_numeric_convert(val):
        """Safely convert values to numeric, handling strings and other types."""
        try:
            if isinstance(val, (int, float, np.integer, np.floating)):
                return float(val)
            elif isinstance(val, str):
                # Try to extract numeric value from string
                import re
                numeric_match = re.search(r'-?\d+\.?\d*', val)
                if numeric_match:
                    return float(numeric_match.group())
                else:
                    return np.nan
            else:
                return float(val) if val is not None else np.nan
        except (ValueError, TypeError):
            return np.nan
    
    # Convert value and error columns to numeric
    df_stats['value'] = df_stats['value'].apply(safe_numeric_convert)
    df_stats['error'] = df_stats['error'].apply(safe_numeric_convert)
    
    # Combine value and error into a single column for the CSV
    def format_value_with_error(row):
        """Safely format value with error, handling NaN values."""
        try:
            val = row['value']
            err = row['error']
            if pd.notna(val):
                if pd.notna(err):
                    return f"{val:.4f} \u00b1 {err:.4f}"
                else:
                    return f"{val:.4f}"
            else:
                return "N/A"
        except (ValueError, TypeError):
            return "N/A"
    
    df_stats['value_with_error'] = df_stats.apply(format_value_with_error, axis=1)
    
    # Save to CSV using the correct encoding
    df_stats.to_csv(csv_filename, index=False, encoding='utf-8-sig')

def collect_statistics(experiment_name, plot_number, subplot, data_name, data):
    """
    Collects statistics for a given dataset and adds to the global list.
    """
    # Handle scalar values by converting to array
    if np.isscalar(data):
        data = np.array([data])
    elif not isinstance(data, (list, np.ndarray)):
        data = np.array([data])
    
    # Convert values to ensure numeric type
    try:
        # Handle different input types
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        elif not isinstance(data, np.ndarray):
            data = np.array([data])
        
        # Filter out non-numeric values and convert to float
        numeric_values = []
        for val in data.flatten():
            try:
                if isinstance(val, (int, float, np.integer, np.floating)):
                    numeric_values.append(float(val))
                elif isinstance(val, str):
                    # Try to extract numeric value from string
                    import re
                    numeric_match = re.search(r'-?\d+\.?\d*', val)
                    if numeric_match:
                        numeric_values.append(float(numeric_match.group()))
                    # Skip non-numeric strings
                elif val is not None:
                    numeric_values.append(float(val))
            except (ValueError, TypeError):
                # Skip values that can't be converted to numeric
                continue
        
        if not numeric_values:
            return
        
        data = np.array(numeric_values, dtype=float)
        
    except Exception as e:
        return
    
    stats_dict = calculate_statistics_with_errors(data)
    for stat_name, stat_value in stats_dict.items():
        if stat_name.endswith('_error') or stat_name == 'count':
            continue
        
        error_key = f"{stat_name}_error"
        error_value = stats_dict.get(error_key, np.nan)
        
        GLOBAL_STATISTICS.append({
            'experiment': experiment_name,
            'plot_number': plot_number,
            'subplot': subplot if subplot is not None else 'N/A',
            'data_name': data_name,
            'statistic': stat_name,
            'value': float(stat_value) if not np.isnan(stat_value) else np.nan,
            'error': float(error_value) if not np.isnan(error_value) else np.nan
        })


def calculate_net_demand_autocorr(repetitions_data):
    """
    Calculate first-order autocorrelation of net demand across repetitions.
    
    Args:
        repetitions_data (list): List of DataFrames, each containing data from one repetition
        
    Returns:
        dict: Dictionary containing autocorrelation statistics
    """
    autocorr_values = []
    
    for rep_data in repetitions_data:
        if 'uniform_noise' in rep_data.columns:
            net_demand = rep_data['uniform_noise'].values
            
            # Calculate first-order autocorrelation manually
            if len(net_demand) > 1:
                # Remove NaN values if any
                net_demand = net_demand[~np.isnan(net_demand)]
                
                if len(net_demand) > 1:
                    # Calculate correlation between D_t and D_{t+1}
                    d_t = net_demand[:-1]  # D_t (all except last)
                    d_t_plus_1 = net_demand[1:]  # D_{t+1} (all except first)
                    
                    # Calculate Pearson correlation coefficient with robust error handling
                    if len(d_t) > 0 and np.std(d_t) > 1e-10 and np.std(d_t_plus_1) > 1e-10:
                        try:
                            corr_matrix = np.corrcoef(d_t, d_t_plus_1)
                            if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2):
                                rho_1 = corr_matrix[0, 1]
                                if not np.isnan(rho_1) and not np.isinf(rho_1):
                                    autocorr_values.append(rho_1)
                        except Exception as e:
                            # Skip this repetition if correlation calculation fails
                            continue
    
    if len(autocorr_values) == 0:
        return {
            'mean': 0,
            'std': 0,
            'values': [],
            'abs_mean': 0,
            'max_abs': 0
        }
    
    autocorr_array = np.array(autocorr_values)
    
    return {
        'mean': np.mean(autocorr_array),
        'std': np.std(autocorr_array),
        'values': autocorr_values,
        'abs_mean': np.mean(np.abs(autocorr_array)),
        'max_abs': np.max(np.abs(autocorr_array))
    }

def average_point_by_point(repetitions_data):
    """
    Average multiple repetitions point by point.
    
    Args:
        repetitions_data (list): List of DataFrames (extract price series)
        
    Returns:
        tuple: (averaged_series, std_series) for confidence intervals
    """
    # Extract price series from DataFrames
    price_series_list = [df['market_price'] for df in repetitions_data]
    
    # Convert to numpy array for easier manipulation
    min_length = min(len(series) for series in price_series_list)
    
    # Truncate all series to same length
    truncated_data = np.array([series[:min_length].values for series in price_series_list])
    
    # Calculate point-wise mean and std
    mean_series = np.mean(truncated_data, axis=0)
    std_series = np.std(truncated_data, axis=0)
    
    return mean_series, std_series

def plot_1_volatility_clustering(repetitions_data, output_folder, exp_name):
    """
    Plot 1: Volatility clustering (absolute returns) averaged point by point across 20 repetitions.
    """
    
    # Calculate absolute log returns for each repetition
    abs_log_returns_list = []
    for rep_data in repetitions_data:
        price_series = rep_data['market_price']
        log_ret = calculate_log_returns(price_series)
        abs_log_returns_list.append(np.abs(log_ret))
    
    # Average point by point
    min_length = min(len(lr) for lr in abs_log_returns_list)
    truncated_abs_returns = np.array([lr[:min_length] for lr in abs_log_returns_list])
    avg_abs_returns = np.mean(truncated_abs_returns, axis=0)
    
    # Collect statistics for Plot 1
    collect_statistics(exp_name, 1, 'N/A', 'Avg_Absolute_Returns', avg_abs_returns)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(avg_abs_returns, linewidth=1, alpha=0.8)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Absolute Returns')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'plot1_volatility_clustering.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_2_stylized_facts(repetitions_data, output_folder, exp_name):
    """
    Plot 2: Stylized facts with 4 subplots matching analyze_stylized_facts style.
    """
    
    # Calculate log returns for each repetition
    log_returns_list = []
    abs_log_returns_list = []
    
    for rep_data in repetitions_data:
        price_series = rep_data['market_price']
        log_ret = calculate_log_returns(price_series)
        log_returns_list.append(log_ret)
        abs_log_returns_list.append(np.abs(log_ret))
    
    # Subplot 1: Returns distribution (pooling)
    pooled_returns = np.concatenate(log_returns_list)
    collect_statistics(exp_name, 2, '1_Returns_Distribution', 'Pooled_Log_Returns', pooled_returns)
    
    # Subplot 2: Logarithmic returns (average log returns point by point)
    min_length_reg = min(len(lr) for lr in log_returns_list)
    truncated_returns = np.array([lr[:min_length_reg] for lr in log_returns_list])
    avg_log_returns = np.mean(truncated_returns, axis=0)
    collect_statistics(exp_name, 2, '2_Logarithmic_Returns', 'Averaged_Log_Returns', avg_log_returns)
    
    # Prepare returns for ACF calculation
    avg_returns = avg_log_returns
    
    # Subplot 3: Returns Autocorrelation
    collect_statistics(exp_name, 2, '3_Returns_ACF', 'Point_Averaged_Returns', avg_returns)
    
    # Subplot 4: Absolute Returns Autocorrelation
    # Recalculate avg_abs_returns for subplot 4
    min_length_abs = min(len(lr) for lr in abs_log_returns_list)
    truncated_abs_returns_acf = np.array([lr[:min_length_abs] for lr in abs_log_returns_list])
    avg_abs_returns_acf = np.mean(truncated_abs_returns_acf, axis=0)
    collect_statistics(exp_name, 2, '4_Abs_Returns_ACF', 'Point_Averaged_Abs_Returns', avg_abs_returns_acf)
    
    # Create subplots with same style as analyze_stylized_facts
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Use seaborn color palette
    plot_color = sns.color_palette()[0]
    
    # 1. Returns Distribution (matching analyze_stylized_facts)
    # Get the histogram data manually to plot points
    hist_counts, bin_edges = np.histogram(pooled_returns, bins=100, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    axs[0, 0].plot(bin_centers, hist_counts, 'o', color=plot_color, label='Empirical Distribution', markersize=4)
    
    # Add normal distribution overlay scaled to match frequency counts
    xmin, xmax = axs[0, 0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, pooled_returns.mean(), pooled_returns.std())
    # Scale the normal distribution to match the frequency counts
    bin_width = bin_edges[1] - bin_edges[0]
    total_count = len(pooled_returns)
    p_scaled = p * total_count * bin_width
    axs[0, 0].plot(x, p_scaled, 'k', linewidth=2, label='Normal Distribution')
    
    # Apply log scale to y-axis
    axs[0, 0].set_yscale('log')
    
    # Calculate statistics with errors for the statistics box
    pooled_stats = calculate_statistics_with_errors(pooled_returns)
    stats_text = (f'Mean: {pooled_stats["mean"]:.4f} ± {pooled_stats["mean_error"]:.4f}\n'
                  f'Std Dev: {pooled_stats["std"]:.4f}\n'
                  f'Skewness: {pooled_stats["skewness"]:.4f} ± {pooled_stats["skewness_error"]:.4f}\n'
                  f'Kurtosis: {pooled_stats["kurtosis"]:.4f} ± {pooled_stats["kurtosis_error"]:.4f}')
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8)
    axs[0, 0].text(0.05, 0.95, stats_text, transform=axs[0, 0].transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    axs[0, 0].set_title('(a) Returns Distribution')
    axs[0, 0].set_xlabel('Logarithmic Returns')
    axs[0, 0].set_ylabel('Frequency (log)')
    axs[0, 0].legend()
    
    # 2. Logarithmic Returns
    axs[0, 1].plot(range(len(avg_log_returns)), avg_log_returns, color=plot_color, alpha=0.7)
    axs[0, 1].set_title('(b) Logarithmic Returns')
    axs[0, 1].set_xlabel('Time Steps')
    axs[0, 1].set_ylabel('Logarithmic Returns')
    
    # 3. Returns Autocorrelation (using statsmodels plot_acf)
    plot_acf(avg_returns, lags=40, ax=axs[1, 0], title='(c) Returns Autocorrelation')
    axs[1, 0].set_xlabel('Lag')
    axs[1, 0].set_ylabel('Autocorrelation')
    
    # 4. Absolute Returns Autocorrelation (using statsmodels plot_acf)
    plot_acf(avg_abs_returns_acf, lags=40, ax=axs[1, 1], title='(d) Absolute Returns Autocorrelation')
    axs[1, 1].set_xlabel('Lag')
    axs[1, 1].set_ylabel('Autocorrelation')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_folder, 'plot2_stylized_facts.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_3_treturns(repetitions_data, output_folder, exp_name):
    """
    Plot 3: T-returns (multiescale returns) with trend identification and pooling.
    Matching visual style of plot_2_stylized_facts.
    """
    
    all_treturns = []
    
    # For each repetition, identify trends and calculate treturns
    for i, rep_data in enumerate(repetitions_data):
        prices = rep_data['market_price'].values
        
        # Identify trends using existing function
        trends = identify_trends(prices)
        
        # Enrich trend data to get log returns (treturns)
        enriched_trends = enrich_trend_data(prices, trends)
        
        # Extract treturns (log_return from each trend)
        treturns_rep = [trend['log_return'] for trend in enriched_trends]
        
        # Extend the pooled list
        all_treturns.extend(treturns_rep)
    
    if len(all_treturns) == 0:
        return
    
    # Calculate ACF of pooled treturns
    all_treturns = np.array(all_treturns)
    if len(all_treturns) > 10:  # Need sufficient data for ACF
        acf_treturns = acf(all_treturns, nlags=min(40, len(all_treturns)//4), fft=True)
    else:
        return
    
    # Subplot 1: T-returns Distribution
    collect_statistics(exp_name, 3, '1_T_Returns_Distribution', 'Pooled_T_Returns', all_treturns)

    # Subplot 2: ACF of T-returns
    collect_statistics(exp_name, 3, '2_T_Returns_ACF', 'Pooled_T_Returns', all_treturns)
    
    # Create plot with same style as plot_2_stylized_facts
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))  # Adjusted for 2 subplots
    
    # Use seaborn color palette (same as plot_2)
    plot_color = sns.color_palette()[0]
    
    # 1. T-returns Distribution
    hist_counts, bin_edges = np.histogram(all_treturns, bins=50, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    axs[0].plot(bin_centers, hist_counts, 'o', color=plot_color, markersize=4, label='Empirical Distribution')
    
    # Apply log scale to y-axis
    axs[0].set_yscale('log')
    
    # Calculate statistics with errors for the statistics box
    treturns_stats = calculate_statistics_with_errors(all_treturns)
    stats_text = (f'Mean: {treturns_stats["mean"]:.4f} ± {treturns_stats["mean_error"]:.4f}\n'
                  f'Std Dev: {treturns_stats["std"]:.4f}\n'
                  f'Skewness: {treturns_stats["skewness"]:.4f} ± {treturns_stats["skewness_error"]:.4f}\n'
                  f'Kurtosis: {treturns_stats["kurtosis"]:.4f} ± {treturns_stats["kurtosis_error"]:.4f}')
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8)
    axs[0].text(0.05, 0.95, stats_text, transform=axs[0].transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
    axs[0].set_title('(a) TReturns Distribution')
    axs[0].set_xlabel('TReturns')
    axs[0].set_ylabel('Frequency (log)')
    axs[0].legend()
    
    # 2. ACF of T-returns (using statsmodels plot_acf without confidence intervals)
    plot_acf(all_treturns, lags=400, ax=axs[1], title='(b) TReturns Autocorrelation', alpha=None)
    axs[1].set_xlabel('Lag')
    axs[1].set_ylabel('Autocorrelation')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_folder, 'plot3_treturns.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_price_sign_autocorr(repetitions_data):
    """
    Calculate first-order autocorrelation of price change signs across repetitions.
    
    Args:
        repetitions_data (list): List of DataFrames, each containing data from one repetition
        
    Returns:
        dict: Dictionary containing autocorrelation statistics for price signs
    """
    autocorr_values = []
    
    for rep_data in repetitions_data:
        if 'market_price' in rep_data.columns:
            prices = rep_data['market_price'].values
            
            # Calculate price changes
            if len(prices) > 2:
                price_changes = np.diff(prices)
                
                # Calculate signs of price changes
                signs = np.sign(price_changes)
                
                # Remove zero changes (if any) for cleaner correlation
                non_zero_mask = signs != 0
                if np.sum(non_zero_mask) > 1:
                    signs_clean = signs[non_zero_mask]
                    
                    if len(signs_clean) > 1:
                        # Calculate correlation between sign(Δt) and sign(Δt+1)
                        sign_t = signs_clean[:-1]  # sign(Δt)
                        sign_t_plus_1 = signs_clean[1:]  # sign(Δt+1)
                        
                        # Calculate Pearson correlation coefficient with robust error handling
                        if len(sign_t) > 0 and np.std(sign_t) > 1e-10 and np.std(sign_t_plus_1) > 1e-10:
                            try:
                                corr_matrix = np.corrcoef(sign_t, sign_t_plus_1)
                                if corr_matrix.ndim == 2 and corr_matrix.shape == (2, 2):
                                    rho_signs = corr_matrix[0, 1]
                                    if not np.isnan(rho_signs) and not np.isinf(rho_signs):
                                        autocorr_values.append(rho_signs)
                            except Exception as e:
                                # Skip this repetition if correlation calculation fails
                                continue
    
    if len(autocorr_values) == 0:
        return {
            'mean': 0,
            'std': 0,
            'values': [],
            'abs_mean': 0,
            'max_abs': 0
        }
    
    autocorr_array = np.array(autocorr_values)
    
    return {
        'mean': np.mean(autocorr_array),
        'std': np.std(autocorr_array),
        'values': autocorr_values,
        'abs_mean': np.mean(np.abs(autocorr_array)),
        'max_abs': np.max(np.abs(autocorr_array))
    }

def plot_5_net_demand_autocorr(repetitions_data, output_folder, exp_name):
    """
    Generate plot for first-order autocorrelation of net demand.
    Theoretical expectation: |ρ₁| ≈ 0.05-0.08 (structural negative correlation)
    """
    
    # Calculate autocorrelation statistics
    autocorr_stats = calculate_net_demand_autocorr(repetitions_data)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Histogram of autocorrelation values
    if len(autocorr_stats['values']) > 0:
        ax1.hist(autocorr_stats['values'], bins=15, alpha=0.7, color='steelblue', 
                edgecolor='black', density=True)
        ax1.axvline(autocorr_stats['mean'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {autocorr_stats["mean"]:.4f}')
        ax1.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        ax1.set_xlabel(r'$\rho_1^D = \text{Corr}(D_t, D_{t+1})$', fontsize=11)
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of First-Order Autocorrelation\nof Net Demand Across Repetitions')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f'Mean: {autocorr_stats["mean"]:.4f}\n'
        stats_text += f'Std: {autocorr_stats["std"]:.4f}\n'
        stats_text += f'|Mean|: {autocorr_stats["abs_mean"]:.4f}\n'
        stats_text += f'Max |ρ₁|: {autocorr_stats["max_abs"]:.4f}\n'
        stats_text += f'N reps: {len(autocorr_stats["values"])}'
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='none', edgecolor='gray', alpha=1.0),
                fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No valid data for autocorrelation calculation', 
                transform=ax1.transAxes, ha='center', va='center')
        ax1.set_title('Distribution of First-Order Autocorrelation\nof Net Demand Across Repetitions')
    
    # Subplot 2: Time series of autocorrelation values
    if len(autocorr_stats['values']) > 0:
        repetition_numbers = range(1, len(autocorr_stats['values']) + 1)
        ax2.plot(repetition_numbers, autocorr_stats['values'], 'o-', color='steelblue', 
                markersize=4, linewidth=1)
        ax2.axhline(autocorr_stats['mean'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {autocorr_stats["mean"]:.4f}')
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # Add theoretical expectation band
        ax2.axhspan(-0.08, -0.05, alpha=0.2, color='orange', 
                   label='Theoretical range')
        
        ax2.set_xlabel('Repetition Number')
        ax2.set_ylabel(r'$\rho_1^D = \text{Corr}(D_t, D_{t+1})$', fontsize=11)
        ax2.set_title('First-Order Autocorrelation by Repetition')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Set y-axis limits to better show the values
        y_range = max(0.12, 1.2 * max(abs(min(autocorr_stats['values'])), 
                                     abs(max(autocorr_stats['values']))))
        ax2.set_ylim(-y_range, y_range)
    else:
        ax2.text(0.5, 0.5, 'No valid data for autocorrelation calculation', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('First-Order Autocorrelation by Repetition')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(output_folder, f'{exp_name}_plot_5_net_demand_autocorr.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Collect statistics
    if len(autocorr_stats['values']) > 0:
        collect_statistics(exp_name, 5, 'histogram', 'autocorr_mean', autocorr_stats['mean'])
        collect_statistics(exp_name, 5, 'histogram', 'autocorr_std', autocorr_stats['std'])
        collect_statistics(exp_name, 5, 'histogram', 'autocorr_abs_mean', autocorr_stats['abs_mean'])
        collect_statistics(exp_name, 5, 'histogram', 'autocorr_max_abs', autocorr_stats['max_abs'])
        collect_statistics(exp_name, 5, 'time_series', 'num_repetitions', len(autocorr_stats['values']))
    


def plot_6_price_sign_autocorr(repetitions_data, output_folder, exp_name):
    """
    Generate plot for first-order autocorrelation of price change signs.
    This is the critical test for the theoretical prediction of temporal independence.
    
    Theoretical validity thresholds:
    - Excellent approximation: |ρ₁^signs| < 0.02 (R > 1.5)
    - Good approximation: |ρ₁^signs| ≤ 0.03 (R ≥ 1)
    - Questionable: |ρ₁^signs| > 0.03 (R < 1)
    """
    
    # Calculate autocorrelation statistics for price signs
    sign_autocorr_stats = calculate_price_sign_autocorr(repetitions_data)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Histogram of sign autocorrelation values
    if len(sign_autocorr_stats['values']) > 0:
        ax1.hist(sign_autocorr_stats['values'], bins=15, alpha=0.7, color='darkgreen', 
                edgecolor='black', density=True)
        ax1.axvline(sign_autocorr_stats['mean'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {sign_autocorr_stats["mean"]:.4f}')
        ax1.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # Add theoretical threshold zones
        
        ax1.set_xlabel(r'$\rho_1^{signs} = \text{Corr}(\text{sign}(\Delta_t), \text{sign}(\Delta_{t+1}))$', fontsize=10)
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of First-Order Autocorrelation\nof Price Change Signs Across Repetitions')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Add text box with statistics and regime classification
        abs_mean = sign_autocorr_stats["abs_mean"]
        stats_text = f'Mean: {sign_autocorr_stats["mean"]:.4f}\n'
        stats_text += f'Std: {sign_autocorr_stats["std"]:.4f}\n'
        stats_text += f'|Mean|: {abs_mean:.4f}\n'
        stats_text += f'Max |ρ|: {sign_autocorr_stats["max_abs"]:.4f}\n'
        stats_text += f'N reps: {len(sign_autocorr_stats["values"])}\n\n'
        
        # Classify regime - only show R value
        if abs_mean < 0.02:
            stats_text += 'R > 1.5'
        elif abs_mean <= 0.03:
            stats_text += '1 < R < 1.5'
        else:
            stats_text += 'R < 1'
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='none', edgecolor='gray', alpha=1.0),
                fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No valid data for sign autocorrelation calculation', 
                transform=ax1.transAxes, ha='center', va='center')
        ax1.set_title('Distribution of First-Order Autocorrelation\nof Price Change Signs Across Repetitions')
    
    # Subplot 2: Time series of sign autocorrelation values
    if len(sign_autocorr_stats['values']) > 0:
        repetition_numbers = range(1, len(sign_autocorr_stats['values']) + 1)
        ax2.plot(repetition_numbers, sign_autocorr_stats['values'], 'o-', color='darkgreen', 
                markersize=4, linewidth=1)
        ax2.axhline(sign_autocorr_stats['mean'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {sign_autocorr_stats["mean"]:.4f}')
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # Add theoretical threshold zones
        ax2.axhspan(-0.02, 0.02, alpha=0.15, color='green', label='Excellent region')
        ax2.axhline(0.03, color='orange', linestyle=':', alpha=0.7, linewidth=2, 
                   label='Good boundary |ρ| = 0.03')
        ax2.axhline(-0.03, color='orange', linestyle=':', alpha=0.7, linewidth=2)
        
        ax2.set_xlabel('Repetition Number')
        ax2.set_ylabel(r'$\rho_1^{signs} = \text{Corr}(\text{sign}(\Delta_t), \text{sign}(\Delta_{t+1}))$', fontsize=10)
        ax2.set_title('Price Change Signs Autocorrelation by Repetition')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Set y-axis limits
        y_range = max(0.05, 1.3 * max(abs(min(sign_autocorr_stats['values'])), 
                                     abs(max(sign_autocorr_stats['values']))))
        ax2.set_ylim(-y_range, y_range)
        
    else:
        ax2.text(0.5, 0.5, 'No valid data for sign autocorrelation calculation', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Price Change Signs Autocorrelation by Repetition')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.join(output_folder, f'{exp_name}_plot_6_price_sign_autocorr.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Collect statistics
    if len(sign_autocorr_stats['values']) > 0:
        collect_statistics(exp_name, 6, 'histogram', 'sign_autocorr_mean', sign_autocorr_stats['mean'])
        collect_statistics(exp_name, 6, 'histogram', 'sign_autocorr_std', sign_autocorr_stats['std'])
        collect_statistics(exp_name, 6, 'histogram', 'sign_autocorr_abs_mean', sign_autocorr_stats['abs_mean'])
        collect_statistics(exp_name, 6, 'histogram', 'sign_autocorr_max_abs', sign_autocorr_stats['max_abs'])
        collect_statistics(exp_name, 6, 'time_series', 'num_repetitions', len(sign_autocorr_stats['values']))
        
        # Regime classification
        abs_mean = sign_autocorr_stats['abs_mean']
        if abs_mean < 0.02:
            regime = 'excellent'
        elif abs_mean <= 0.03:
            regime = 'good'
        else:
            regime = 'questionable'
        
        collect_statistics(exp_name, 6, 'validation', 'regime', regime)

def plot_4_price_series_with_ci(repetitions_data, output_folder, exp_name):
    """
    Plot 4: Price series averaged point by point with confidence intervals.
    """
    
    # Calculate point-wise average and std
    avg_prices, std_prices = average_point_by_point(repetitions_data)
    
    # Calculate 95% confidence interval
    n_reps = len(repetitions_data)
    sem_prices = std_prices / np.sqrt(n_reps)  # Standard error of mean
    ci_95 = 1.96 * sem_prices
    
    # Collect statistics for Plot 4
    collect_statistics(exp_name, 4, 'N/A', 'Point_Averaged_Prices', avg_prices)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    time_steps = range(len(avg_prices))
    
    # Plot average line
    ax.plot(time_steps, avg_prices, linewidth=2, color='blue', label='Average Price')
    
    # Plot confidence interval
    ax.fill_between(time_steps, 
                    avg_prices - ci_95, 
                    avg_prices + ci_95,
                    alpha=0.3, color='lightblue', label='95% CI')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'plot4_price_series_ci.png'), dpi=300, bbox_inches='tight')
    plt.close()

def process_single_experiment(experiment_folder, experiment_name):
    """
    Process a single experiment: load repetitions and generate all plots.
    
    Args:
        experiment_folder (str): Path to experiment folder
        experiment_name (str): Name of experiment for plot titles
        
    Returns:
        bool: True if successful, False otherwise
    """
    global GLOBAL_STATISTICS
    # Reset statistics for this experiment
    GLOBAL_STATISTICS = []
    
    # Load all 20 repetitions
    repetitions_data = load_experiment_repetitions(experiment_folder)
    if repetitions_data is None:
        return False
    
    # Create output folder for plots
    plots_output_folder = os.path.join(experiment_folder, 'analysis_plots')
    if not os.path.exists(plots_output_folder):
        os.makedirs(plots_output_folder)
    
    try:
        # Generate all 6 plots
        plot_1_volatility_clustering(repetitions_data, plots_output_folder, experiment_name)
        plot_2_stylized_facts(repetitions_data, plots_output_folder, experiment_name)
        plot_3_treturns(repetitions_data, plots_output_folder, experiment_name)
        plot_4_price_series_with_ci(repetitions_data, plots_output_folder, experiment_name)
        plot_5_net_demand_autocorr(repetitions_data, plots_output_folder, experiment_name)
        plot_6_price_sign_autocorr(repetitions_data, plots_output_folder, experiment_name)
        
        # Generate and save tables for this experiment's statistics
        generate_and_save_stats_tables(plots_output_folder, experiment_name)
        
        return True
        
    except Exception as e:
        return False

def process_experiment_worker(args):
    """
    Worker function for parallel processing of a single experiment.
    
    Args:
        args (tuple): (experiments_root, exp_folder)
        
    Returns:
        tuple: (exp_name, success, processing_time)
    """
    experiments_root, exp_folder = args
    start_time = time.time()
    
    try:
        exp_path = os.path.join(experiments_root, exp_folder)
        success = process_single_experiment(exp_path, exp_folder)
        processing_time = time.time() - start_time
        return exp_folder, success, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        return exp_folder, False, processing_time

def process_all_experiments_parallel(max_workers=None):
    """
    Process all experiments in experiments_multi folder in parallel.
    
    Args:
        max_workers (int): Maximum number of parallel workers. If None, uses min(29, cpu_count)
    """
    experiments_root = 'experiments_multi'
    
    if not os.path.exists(experiments_root):
        return
    
    # Get all experiment folders
    experiment_folders = []
    for item in os.listdir(experiments_root):
        if item.startswith('exp_') and os.path.isdir(os.path.join(experiments_root, item)):
            experiment_folders.append(item)
    
    # Sort experiments numerically
    experiment_folders.sort(key=lambda x: int(x.split('_')[1]))
    
    # Determine number of workers
    cpu_count = multiprocessing.cpu_count()
    if max_workers is None:
        max_workers = min(29, cpu_count)
    else:
        max_workers = min(max_workers, cpu_count, len(experiment_folders))
    
    
    successful_experiments = 0
    failed_experiments = 0
    processing_times = []
    
    start_total_time = time.time()
    
    # Prepare arguments for worker processes
    worker_args = [(experiments_root, exp_folder) for exp_folder in experiment_folders]
    
    # Process experiments in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_experiment = {executor.submit(process_experiment_worker, args): args[1] 
                              for args in worker_args}
        
        # Process completed jobs as they finish
        for future in as_completed(future_to_experiment):
            exp_folder = future_to_experiment[future]
            
            try:
                exp_name, success, proc_time = future.result()
                processing_times.append(proc_time)
                
                if success:
                    successful_experiments += 1
                else:
                    failed_experiments += 1
                    
            except Exception as e:
                failed_experiments += 1
                
    total_time = time.time() - start_total_time
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    

def process_all_experiments():
    """
    Process all experiments - wrapper to choose between parallel and sequential processing.
    """
    return process_all_experiments_parallel(max_workers=29)

if __name__ == "__main__":
    process_all_experiments_parallel(max_workers=29)