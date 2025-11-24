import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import seaborn as sns
import warnings
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time
import random

warnings.filterwarnings('ignore')

# ==========================================================
# FUNCTIONS COPIED FROM trend_acceleration_plots.py FOR INDEPENDENCE
# ==========================================================

def identify_trends(prices):
    """Identify ascending and descending trends in price data."""
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
    """Enrich trend data with metrics like duration, returns, and velocity."""
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

# ==========================================================
# END OF COPIED FUNCTIONS
# ==========================================================

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

def load_experiment_repetitions(experiment_folder):
    """
    Load all repetitions for a single experiment.
    
    Args:
        experiment_folder (str): Path to experiment folder containing rep_X subfolders
        
    Returns:
        list: List of price series DataFrames, one per repetition
    """
    repetitions_data = []
    
    # Find repetition folders
    rep_folders = []
    for item in os.listdir(experiment_folder):
        if item.startswith('rep_') and os.path.isdir(os.path.join(experiment_folder, item)):
            rep_folders.append(item)
    
    # Sort repetition folders
    rep_folders.sort(key=lambda x: int(x.split('_')[1]))
    
    for rep_folder in rep_folders:
        csv_path = os.path.join(experiment_folder, rep_folder, 'price_series.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, index_col=0)
                # Exclude warm-up period
                if len(df) > 500:
                    df_filtered = df.iloc[500:]
                    repetitions_data.append(df_filtered['market_price'])
                else:
                    continue
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
    
    # Combine value and error into a single column for the CSV
    df_stats['value_with_error'] = df_stats.apply(
        lambda row: f"{row['value']:.4f} +/- {row['error']:.4f}" if pd.notna(row['error']) else f"{row['value']:.4f}",
        axis=1
    )
    
    # Save to CSV using the correct encoding
    df_stats.to_csv(csv_filename, index=False, encoding='utf-8-sig')

def collect_statistics(experiment_name, plot_number, subplot, data_name, data):
    """
    Collects statistics for a given dataset and adds to the global list.
    """
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
            'value': stat_value,
            'error': error_value
        })


def average_point_by_point(repetitions_data):
    """
    Average multiple repetitions point by point.
    
    Args:
        repetitions_data (list): List of price series
        
    Returns:
        tuple: (averaged_series, std_series) for confidence intervals
    """
    # Convert to numpy array for easier manipulation
    min_length = min(len(series) for series in repetitions_data)
    
    # Truncate all series to same length
    truncated_data = np.array([series[:min_length].values for series in repetitions_data])
    
    # Calculate point-wise mean and std
    mean_series = np.mean(truncated_data, axis=0)
    std_series = np.std(truncated_data, axis=0)
    
    return mean_series, std_series

def plot_1_logarithmic_returns(repetitions_data, output_folder, exp_name):
    """
    Plot 1: Volatility clustering (absolute returns) averaged point by point across 20 repetitions.
    """
    
    # Calculate absolute log returns for each repetition
    abs_log_returns_list = []
    for rep_data in repetitions_data:
        log_ret = calculate_log_returns(rep_data)
        abs_log_returns_list.append(np.abs(log_ret))
    
    # Average point by point
    min_length = min(len(lr) for lr in abs_log_returns_list)
    truncated_abs_returns = np.array([lr[:min_length] for lr in abs_log_returns_list])
    avg_abs_returns = np.mean(truncated_abs_returns, axis=0)
    
    # Collect statistics for Plot 1 (now volatility clustering)
    collect_statistics(exp_name, 1, 'N/A', 'Avg_Absolute_Returns', avg_abs_returns)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(avg_abs_returns, linewidth=1, alpha=0.8)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Absolute Logarithmic Returns')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'plot1_log_returns.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_2_stylized_facts(repetitions_data, output_folder, exp_name):
    """
    Plot 2: Stylized facts with 4 subplots matching analyze_stylized_facts style.
    """
    
    # Calculate log returns for each repetition
    log_returns_list = []
    abs_log_returns_list = []
    
    for rep_data in repetitions_data:
        log_ret = calculate_log_returns(rep_data)
        log_returns_list.append(log_ret)
        abs_log_returns_list.append(np.abs(log_ret))
    
    # Subplot 1: Returns distribution (pooling)
    pooled_returns = np.concatenate(log_returns_list)
    collect_statistics(exp_name, 2, '1_Returns_Distribution', 'Pooled_Log_Returns', pooled_returns)
    
    # Subplot 2: Logarithmic returns (average returns point by point)
    min_length_reg = min(len(lr) for lr in log_returns_list)
    truncated_returns = np.array([lr[:min_length_reg] for lr in log_returns_list])
    avg_log_returns = np.mean(truncated_returns, axis=0)
    collect_statistics(exp_name, 2, '2_Logarithmic_Returns', 'Averaged_Log_Returns', avg_log_returns)
    
    # Point-wise averaged absolute returns for ACF
    min_length = min(len(lr) for lr in abs_log_returns_list)
    truncated_abs_returns = np.array([lr[:min_length] for lr in abs_log_returns_list])
    avg_abs_returns = np.mean(truncated_abs_returns, axis=0)
    
    # Subplot 3: Returns Autocorrelation
    collect_statistics(exp_name, 2, '3_Returns_ACF', 'Point_Averaged_Returns', avg_log_returns)
    
    # Subplot 4: Absolute Returns Autocorrelation
    collect_statistics(exp_name, 2, '4_Abs_Returns_ACF', 'Point_Averaged_Abs_Returns', avg_abs_returns)
    
    # Create subplots with same style as analyze_stylized_facts
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Use seaborn color palette
    plot_color = sns.color_palette()[0]
    
    # --- Cambios para el Plot 2, Subplot 1 ---
    # 1. Returns Distribution (ahora con puntos y escala log, sin ajuste normal)
    # Calculamos el histograma sin normalización (frecuencias absolutas)
    counts, bin_edges = np.histogram(pooled_returns, bins=100, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Graficamos los puntos
    axs[0, 0].scatter(bin_centers, counts, color=plot_color, label='Empirical Distribution')
    
    # Escalamos el eje Y a log
    axs[0, 0].set_yscale('log')
    
    # El ajuste normal ha sido removido como se solicitó
    # ----------------------------------------
    
    # Calculate statistics with errors for the statistics box
    pooled_stats = calculate_statistics_with_errors(pooled_returns)
    stats_text = (f'Mean: {pooled_stats["mean"]:.4f} ± {pooled_stats["mean_error"]:.4f}\n'
                  f'Std Dev: {pooled_stats["std"]:.4f}\n'
                  f'Skewness: {pooled_stats["skewness"]:.4f} ± {pooled_stats["skewness_error"]:.4f}\n'
                  f'Kurtosis: {pooled_stats["kurtosis"]:.4f} ± {pooled_stats["kurtosis_error"]:.4f}')
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8)
    axs[0, 0].text(0.05, 0.95, stats_text, transform=axs[0, 0].transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    
    # --- Cambio de títulos a letras (a), (b), (c), (d) ---
    axs[0, 0].set_title('(a) Returns Distribution')
    axs[0, 0].set_xlabel('Logarithmic Returns')
    axs[0, 0].set_ylabel('Frequency (log)')
    axs[0, 0].legend()
    
    # 2. Logarithmic Returns (mostrar una repetición representativa aleatoria)
    random_index = random.randint(0, len(log_returns_list) - 1)
    representative_returns = log_returns_list[random_index]
    axs[0, 1].plot(range(len(representative_returns)), representative_returns, color=plot_color, alpha=0.7)
    axs[0, 1].set_title(f'(b) Logarithmic Returns (Rep. {random_index + 1})')
    axs[0, 1].set_xlabel('Time Steps')
    axs[0, 1].set_ylabel('Logarithmic Returns')
    
    # 3. Returns Autocorrelation (CORRECTO: promedia las ACFs, no las series)
    acf_list = []
    for log_ret in log_returns_list:
        acf_values = acf(log_ret, nlags=40, fft=True)
        acf_list.append(acf_values)
    avg_acf_returns = np.mean(acf_list, axis=0)  # Promedio de las 20 ACFs
    
    # Plotear manualmente para mantener el estilo visual
    lags = np.arange(len(avg_acf_returns))
    axs[1, 0].plot(lags, avg_acf_returns, 'o-', markersize=5)
    axs[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[1, 0].axhline(y=1.96/np.sqrt(len(log_returns_list[0])), color='r', linestyle='--', alpha=0.5)
    axs[1, 0].axhline(y=-1.96/np.sqrt(len(log_returns_list[0])), color='r', linestyle='--', alpha=0.5)
    axs[1, 0].set_title('(c) Returns Autocorrelation')
    axs[1, 0].set_xlabel('Lag')
    axs[1, 0].set_ylabel('Autocorrelation')
    
    # 4. Absolute Returns Autocorrelation (CORRECTO: promedia las ACFs, no las series)
    acf_abs_list = []
    for abs_log_ret in abs_log_returns_list:
        acf_abs_values = acf(abs_log_ret, nlags=120, fft=True)
        acf_abs_list.append(acf_abs_values)
    avg_acf_abs_returns = np.mean(acf_abs_list, axis=0)  # Promedio de las 20 ACFs
    
    # Plotear manualmente para mantener el estilo visual
    lags_abs = np.arange(len(avg_acf_abs_returns))
    axs[1, 1].plot(lags_abs, avg_acf_abs_returns, 'o-', markersize=5)
    axs[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axs[1, 1].axhline(y=1.96/np.sqrt(len(abs_log_returns_list[0])), color='r', linestyle='--', alpha=0.5)
    axs[1, 1].axhline(y=-1.96/np.sqrt(len(abs_log_returns_list[0])), color='r', linestyle='--', alpha=0.5)
    axs[1, 1].set_title('(d) Absolute Returns Autocorrelation')
    axs[1, 1].set_xlabel('Lag')
    axs[1, 1].set_ylabel('Autocorrelation')
    # ----------------------------------------------------
    
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
        prices = rep_data.values
        
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
    
    # --- Cambios para el Plot 3, Subplot 1 ---
    # 1. T-returns Distribution (ahora con puntos y escala log, sin ajuste normal)
    # Calculamos la densidad del histograma manualmente
    counts, bin_edges = np.histogram(all_treturns, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Graficamos los puntos
    axs[0].scatter(bin_centers, counts, color=plot_color, label='Empirical Distribution')
    
    # Escalamos el eje Y a log
    axs[0].set_yscale('log')
    
    # El ajuste normal ha sido removido como se solicitó
    # ----------------------------------------
    
    # Calculate statistics with errors for the statistics box
    treturns_stats = calculate_statistics_with_errors(all_treturns)
    stats_text = (f'Mean: {treturns_stats["mean"]:.4f} ± {treturns_stats["mean_error"]:.4f}\n'
                  f'Std Dev: {treturns_stats["std"]:.4f}\n'
                  f'Skewness: {treturns_stats["skewness"]:.4f} ± {treturns_stats["skewness_error"]:.4f}\n'
                  f'Kurtosis: {treturns_stats["kurtosis"]:.4f} ± {treturns_stats["kurtosis_error"]:.4f}')
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8)
    axs[0].text(0.05, 0.95, stats_text, transform=axs[0].transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
    
    # --- Cambios de títulos a letras (a) y (b) ---
    axs[0].set_title('(a) TReturns Distribution')
    axs[0].set_xlabel('T-returns')
    axs[0].set_ylabel('Density (log)')
    axs[0].legend()
    
    # 2. ACF of T-returns (using statsmodels plot_acf without confidence intervals)
    plot_acf(all_treturns, lags=400, ax=axs[1], title='(b) TReturns Autocorrelation', alpha=None)
    axs[1].set_xlabel('Lag')
    axs[1].set_ylabel('Autocorrelation')
    # ----------------------------------------------------
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_folder, 'plot3_treturns.png'), dpi=300, bbox_inches='tight')
    plt.close()

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
        # Generate all 4 plots
        plot_1_logarithmic_returns(repetitions_data, plots_output_folder, experiment_name)
        plot_2_stylized_facts(repetitions_data, plots_output_folder, experiment_name)
        plot_3_treturns(repetitions_data, plots_output_folder, experiment_name)
        plot_4_price_series_with_ci(repetitions_data, plots_output_folder, experiment_name)
        
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

def process_all_experiments_parallel(max_workers=None, experiments_root='experiments_multi'):
    """
    Process all experiments in experiments_multi folder in parallel.
    
    Args:
        max_workers (int): Maximum number of parallel workers. If None, uses min(29, cpu_count)
        experiments_root (str): Root folder containing experiments (can be a single experiment folder)
    """
    if not os.path.exists(experiments_root):
        return
    
    # Check if this is a single experiment folder (has rep_ subfolders)
    has_reps = any(item.startswith('rep_') for item in os.listdir(experiments_root) if os.path.isdir(os.path.join(experiments_root, item)))
    
    if has_reps:
        # This is a single experiment folder, process it directly
        exp_name = os.path.basename(experiments_root)
        success = process_single_experiment(experiments_root, exp_name)
        return
    
    # Otherwise, look for exp_ subfolders (legacy behavior)
    experiment_folders = []
    for item in os.listdir(experiments_root):
        if item.startswith('exp_') and os.path.isdir(os.path.join(experiments_root, item)):
            experiment_folders.append(item)
    
    if not experiment_folders:
        return
    
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

def process_all_experiments(experiments_root='experiments_multi'):
    """
    Process all experiments - wrapper to choose between parallel and sequential processing.
    
    Args:
        experiments_root (str): Root folder containing experiments
    """
    return process_all_experiments_parallel(max_workers=29, experiments_root=experiments_root)

if __name__ == "__main__":
    import sys
    
    # Check if custom folder was provided as command line argument
    if len(sys.argv) > 1:
        experiments_root = sys.argv[1]
    else:
        experiments_root = 'experiments_multi'
    
    
    process_all_experiments(experiments_root=experiments_root)
