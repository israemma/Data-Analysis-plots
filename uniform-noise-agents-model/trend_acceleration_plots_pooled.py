import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import pandas as pd
import pickle
import warnings
from scipy.stats import geom, lognorm, burr12, genextreme
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

warnings.filterwarnings('ignore')

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
    'lines.markersize': 5,
})

ASCENDING_COLOR = '#2ca02c'  # Green
DESCENDING_COLOR = '#d62728' # Red

# Global lists to collect metrics
ALL_FIT_METRICS = []
ALL_STATISTICAL_METRICS = []
GLOBAL_COMBINED_METRICS = []

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

def _create_trend_info(trends_in_sequence):
    """Create trend sequence information from a list of trends."""
    start_trend = trends_in_sequence[0]
    end_trend = trends_in_sequence[-1]
    amplification = 0.0
    if start_trend['abs_velocity'] > 1e-9:
        amplification = end_trend['abs_velocity'] / start_trend['abs_velocity']
    return {
        'trends': trends_in_sequence,
        'length': len(trends_in_sequence),
        'total_duration': sum(t['duration'] for t in trends_in_sequence),
        'velocity_amplification': amplification,
        'max_velocity': max(t['abs_velocity'] for t in trends_in_sequence),
        'min_velocity': min(t['abs_velocity'] for t in trends_in_sequence),
        'total_return': sum(t['abs_return'] for t in trends_in_sequence)
    }

def analyze_trend_sequences(enriched_trends_df, min_sequence_length=2):
    """Analyze sequences of trends (acceleration/deceleration patterns)."""
    df = enriched_trends_df.sort_values('start').reset_index(drop=True)
    ascending_sequences, descending_sequences = [], []
    current_ascending, current_descending = [], []
    for i in range(len(df) - 1):
        current_trend = df.iloc[i].to_dict()
        next_trend = df.iloc[i+1].to_dict()
        if np.isnan(current_trend['abs_velocity']) or np.isnan(next_trend['abs_velocity']):
            if len(current_ascending) >= min_sequence_length:
                ascending_sequences.append(_create_trend_info(current_ascending))
            if len(current_descending) >= min_sequence_length:
                descending_sequences.append(_create_trend_info(current_descending))
            current_ascending = []
            current_descending = []
            continue
        if next_trend['abs_velocity'] > current_trend['abs_velocity']:
            if not current_ascending:
                current_ascending.append(current_trend)
            current_ascending.append(next_trend)
            if len(current_descending) >= min_sequence_length:
                descending_sequences.append(_create_trend_info(current_descending))
            current_descending = []
        elif next_trend['abs_velocity'] < current_trend['abs_velocity']:
            if not current_descending:
                current_descending.append(current_trend)
            current_descending.append(next_trend)
            if len(current_ascending) >= min_sequence_length:
                ascending_sequences.append(_create_trend_info(current_ascending))
            current_ascending = []
        else:
            if len(current_ascending) >= min_sequence_length:
                ascending_sequences.append(_create_trend_info(current_ascending))
            if len(current_descending) >= min_sequence_length:
                descending_sequences.append(_create_trend_info(current_descending))
            current_ascending, current_descending = [], []
    if len(current_ascending) >= min_sequence_length:
        ascending_sequences.append(_create_trend_info(current_ascending))
    if len(current_descending) >= min_sequence_length:
        descending_sequences.append(_create_trend_info(current_descending))
    return {'ascending': ascending_sequences, 'descending': descending_sequences}

# ===== STATISTICAL AND FITTING FUNCTIONS =====

def calculate_rmse(observed_freq, expected_freq):
    """Calculate RMSE between observed and expected frequencies."""
    if len(observed_freq) == 0 or len(expected_freq) == 0:
        return np.nan
    valid_indices = ~(np.isnan(observed_freq) | np.isinf(observed_freq) | np.isnan(expected_freq) | np.isinf(expected_freq))
    observed_freq = observed_freq[valid_indices]
    expected_freq = expected_freq[valid_indices]
    if len(observed_freq) == 0:
        return np.nan
    return np.sqrt(np.mean((observed_freq - expected_freq)**2))

def calculate_statistics_with_errors(data):
    """Calculate comprehensive statistics with bootstrap errors."""
    if len(data) == 0:
        return {
            'mean': np.nan, 'mean_error': np.nan,
            'median': np.nan, 'median_error': np.nan,
            'std': np.nan, 'std_error': np.nan,
            'count': 0, 'count_error': 0,
            'min': np.nan, 'min_error': np.nan,
            'max': np.nan, 'max_error': np.nan
        }
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1) if n > 1 else 0
    median = np.median(data)
    mean_error = std / np.sqrt(n) if n > 0 else np.nan
    median_error = 1.253 * std / np.sqrt(n) if n > 0 else np.nan
    std_error = std / np.sqrt(2 * (n - 1)) if n > 1 else np.nan
    return {
        'mean': mean, 'mean_error': mean_error,
        'median': median, 'median_error': median_error,
        'std': std, 'std_error': std_error,
        'count': n, 'count_error': 0,
        'min': np.min(data), 'min_error': np.nan,
        'max': np.max(data), 'max_error': np.nan
    }

def fit_geometric_to_duration(data):
    """Fit geometric distribution to discrete duration data."""
    if len(data) < 2 or not np.all(data > 0):
        return None, None, None, None, None, None
    try:
        data = np.round(data).astype(int)
        sample_mean = np.mean(data)
        if sample_mean <= 0:
            return None, None, None, None, None, None
        
        p_est = 1.0 / sample_mean
        p_est = min(max(p_est, 1e-10), 1.0)
        params = (0, p_est)
        
        unique_values = np.unique(data)
        max_val = min(unique_values.max(), int(sample_mean + 3 * np.sqrt(sample_mean)))
        
        observed_freq = []
        expected_freq = []
        values_range = range(1, max_val + 1)
        
        for val in values_range:
            obs_count = np.sum(data == val)
            exp_prob = geom.pmf(val, p_est)
            exp_count = exp_prob * len(data)
            observed_freq.append(obs_count)
            expected_freq.append(exp_count)
        
        observed_freq = np.array(observed_freq)
        expected_freq = np.array(expected_freq)
        
        mask = expected_freq >= 1
        if np.sum(mask) >= 2:
            observed_freq_filtered = observed_freq[mask]
            expected_freq_filtered = expected_freq[mask]
            
            observed_sum = np.sum(observed_freq_filtered)
            expected_sum = np.sum(expected_freq_filtered)
            if expected_sum > 0:
                expected_freq_filtered = expected_freq_filtered * (observed_sum / expected_sum)
            
            from scipy.stats import chisquare
            chi2_stat, chi2_p_value = chisquare(observed_freq_filtered, expected_freq_filtered)
        else:
            chi2_stat, chi2_p_value = np.nan, np.nan
        
        rmse_val = calculate_rmse(observed_freq, expected_freq)
        
        n_bootstrap = 100
        bootstrap_params = []
        for _ in range(n_bootstrap):
            try:
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_sample = np.round(bootstrap_sample).astype(int)
                bootstrap_mean = np.mean(bootstrap_sample)
                if bootstrap_mean > 0:
                    bootstrap_p = 1.0 / bootstrap_mean
                    bootstrap_p = min(max(bootstrap_p, 1e-10), 1.0)
                    bootstrap_params.append((0, bootstrap_p))
            except:
                continue
        
        param_errors = np.std(bootstrap_params, axis=0) if bootstrap_params else [np.nan, np.nan]
        
        return params, param_errors, chi2_stat, chi2_p_value, rmse_val, (observed_freq, expected_freq, values_range)
        
    except Exception as e:
        return None, None, None, None, None, None

def fit_lognormal_to_data(data):
    """Fit lognormal distribution to data."""
    if len(data) < 2 or not np.all(data > 0):
        return None, None, None, None, None
    try:
        params = lognorm.fit(data, floc=0)
        s, loc, scale = params
        ks_D, ks_p_value = stats.kstest(data, 'lognorm', args=params)
        n_bootstrap = 100
        bootstrap_params = []
        for _ in range(n_bootstrap):
            try:
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_fit = lognorm.fit(bootstrap_sample, floc=0)
                bootstrap_params.append(bootstrap_fit)
            except:
                continue
        param_errors = np.std(bootstrap_params, axis=0) if bootstrap_params else [np.nan, np.nan, np.nan]
        num_bins = max(10, int(np.sqrt(len(data))))
        observed_counts, bin_edges = np.histogram(data, bins=num_bins, density=False)
        expected_counts = [(lognorm.cdf(bin_edges[i + 1], *params) - lognorm.cdf(bin_edges[i], *params)) * len(data) for i in range(len(bin_edges) - 1)]
        rmse_val = calculate_rmse(observed_counts, np.array(expected_counts))
        return params, param_errors, ks_D, ks_p_value, rmse_val
    except Exception:
        return None, None, None, None, None

def fit_burr12_to_data(data):
    """Fit Burr Type XII distribution to data."""
    if len(data) < 2 or not np.all(data > 0):
        return None, None, None, None, None
    try:
        params = burr12.fit(data, floc=0)
        c, d, loc, scale = params
        ks_D, ks_p_value = stats.kstest(data, 'burr12', args=params)
        n_bootstrap = 100
        bootstrap_params = []
        for _ in range(n_bootstrap):
            try:
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_fit = burr12.fit(bootstrap_sample, floc=0)
                bootstrap_params.append(bootstrap_fit)
            except:
                continue
        param_errors = np.std(bootstrap_params, axis=0) if bootstrap_params else [np.nan, np.nan, np.nan, np.nan]
        num_bins = max(10, int(np.sqrt(len(data))))
        observed_counts, bin_edges = np.histogram(data, bins=num_bins, density=False)
        expected_counts = [(burr12.cdf(bin_edges[i + 1], *params) - burr12.cdf(bin_edges[i], *params)) * len(data) for i in range(len(bin_edges) - 1)]
        rmse_val = calculate_rmse(observed_counts, np.array(expected_counts))
        return params, param_errors, ks_D, ks_p_value, rmse_val
    except Exception:
        return None, None, None, None, None

def fit_genextreme_to_data(data):
    """Fit Generalized Extreme Value distribution to data."""
    if len(data) < 2 or not np.all(data > 0):
        return None, None, None, None, None
    try:
        params = genextreme.fit(data)
        c, loc, scale = params
        ks_D, ks_p_value = stats.kstest(data, 'genextreme', args=params)
        n_bootstrap = 100
        bootstrap_params = []
        for _ in range(n_bootstrap):
            try:
                bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_fit = genextreme.fit(bootstrap_sample)
                bootstrap_params.append(bootstrap_fit)
            except:
                continue
        param_errors = np.std(bootstrap_params, axis=0) if bootstrap_params else [np.nan, np.nan, np.nan]
        num_bins = max(10, int(np.sqrt(len(data))))
        observed_counts, bin_edges = np.histogram(data, bins=num_bins, density=False)
        expected_counts = [(genextreme.cdf(bin_edges[i + 1], *params) - genextreme.cdf(bin_edges[i], *params)) * len(data) for i in range(len(bin_edges) - 1)]
        rmse_val = calculate_rmse(observed_counts, np.array(expected_counts))
        return params, param_errors, ks_D, ks_p_value, rmse_val
    except Exception:
        return None, None, None, None, None

# ===== PLOTTING HELPER FUNCTIONS =====

def _add_statistical_summary_to_plot(ax, data, plot_type, x_label_for_mean_median, plot_title_category, subplot_category, direction_label, index_name, plot_number, metric_abbr, geom_fit_params=None, geom_fit_rmse=None, geom_fit_errors=None, geom_chi2=None, geom_chi2_p=None):
    """Add statistical summary to plot."""
    if data is None or len(data) < 2:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=True, loc='best', fontsize=8)
        return
    stats_dict = calculate_statistics_with_errors(data)
    mean_val = stats_dict['mean']
    median_val = stats_dict['median']
    count_val = stats_dict['count']
    std_val = stats_dict['std']
    min_val = stats_dict['min']
    max_val = stats_dict['max']
    mean_error = stats_dict['mean_error']
    median_error = stats_dict['median_error']
    std_error = stats_dict['std_error']

    ax.axvline(mean_val, color='green', linestyle='--', linewidth=1.5, label=f"Mean: {mean_val:.2f}±{mean_error:.2f}")
    ax.axvline(median_val, color='red', linestyle=':', linewidth=1.5, label=f"Median: {median_val:.2f}")
    
    if geom_fit_params is not None and geom_chi2 is not None and geom_chi2_p is not None:
        loc, p = geom_fit_params
        goodness_text = f"Geometric Fit:\nχ² = {geom_chi2:.3f}\np-value = {geom_chi2_p:.3f}\nRMSE = {geom_fit_rmse:.3f}"
        ax.text(0.98, 0.70, goodness_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.2, edgecolor='black', linewidth=1.5))
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, loc='best', fontsize=8)

def _add_statistical_summary_to_plot_continuous(ax, data, plot_type, x_label_for_mean_median, plot_title_category, subplot_category, direction_label, index_name, plot_number, metric_abbr, fit_distribution=None, fit_params=None, fit_rmse=None, fit_errors=None, ks_D=None, ks_p=None):
    """Add statistical summary for continuous distributions with KS test metrics."""
    if data is None or len(data) < 2:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(frameon=True, loc='best', fontsize=8)
        return
    
    stats_dict = calculate_statistics_with_errors(data)
    mean_val = stats_dict['mean']
    median_val = stats_dict['median']
    count_val = stats_dict['count']
    std_val = stats_dict['std']
    min_val = stats_dict['min']
    max_val = stats_dict['max']
    mean_error = stats_dict['mean_error']
    median_error = stats_dict['median_error']
    std_error = stats_dict['std_error']

    ax.axvline(mean_val, color='green', linestyle='--', linewidth=1.5, label=f"Mean: {mean_val:.2f}±{mean_error:.2f}")
    ax.axvline(median_val, color='red', linestyle=':', linewidth=1.5, label=f"Median: {median_val:.2f}")
    
    if fit_distribution is not None and ks_D is not None and ks_p is not None:
        if fit_rmse is not None and not np.isnan(fit_rmse):
            goodness_text = f"{fit_distribution} Fit:\nKS-D = {ks_D:.3f}\nKS p-value = {ks_p:.3f}"
        else:
            goodness_text = f"{fit_distribution} Fit:\nKS-D = {ks_D:.3f}\nKS p-value = {ks_p:.3f}"
        ax.text(0.02, 0.70, goodness_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.2, edgecolor='black', linewidth=1.5))
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, loc='best', fontsize=8)

def load_repetitions_for_experiment(experiment_folder):
    """
    Load all repetitions for a single experiment.
    
    Args:
        experiment_folder (str): Path to experiment folder containing rep_X subfolders
        
    Returns:
        list: List of price series DataFrames, one per repetition
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
                # Skip first 500 steps (warm-up period)
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

def analyze_trends_from_repetitions(repetitions_data):
    """
    Analyze trends from all repetitions and return both pooled data and per-repetition data.
    
    Args:
        repetitions_data (list): List of price series from 20 repetitions
        
    Returns:
        tuple: (pooled_trends_df, pooled_sequences_dict, sequences_by_repetition)
    """
    all_trends = []
    all_sequences = {'ascending': [], 'descending': []}
    sequences_by_repetition = {'ascending': [], 'descending': []}
    
    for i, rep_data in enumerate(repetitions_data):
        try:
            prices = rep_data.values
            
            # Remove any NaN values
            prices_clean = prices[~np.isnan(prices)]
            if len(prices_clean) < 2:
                continue
                
            # Identify trends
            trends = identify_trends(prices_clean)
            if not trends:
                continue
                
            # Enrich trend data
            enriched_trends = enrich_trend_data(prices_clean, trends)
            if not enriched_trends:
                continue
                
            # Convert to DataFrame and add repetition info
            df_trends = pd.DataFrame(enriched_trends)
            df_trends['repetition'] = i
            all_trends.append(df_trends)
            
            # Analyze trend sequences
            sequences = analyze_trend_sequences(df_trends)
            
            # Store sequences by repetition (for scientific CDF averaging)
            sequences_by_repetition['ascending'].append(sequences.get('ascending', []))
            sequences_by_repetition['descending'].append(sequences.get('descending', []))
            
            # Also keep pooled sequences for other plots
            all_sequences['ascending'].extend(sequences.get('ascending', []))
            all_sequences['descending'].extend(sequences.get('descending', []))
            
        except Exception as e:
            continue
    
    # Combine all trends
    if all_trends:
        pooled_trends_df = pd.concat(all_trends, ignore_index=True)
    else:
        pooled_trends_df = pd.DataFrame()
    
    return pooled_trends_df, all_sequences, sequences_by_repetition

def calculate_empirical_cdf_for_repetitions(sequences_by_repetition, sequence_type, metric, x_grid):
    """
    Calculate empirical CDF for each repetition and return averaged CDF with confidence bands.
    
    Args:
        sequences_by_repetition (dict): Sequences organized by repetition
        sequence_type (str): 'ascending' or 'descending'
        metric (str): metric name
        x_grid (array): x values to evaluate CDF at
        
    Returns:
        tuple: (mean_cdf, ci_lower, ci_upper, pooled_data)
    """
    cdfs_per_rep = []
    all_data_per_rep = []
    
    for rep_sequences in sequences_by_repetition[sequence_type]:
        # Extract metric data for this repetition
        rep_data = np.array([seq[metric] for seq in rep_sequences if seq.get(metric, 0) > 1e-9])
        all_data_per_rep.append(rep_data)
        
        if len(rep_data) == 0:
            # If no data for this repetition, use zeros
            rep_cdf = np.zeros_like(x_grid)
        else:
            # Calculate empirical CDF
            rep_cdf = []
            for x_val in x_grid:
                cdf_val = np.sum(rep_data <= x_val) / len(rep_data)
                rep_cdf.append(cdf_val)
            rep_cdf = np.array(rep_cdf)
        
        cdfs_per_rep.append(rep_cdf)
    
    # Convert to numpy array for easier manipulation
    cdfs_per_rep = np.array(cdfs_per_rep)
    
    # Calculate mean CDF and confidence bands across repetitions
    mean_cdf = np.mean(cdfs_per_rep, axis=0)
    ci_lower = np.percentile(cdfs_per_rep, 2.5, axis=0)
    ci_upper = np.percentile(cdfs_per_rep, 97.5, axis=0)
    
    # Pool all data for distribution fitting
    pooled_data = np.concatenate([rep_data for rep_data in all_data_per_rep if len(rep_data) > 0])
    
    return mean_cdf, ci_lower, ci_upper, pooled_data

def fit_distribution_to_averaged_cdf(x_grid, mean_cdf, distribution_name):
    """
    Fit distribution parameters to averaged CDF using optimization.
    
    Args:
        x_grid (array): x values where CDF is evaluated
        mean_cdf (array): averaged CDF values
        distribution_name (str): name of distribution ('genextreme', 'lognorm', 'burr12')
        
    Returns:
        tuple: (fit_params, fit_errors, ks_D, ks_p, rmse_val)
    """
    from scipy import optimize
    from scipy.stats import ks_2samp
    
    def objective_function(params, x_grid, empirical_cdf, dist_func):
        """Objective function to minimize - RMSE between theoretical and empirical CDF"""
        try:
            theoretical_cdf = dist_func.cdf(x_grid, *params)
            # Remove any nan or inf values
            valid_mask = np.isfinite(theoretical_cdf) & np.isfinite(empirical_cdf)
            if not np.any(valid_mask):
                return np.inf
            return np.sqrt(np.mean((theoretical_cdf[valid_mask] - empirical_cdf[valid_mask]) ** 2))
        except:
            return np.inf
    
    try:
        if distribution_name == 'genextreme':
            from scipy.stats import genextreme
            # Initial guess for GenExtreme parameters (c, loc, scale)
            initial_guess = [0.1, np.percentile(x_grid, 50), np.percentile(x_grid, 75) - np.percentile(x_grid, 25)]
            bounds = [(-2, 2), (x_grid.min(), x_grid.max()), (0.001, x_grid.max())]
            
            result = optimize.minimize(objective_function, initial_guess, 
                                     args=(x_grid, mean_cdf, genextreme),
                                     bounds=bounds, method='L-BFGS-B')
            fit_params = result.x if result.success else None
            dist_func = genextreme
            
        elif distribution_name == 'lognorm':
            from scipy.stats import lognorm
            # Initial guess for LogNormal parameters (s, loc, scale)
            log_data = np.log(x_grid[x_grid > 0])
            initial_guess = [np.std(log_data), 0, np.exp(np.mean(log_data))]
            bounds = [(0.01, 10), (0, x_grid.min()), (0.001, x_grid.max())]
            
            result = optimize.minimize(objective_function, initial_guess,
                                     args=(x_grid, mean_cdf, lognorm),
                                     bounds=bounds, method='L-BFGS-B')
            fit_params = result.x if result.success else None
            dist_func = lognorm
            
        elif distribution_name == 'burr12':
            from scipy.stats import burr12
            # Initial guess for Burr12 parameters (c, d, loc, scale)
            initial_guess = [2.0, 1.0, 0, np.median(x_grid)]
            bounds = [(0.1, 10), (0.1, 10), (0, x_grid.min()), (0.001, x_grid.max())]
            
            result = optimize.minimize(objective_function, initial_guess,
                                     args=(x_grid, mean_cdf, burr12),
                                     bounds=bounds, method='L-BFGS-B')
            fit_params = result.x if result.success else None
            dist_func = burr12
        else:
            return None, None, None, None, None
            
        if fit_params is None:
            return None, None, None, None, None
            
        # Calculate goodness of fit metrics
        theoretical_cdf = dist_func.cdf(x_grid, *fit_params)
        
        # RMSE between theoretical and empirical CDF
        rmse_val = np.sqrt(np.mean((theoretical_cdf - mean_cdf) ** 2))
        
        # KS test approximation (using max difference)
        ks_D = np.max(np.abs(theoretical_cdf - mean_cdf))
        
        # Approximate p-value (this is simplified)
        n_eff = len(x_grid)  # Effective sample size
        ks_p = 2 * np.exp(-2 * n_eff * ks_D**2) if ks_D > 0 else 1.0
        ks_p = min(ks_p, 1.0)
        
        # Calculate parameter errors using bootstrap method
        def bootstrap_fit_errors(x_grid, mean_cdf, dist_func, fit_params, n_bootstrap=100):
            """Calculate parameter errors using bootstrap resampling of the CDF"""
            bootstrap_params = []
            
            for _ in range(n_bootstrap):
                try:
                    # Add small random noise to the empirical CDF for bootstrap
                    noise = np.random.normal(0, 0.01, size=len(mean_cdf))
                    noisy_cdf = np.clip(mean_cdf + noise, 0, 1)
                    
                    # Fit to the noisy CDF
                    bootstrap_result = optimize.minimize(objective_function, fit_params,
                                                         args=(x_grid, noisy_cdf, dist_func),
                                                         bounds=bounds, method='L-BFGS-B')
                    
                    if bootstrap_result.success:
                        bootstrap_params.append(bootstrap_result.x)
                except:
                    continue
            
            if len(bootstrap_params) > 10:  # Need sufficient successful bootstraps
                bootstrap_params = np.array(bootstrap_params)
                param_errors = np.std(bootstrap_params, axis=0)
                return param_errors
            else:
                return np.full_like(fit_params, np.nan)
        
        # Calculate errors using bootstrap method
        fit_errors = bootstrap_fit_errors(x_grid, mean_cdf, dist_func, fit_params)
        
        return fit_params, fit_errors, ks_D, ks_p, rmse_val
        
    except Exception as e:
        return None, None, None, None, None

def calculate_rmse_error_bootstrap(x_grid, mean_cdf, distribution_name, fit_params, n_bootstrap=50):
    """
    Calculate RMSE error using bootstrap method.
    
    Args:
        x_grid (array): x values where CDF is evaluated
        mean_cdf (array): averaged CDF values
        distribution_name (str): name of distribution
        fit_params (array): fitted parameters
        n_bootstrap (int): number of bootstrap samples
        
    Returns:
        float: bootstrap estimate of RMSE error
    """
    from scipy import optimize
    
    def objective_function(params, x_grid, empirical_cdf, dist_func):
        """Objective function to minimize - RMSE between theoretical and empirical CDF"""
        try:
            theoretical_cdf = dist_func.cdf(x_grid, *params)
            valid_mask = np.isfinite(theoretical_cdf) & np.isfinite(empirical_cdf)
            if not np.any(valid_mask):
                return np.inf
            return np.sqrt(np.mean((theoretical_cdf[valid_mask] - empirical_cdf[valid_mask]) ** 2))
        except:
            return np.inf
    
    try:
        if distribution_name == 'genextreme':
            from scipy.stats import genextreme
            dist_func = genextreme
            bounds = [(-2, 2), (x_grid.min(), x_grid.max()), (0.001, x_grid.max())]
        elif distribution_name == 'lognorm':
            from scipy.stats import lognorm
            dist_func = lognorm
            log_data = np.log(x_grid[x_grid > 0])
            bounds = [(0.01, 10), (0, x_grid.min()), (0.001, x_grid.max())]
        elif distribution_name == 'burr12':
            from scipy.stats import burr12
            dist_func = burr12
            bounds = [(0.1, 10), (0.1, 10), (0, x_grid.min()), (0.001, x_grid.max())]
        else:
            return np.nan
        
        bootstrap_rmse = []
        
        for _ in range(n_bootstrap):
            try:
                # Add small random noise to the empirical CDF for bootstrap
                noise = np.random.normal(0, 0.01, size=len(mean_cdf))
                noisy_cdf = np.clip(mean_cdf + noise, 0, 1)
                
                # Fit to the noisy CDF
                bootstrap_result = optimize.minimize(objective_function, fit_params,
                                                     args=(x_grid, noisy_cdf, dist_func),
                                                     bounds=bounds, method='L-BFGS-B')
                
                if bootstrap_result.success:
                    # Calculate RMSE for this bootstrap sample
                    theoretical_cdf = dist_func.cdf(x_grid, *bootstrap_result.x)
                    rmse_boot = np.sqrt(np.mean((theoretical_cdf - noisy_cdf) ** 2))
                    bootstrap_rmse.append(rmse_boot)
            except:
                continue
        
        if len(bootstrap_rmse) > 10:  # Need sufficient successful bootstraps
            return np.std(bootstrap_rmse)
        else:
            return np.nan
            
    except Exception as e:
        return np.nan

def calculate_geometric_rmse_error_bootstrap(data, fit_params, n_bootstrap=50):
    """
    Calculate RMSE error for geometric distribution using bootstrap method.
    
    Args:
        data (array): original discrete data
        fit_params (tuple): fitted parameters (loc, p)
        n_bootstrap (int): number of bootstrap samples
        
    Returns:
        float: bootstrap estimate of RMSE error
    """
    from scipy.stats import geom
    
    try:
        loc, p = fit_params
        bootstrap_rmse = []
        
        for _ in range(n_bootstrap):
            try:
                # Bootstrap resample the data
                bootstrap_data = np.random.choice(data, size=len(data), replace=True)
                
                # Fit geometric to bootstrap sample
                boot_params, _, _, _, boot_rmse, _ = fit_geometric_to_duration(bootstrap_data)
                
                if boot_params is not None and boot_rmse is not None:
                    bootstrap_rmse.append(boot_rmse)
            except:
                continue
        
        if len(bootstrap_rmse) > 10:  # Need sufficient successful bootstraps
            return np.std(bootstrap_rmse)
        else:
            return np.nan
            
    except Exception as e:
        return np.nan

def fit_exponential_to_data(x_data, y_data):
    """
    Fit exponential model y = a * e^(-b * x) to data points.
    
    Args:
        x_data (array): x values (trend durations)
        y_data (array): y values (frequencies)
        
    Returns:
        tuple: (fit_params, fit_errors, r_squared, p_value, rmse, rmse_error)
    """
    from scipy.optimize import curve_fit
    from scipy import stats
    
    def exponential_func(x, a, b):
        """Exponential function y = a * e^(-b * x)"""
        return a * np.exp(-b * x)
    
    try:
        # Remove any zero or negative y values for fitting
        valid_mask = (y_data > 0) & np.isfinite(x_data) & np.isfinite(y_data)
        if np.sum(valid_mask) < 3:  # Need at least 3 points
            return None, None, None, None, None, None
            
        x_fit = x_data[valid_mask]
        y_fit = y_data[valid_mask]
        
        # Initial parameter guess
        a_init = np.max(y_fit)
        b_init = 0.5
        
        # Fit the exponential model
        popt, pcov = curve_fit(exponential_func, x_fit, y_fit, 
                             p0=[a_init, b_init], 
                             bounds=([0, 0], [np.inf, np.inf]),
                             maxfev=5000)
        
        a_fit, b_fit = popt
        
        # Calculate parameter errors
        param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]
        
        # Calculate predictions
        y_pred = exponential_func(x_fit, a_fit, b_fit)
        
        # Calculate R-squared
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate chi-squared and p-value
        try:
            # Chi-squared test for goodness of fit
            # Use observed vs expected frequencies
            chi2_stat = np.sum((y_fit - y_pred) ** 2 / y_pred)
            dof = len(y_fit) - len(popt)  # degrees of freedom
            p_value = 1 - stats.chi2.cdf(chi2_stat, dof) if dof > 0 else np.nan
        except:
            chi2_stat = np.nan
            p_value = np.nan
            
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_fit - y_pred) ** 2))
        
        # Estimate RMSE error using bootstrap
        rmse_error = calculate_exponential_rmse_error_bootstrap(x_fit, y_fit, popt)
        
        return popt, param_errors, r_squared, p_value, rmse, rmse_error, chi2_stat
        
    except Exception as e:
        return None, None, None, None, None, None

def calculate_exponential_rmse_error_bootstrap(x_data, y_data, fit_params, n_bootstrap=50):
    """
    Calculate RMSE error for exponential fit using bootstrap method.
    
    Args:
        x_data (array): x values
        y_data (array): y values  
        fit_params (tuple): fitted parameters (a, b)
        n_bootstrap (int): number of bootstrap samples
        
    Returns:
        float: bootstrap estimate of RMSE error
    """
    from scipy.optimize import curve_fit
    
    def exponential_func(x, a, b):
        return a * np.exp(-b * x)
    
    try:
        bootstrap_rmse = []
        
        for _ in range(n_bootstrap):
            try:
                # Bootstrap resample the data
                indices = np.random.choice(len(x_data), size=len(x_data), replace=True)
                boot_x = x_data[indices]
                boot_y = y_data[indices]
                
                # Fit exponential to bootstrap sample
                popt, _ = curve_fit(exponential_func, boot_x, boot_y, 
                                  p0=fit_params, 
                                  bounds=([0, 0], [np.inf, np.inf]),
                                  maxfev=5000)
                
                # Calculate RMSE for this bootstrap sample
                y_pred = exponential_func(boot_x, *popt)
                boot_rmse = np.sqrt(np.mean((boot_y - y_pred) ** 2))
                bootstrap_rmse.append(boot_rmse)
                
            except:
                continue
        
        if len(bootstrap_rmse) > 10:  # Need sufficient successful bootstraps
            return np.std(bootstrap_rmse)
        else:
            return np.nan
            
    except Exception as e:
        return np.nan

def plot_1_trend_duration_distribution_pooled(pooled_trends_df, save_path, experiment_name, plot_number=1):
    """
    Plot 1: Trend duration distribution with geometric fits using pooled data from 20 repetitions.
    """
    output_folder = os.path.join(save_path, 'trend_acceleration_plots')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_up = axes[0]
    ax_down = axes[1]
    
    # Ascending trends
    up_trends = pooled_trends_df[pooled_trends_df['direction'] == 1]
    up_durations = up_trends['duration'].values
    
    if len(up_durations) > 0:
        bins_up = np.arange(up_durations.min(), up_durations.max() + 2) - 0.5
        counts_up, bins_up, _ = ax_up.hist(up_durations, bins=bins_up, density=False, alpha=0.7, 
                                         color=ASCENDING_COLOR, edgecolor='black', label='Uptrends')
        
        geom_params_up, geom_errors_up, chi2_up, chi2_p_up, rmse_up, hist_data_up = fit_geometric_to_duration(up_durations)
        
        if geom_params_up is not None:
            loc, p = geom_params_up
            p_error_up = geom_errors_up[1] if geom_errors_up is not None and len(geom_errors_up) > 1 else np.nan
            label_up = f'Geometric Fit (p={p:.3f}±{p_error_up:.3f})' if pd.notna(p_error_up) else f'Geometric Fit (p={p:.3f})'
            
            discrete_x = np.arange(1, int(up_durations.max()) + 1)
            pmf_values = geom.pmf(discrete_x, p) * len(up_durations)
            ax_up.bar(discrete_x, pmf_values, alpha=0.6, color='red', width=0.4, label=label_up)
            
            # Calculate RMSE error using bootstrap for geometric distribution
            rmse_error_up = calculate_geometric_rmse_error_bootstrap(up_durations, geom_params_up)
            
            # Collect fit metrics
            collect_fit_metrics(experiment_name, plot_number, 'Ascending', 'Duration', 'Geometric',
                               geom_params_up, geom_errors_up, 'Chi2', chi2_up, chi2_p_up, rmse_up, rmse_error_up)
            
            ALL_FIT_METRICS.append({
                'Experiment': experiment_name,
                'Plot_Number': plot_number,
                'Distribution': 'Geometric',
                'Direction': 'Ascending',
                'Metric_Type': 'Duration',
                'Chi2_Stat': chi2_up,
                'Chi2_p_value': chi2_p_up,
                'RMSE': rmse_up,
                'Param_loc': loc,
                'Param_p': p,
                'Param_p_error': p_error_up,
            })
        
        # Collect statistical metrics
        collect_statistical_metrics(experiment_name, plot_number, 'Ascending', 'Trend_Duration', up_durations)
        
        _add_statistical_summary_to_plot(ax_up, up_durations, 'duration', 'Trend Duration (Days)', 
                                       'Trend Length Distribution', 'Ascending', 'Ascending', 
                                       experiment_name, plot_number, metric_abbr='Dur', 
                                       geom_fit_params=geom_params_up, geom_fit_rmse=rmse_up, 
                                       geom_fit_errors=geom_errors_up, geom_chi2=chi2_up, geom_chi2_p=chi2_p_up)
    else:
        ax_up.text(0.5, 0.5, 'No ascending trends found', ha='center', va='center', transform=ax_up.transAxes)
    
    ax_up.set_title('(a) Uptrends')
    ax_up.set_xlabel('Trend Duration (Ticks)')
    ax_up.set_ylabel('Frequency (log)')
    ax_up.set_yscale('log')
    ax_up.legend()
    ax_up.grid(True, which="both", ls="--", alpha=0.3)
    
    # Descending trends
    down_trends = pooled_trends_df[pooled_trends_df['direction'] == -1]
    down_durations = down_trends['duration'].values
    
    if len(down_durations) > 0:
        bins_down = np.arange(down_durations.min(), down_durations.max() + 2) - 0.5
        counts_down, bins_down, _ = ax_down.hist(down_durations, bins=bins_down, density=False, alpha=0.7, 
                                               color=DESCENDING_COLOR, edgecolor='black', label='Downtrends')
        
        geom_params_down, geom_errors_down, chi2_down, chi2_p_down, rmse_down, hist_data_down = fit_geometric_to_duration(down_durations)
        
        if geom_params_down is not None:
            loc, p = geom_params_down
            p_error_down = geom_errors_down[1] if geom_errors_down is not None and len(geom_errors_down) > 1 else np.nan
            label_down = f'Geometric Fit (p={p:.3f}±{p_error_down:.3f})' if pd.notna(p_error_down) else f'Geometric Fit (p={p:.3f})'
            
            discrete_x = np.arange(1, int(down_durations.max()) + 1)
            pmf_values = geom.pmf(discrete_x, p) * len(down_durations)
            ax_down.bar(discrete_x, pmf_values, alpha=0.6, color='red', width=0.4, label=label_down)
            
            # Calculate RMSE error using bootstrap for geometric distribution
            rmse_error_down = calculate_geometric_rmse_error_bootstrap(down_durations, geom_params_down)
            
            # Collect fit metrics
            collect_fit_metrics(experiment_name, plot_number, 'Descending', 'Duration', 'Geometric',
                               geom_params_down, geom_errors_down, 'Chi2', chi2_down, chi2_p_down, rmse_down, rmse_error_down)
            
            ALL_FIT_METRICS.append({
                'Experiment': experiment_name,
                'Plot_Number': plot_number,
                'Distribution': 'Geometric',
                'Direction': 'Descending',
                'Metric_Type': 'Duration',
                'Chi2_Stat': chi2_down,
                'Chi2_p_value': chi2_p_down,
                'RMSE': rmse_down,
                'Param_loc': loc,
                'Param_p': p,
                'Param_p_error': p_error_down,
            })
        
        # Collect statistical metrics
        collect_statistical_metrics(experiment_name, plot_number, 'Descending', 'Trend_Duration', down_durations)
        
        _add_statistical_summary_to_plot(ax_down, down_durations, 'duration', 'Sequence Duration (Days)', 
                                       'Trend Length Distribution', 'Descending', 'Descending', 
                                       experiment_name, plot_number, metric_abbr='Dur', 
                                       geom_fit_params=geom_params_down, geom_fit_rmse=rmse_down, 
                                       geom_fit_errors=geom_errors_down, geom_chi2=chi2_down, geom_chi2_p=chi2_p_down)
    else:
        ax_down.text(0.5, 0.5, 'No descending trends found', ha='center', va='center', transform=ax_down.transAxes)
    
    ax_down.set_title('(b) Downtrends')
    ax_down.set_xlabel('Trend Duration (Ticks)')
    ax_down.set_ylabel('Frequency (log)')
    ax_down.set_yscale('log')
    ax_down.legend()
    ax_down.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f'plot_{plot_number:02d}_trend_duration_distribution_{experiment_name}.png'
    plt.savefig(f'{output_folder}/{plot_filename}', dpi=300, bbox_inches='tight')
    plt.close()

def plot_2_trend_duration_exponential_pooled(pooled_trends_df, save_path, experiment_name, plot_number=2):
    """
    Plot 2: Trend duration distribution with exponential fits using points instead of bars.
    """
    output_folder = os.path.join(save_path, 'trend_acceleration_plots')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_up = axes[0]
    ax_down = axes[1]
    
    # Ascending trends
    up_trends = pooled_trends_df[pooled_trends_df['direction'] == 1]
    up_durations = up_trends['duration'].values
    
    if len(up_durations) > 0:
        # Calculate histogram data for points
        bins_up = np.arange(up_durations.min(), up_durations.max() + 2) - 0.5
        counts_up, bin_edges = np.histogram(up_durations, bins=bins_up)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot points instead of bars
        ax_up.scatter(bin_centers, counts_up, color=ASCENDING_COLOR, s=50, alpha=0.8, 
                      edgecolors='black', linewidth=1, label='Uptrends', zorder=3)
        
        # Fit exponential model y = a * e^(-b * x)
        valid_mask = counts_up > 0
        if np.sum(valid_mask) >= 3:
            x_fit = bin_centers[valid_mask]
            y_fit = counts_up[valid_mask]
            
            exp_params, exp_errors, r_squared, p_value, rmse, rmse_error, chi2_stat = fit_exponential_to_data(x_fit, y_fit)
            
            if exp_params is not None:
                a_fit, b_fit = exp_params
                a_error, b_error = exp_errors if exp_errors is not None else [np.nan, np.nan]
                
                # Generate smooth curve for plotting
                x_smooth = np.linspace(x_fit.min(), x_fit.max(), 100)
                y_smooth = a_fit * np.exp(-b_fit * x_smooth)
                
                label_up = f'Exponential Fit (a={a_fit:.1f}±{a_error:.1f}, b={b_fit:.3f}±{b_error:.3f})'
                ax_up.plot(x_smooth, y_smooth, color='red', linewidth=2, label=label_up)
                
                # Collect fit metrics
                collect_fit_metrics(experiment_name, plot_number, 'Ascending', 'Duration', 'Exponential',
                                   exp_params, exp_errors, 'χ²', chi2_stat, p_value, rmse, rmse_error)
                
                ALL_FIT_METRICS.append({
                    'Experiment': experiment_name,
                    'Plot_Number': plot_number,
                    'Distribution': 'Exponential',
                    'Direction': 'Ascending',
                    'Metric_Type': 'Duration',
                    'R_squared': r_squared,
                    'p_value': p_value,
                    'RMSE': rmse,
                    'Chi2': chi2_stat,
                    'Param_a': a_fit,
                    'Param_b': b_fit,
                    'Param_a_error': a_error,
                    'Param_b_error': b_error,
                })
                
                # Add goodness-of-fit metrics box
                metrics_text = f'Exponential Fit\n'
                metrics_text += f'χ² = {chi2_stat:.3f}\n'
                metrics_text += f'p-value = {p_value:.3f}\n'
                metrics_text += f'RMSE = {rmse:.3f}\n'
                metrics_text += f'R² = {r_squared:.3f}\n'
                
                ax_up.text(0.98, 0.98, metrics_text, transform=ax_up.transAxes, 
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.2, 
                                   edgecolor='black', linewidth=1.5),
                          fontsize=9)
        
        # Collect statistical metrics
        collect_statistical_metrics(experiment_name, plot_number, 'Ascending', 'Trend_Duration', up_durations)
    else:
        ax_up.text(0.5, 0.5, 'No ascending trends found', ha='center', va='center', transform=ax_up.transAxes)
    
    ax_up.set_title('(a) Uptrends')
    ax_up.set_xlabel('Trend Duration (Ticks)')
    ax_up.set_ylabel('Frequency (log)')
    ax_up.set_yscale('log')
    ax_up.legend()
    ax_up.grid(True, which="both", ls="--", alpha=0.3)
    
    # Descending trends
    down_trends = pooled_trends_df[pooled_trends_df['direction'] == -1]
    down_durations = down_trends['duration'].values
    
    if len(down_durations) > 0:
        # Calculate histogram data for points
        bins_down = np.arange(down_durations.min(), down_durations.max() + 2) - 0.5
        counts_down, bin_edges = np.histogram(down_durations, bins=bins_down)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot points instead of bars
        ax_down.scatter(bin_centers, counts_down, color=DESCENDING_COLOR, s=50, alpha=0.8, 
                        edgecolors='black', linewidth=1, label='Downtrends', zorder=3)
        
        # Fit exponential model y = a * e^(-b * x)
        valid_mask = counts_down > 0
        if np.sum(valid_mask) >= 3:
            x_fit = bin_centers[valid_mask]
            y_fit = counts_down[valid_mask]
            
            exp_params, exp_errors, r_squared, p_value, rmse, rmse_error, chi2_stat = fit_exponential_to_data(x_fit, y_fit)
            
            if exp_params is not None:
                a_fit, b_fit = exp_params
                a_error, b_error = exp_errors if exp_errors is not None else [np.nan, np.nan]
                
                # Generate smooth curve for plotting
                x_smooth = np.linspace(x_fit.min(), x_fit.max(), 100)
                y_smooth = a_fit * np.exp(-b_fit * x_smooth)
                
                label_down = f'Exponential Fit (a={a_fit:.1f}±{a_error:.1f}, b={b_fit:.3f}±{b_error:.3f})'
                ax_down.plot(x_smooth, y_smooth, color='red', linewidth=2, label=label_down)
                
                # Collect fit metrics
                collect_fit_metrics(experiment_name, plot_number, 'Descending', 'Duration', 'Exponential',
                                   exp_params, exp_errors, 'χ²', chi2_stat, p_value, rmse, rmse_error)
                
                ALL_FIT_METRICS.append({
                    'Experiment': experiment_name,
                    'Plot_Number': plot_number,
                    'Distribution': 'Exponential',
                    'Direction': 'Descending',
                    'Metric_Type': 'Duration',
                    'R_squared': r_squared,
                    'p_value': p_value,
                    'RMSE': rmse,
                    'Chi2': chi2_stat,
                    'Param_a': a_fit,
                    'Param_b': b_fit,
                    'Param_a_error': a_error,
                    'Param_b_error': b_error,
                })
                
                # Add goodness-of-fit metrics box
                metrics_text = f'Exponential Fit\n'
                metrics_text += f'χ² = {chi2_stat:.3f}\n'
                metrics_text += f'p-value = {p_value:.3f}\n'
                metrics_text += f'RMSE = {rmse:.3f}\n'
                metrics_text += f'R² = {r_squared:.3f}\n'
                
                ax_down.text(0.98, 0.98, metrics_text, transform=ax_down.transAxes, 
                              verticalalignment='top', horizontalalignment='right',
                              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.2, 
                                       edgecolor='black', linewidth=1.5),
                              fontsize=9)
        
        # Collect statistical metrics
        collect_statistical_metrics(experiment_name, plot_number, 'Descending', 'Trend_Duration', down_durations)
    else:
        ax_down.text(0.5, 0.5, 'No descending trends found', ha='center', va='center', transform=ax_down.transAxes)
    
    ax_down.set_title('(b) Downtrends')
    ax_down.set_xlabel('Trend Duration (Ticks)')
    ax_down.set_ylabel('Frequency (log)')
    ax_down.set_yscale('log')
    ax_down.legend()
    ax_down.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f'plot_{plot_number:02d}_trend_duration_exponential_{experiment_name}.png'
    plt.savefig(f'{output_folder}/{plot_filename}', dpi=300, bbox_inches='tight')
    plt.close()

def collect_statistical_metrics(experiment_name, plot_number, subplot, data_name, data):
    """
    Collect comprehensive statistical metrics for a given dataset and add to global list.
    Similar to plots_multiple_repetitions.py collect_statistics function.
    """
    global GLOBAL_COMBINED_METRICS
    
    # Calculate comprehensive statistics with bootstrap errors
    stats_dict = calculate_statistics_with_errors(data)
    
    for stat_name, stat_value in stats_dict.items():
        if stat_name.endswith('_error') or stat_name == 'count':
            continue
        
        error_key = f"{stat_name}_error"
        error_value = stats_dict.get(error_key, np.nan)
        
        GLOBAL_COMBINED_METRICS.append({
            'experiment': experiment_name,
            'plot_number': plot_number,
            'subplot': subplot if subplot is not None else 'N/A',
            'data_name': data_name,
            'metric_type': 'statistical',
            'metric_name': stat_name,
            'value': stat_value,
            'error': error_value,
            'distribution': 'N/A',
            'goodness_metric': 'N/A',
            'goodness_value': np.nan,
            'goodness_p_value': np.nan
        })
    

def collect_fit_metrics(experiment_name, plot_number, subplot, metric_type, distribution_name, 
                        fit_params, fit_errors, goodness_metric, goodness_value, goodness_p_value=None, rmse_value=None, rmse_error=None):
    """
    Collect model fitting parameters and goodness-of-fit metrics.
    """
    global GLOBAL_COMBINED_METRICS
    
    # Define parameter names for different distributions
    param_names = {
        'Geometric': ['loc', 'p'],
        'GenExtreme': ['c', 'loc', 'scale'],
        'LogNormal': ['s', 'loc', 'scale'],
        'Burr12': ['c', 'd', 'loc', 'scale']
    }
    
    if distribution_name in param_names and fit_params is not None:
        param_labels = param_names[distribution_name]
        
        for i, (param_name, param_value) in enumerate(zip(param_labels, fit_params)):
            param_error = fit_errors[i] if fit_errors is not None and i < len(fit_errors) else np.nan
            
            GLOBAL_COMBINED_METRICS.append({
                'experiment': experiment_name,
                'plot_number': plot_number,
                'subplot': subplot if subplot is not None else 'N/A',
                'data_name': f"{metric_type}_{distribution_name}_fit",
                'metric_type': 'model_parameter',
                'metric_name': f"{distribution_name}_{param_name}",
                'value': param_value,
                'error': param_error,
                'distribution': distribution_name,
                'goodness_metric': goodness_metric,
                'goodness_value': goodness_value,
                'goodness_p_value': goodness_p_value if goodness_p_value is not None else np.nan
            })
        
        # Add the goodness-of-fit metric as a separate entry
        GLOBAL_COMBINED_METRICS.append({
            'experiment': experiment_name,
            'plot_number': plot_number,
            'subplot': subplot if subplot is not None else 'N/A',
            'data_name': f"{metric_type}_{distribution_name}_fit",
            'metric_type': 'goodness_of_fit',
            'metric_name': goodness_metric,
            'value': goodness_value,
            'error': np.nan,
            'distribution': distribution_name,
            'goodness_metric': goodness_metric,
            'goodness_value': goodness_value,
            'goodness_p_value': goodness_p_value
        })
        
        # Add the p-value as a separate metric entry
        if goodness_p_value is not None:
            GLOBAL_COMBINED_METRICS.append({
                'experiment': experiment_name,
                'plot_number': plot_number,
                'subplot': subplot if subplot is not None else 'N/A',
                'data_name': f"{metric_type}_{distribution_name}_fit",
                'metric_type': 'goodness_of_fit_p_value',
                'metric_name': f"{goodness_metric}_p_value",
                'value': goodness_p_value,
                'error': np.nan,
                'distribution': distribution_name,
                'goodness_metric': goodness_metric,
                'goodness_value': goodness_value,
                'goodness_p_value': goodness_p_value
            })
        
        # Add RMSE as a separate metric entry if provided
        if rmse_value is not None:
            GLOBAL_COMBINED_METRICS.append({
                'experiment': experiment_name,
                'plot_number': plot_number,
                'subplot': subplot if subplot is not None else 'N/A',
                'data_name': f"{metric_type}_{distribution_name}_fit",
                'metric_type': 'goodness_of_fit',
                'metric_name': 'RMSE',
                'value': rmse_value,
                'error': rmse_error if rmse_error is not None else np.nan,
                'distribution': distribution_name,
                'goodness_metric': 'RMSE',
                'goodness_value': rmse_value,
                'goodness_p_value': np.nan
            })

def generate_comprehensive_tables(output_folder, experiment_name):
    """
    Generate comprehensive CSV and LaTeX tables combining statistical and fitting metrics.
    Similar to plots_multiple_repetitions.py generate_and_save_stats_tables function.
    """
    global GLOBAL_COMBINED_METRICS
    
    if not GLOBAL_COMBINED_METRICS:
        return
    
    df_metrics = pd.DataFrame(GLOBAL_COMBINED_METRICS)
    
    # Create filenames
    csv_filename = os.path.join(output_folder, f'{experiment_name}_comprehensive_metrics.csv')
    
    # Combine value and error into a single column
    df_metrics['value_with_error'] = df_metrics.apply(
        lambda row: f"{row['value']:.4f} +/- {row['error']:.4f}" if pd.notna(row['error']) and row['error'] != 0 else f"{row['value']:.4f}",
        axis=1
    )
    
    # Create simplified DataFrame for CSV with additional columns needed for calibration
    simplified_df = df_metrics[['plot_number', 'subplot', 'metric_name', 'value', 'error', 'value_with_error', 'distribution', 'goodness_p_value']].copy()
    
    # Save to CSV
    simplified_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

def process_single_experiment(experiment_folder, experiment_name):
    """
    Process a single experiment: load repetitions, pool data, and generate all plots.
    
    Args:
        experiment_folder (str): Path to experiment folder
        experiment_name (str): Name of experiment for plot titles
        
    Returns:
        bool: True if successful, False otherwise
    """
    global ALL_FIT_METRICS, ALL_STATISTICAL_METRICS, GLOBAL_COMBINED_METRICS
    # Reset global metrics for this experiment
    ALL_FIT_METRICS = []
    ALL_STATISTICAL_METRICS = []
    GLOBAL_COMBINED_METRICS = []
    
    # Load repetitions data
    repetitions_data = load_repetitions_for_experiment(experiment_folder)
    if repetitions_data is None:
        return
    
    # Analyze trends from all repetitions
    pooled_trends_df, pooled_sequences, sequences_by_repetition = analyze_trends_from_repetitions(repetitions_data)
    
    if pooled_trends_df.empty:
        return
    
    try:
        # Generate only 2 plots
        plot_1_trend_duration_distribution_pooled(pooled_trends_df, experiment_folder, experiment_name, plot_number=1)
        plot_2_trend_duration_exponential_pooled(pooled_trends_df, experiment_folder, experiment_name, plot_number=2)
        
        # Generate comprehensive tables combining all metrics
        output_folder = os.path.join(experiment_folder, 'trend_acceleration_plots')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        generate_comprehensive_tables(output_folder, experiment_name)
        
        return True  # Return success
        
    except Exception as e:
        return False

def process_experiment_worker(args):
    """
    Worker function for parallel processing of a single experiment.
    
    Args:
        args: Tuple containing (experiments_root, exp_folder)
        
    Returns:
        Tuple: (exp_folder, success_status, processing_time)
    """
    experiments_root, exp_folder = args
    start_time = time.time()
    
    try:
        exp_path = os.path.join(experiments_root, exp_folder)
        success = process_single_experiment(exp_path, exp_folder)
        processing_time = time.time() - start_time
        
        return (exp_folder, success, processing_time)
        
    except Exception as e:
        processing_time = time.time() - start_time
        return (exp_folder, False, processing_time)

def process_all_experiments_parallel(max_workers=None):
    """
    Process all experiments in experiments_multi folder in parallel using multiple CPU cores.
    
    Args:
        max_workers (int): Maximum number of worker processes (default: min(29, cpu_count))
    """
    # Look for experiment folders - check multiple possible locations
    possible_folders = ['experiments_multi']
    # Also check for temporary experiment folders
    temp_folders = [f for f in os.listdir('.') if f.startswith('experiments_temp_')]
    possible_folders.extend(temp_folders)
    
    experiments_root = None
    for folder in possible_folders:
        if os.path.exists(folder):
            experiments_root = folder
            break
    
    if experiments_root is None:
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
    process_all_experiments()
