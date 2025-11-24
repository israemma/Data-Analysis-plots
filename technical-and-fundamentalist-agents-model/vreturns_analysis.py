import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import pandas as pd
import warnings
from scipy.stats import gamma, chisquare, cramervonmises
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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

# Color constants
ASCENDING_COLOR = '#2ca02c'  # Green
DESCENDING_COLOR = '#d62728' # Red

# Data Loading Functions

def load_repetitions_for_experiment(experiment_folder):
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
                # Exclude first 500 steps as warm-up period (consistent with other scripts)
                if len(df) > 500:
                    df_filtered = df.iloc[500:]
                    repetitions_data.append(df_filtered['market_price'])
            except Exception as e:
                return None
    
    if len(repetitions_data) == 0:
        return None
    
    return repetitions_data

# Trend Analysis Functions

def identify_trends(prices):
    """Identify trends in price data using simple up/down classification."""
    if len(prices) < 2:
        return []
    
    trends = []
    current_trend = None
    trend_start = 0
    
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:  # Price going up
            direction = 'up'
        elif prices[i] < prices[i-1]:  # Price going down
            direction = 'down'
        else:  # Price unchanged
            continue
            
        if current_trend is None:
            current_trend = direction
            trend_start = i-1
        elif current_trend != direction:
            # Trend change detected
            trends.append({
                'start': trend_start,
                'end': i-1,
                'direction': current_trend,
                'duration': i-1 - trend_start
            })
            current_trend = direction
            trend_start = i-1
    
    # Add the last trend
    if current_trend is not None:
        trends.append({
            'start': trend_start,
            'end': len(prices)-1,
            'direction': current_trend,
            'duration': len(prices)-1 - trend_start
        })
    
    return trends

def calculate_vreturns(prices, trends):
    """Calculate VReturns for each trend.
    
    For each trend of duration n days, calculate:
    (log(P_2)-log(P_1))/1, (log(P_3)-log(P_1))/2, ..., (log(P_n+1)-log(P_1))/n
    
    Args:
        prices: Array of price values
        trends: List of trend dictionaries
    
    Returns:
        np.array of vreturns
    """
    vreturns = []
    
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
                    vreturn = (log_p_i - log_p_start) / i
                    vreturns.append(vreturn)
    
    return np.array(vreturns)

def extract_vreturns_from_repetitions(repetitions_data):
    """Extract VReturns from all repetitions and pool them together."""
    all_vreturns = []
    
    for rep_data in repetitions_data:
        if rep_data.empty:
            continue
            
        # Clean data
        prices = rep_data.values
        prices_clean = prices[~np.isnan(prices)]
        if len(prices_clean) < 2:
            continue
            
        trends = identify_trends(prices_clean)
        if not trends:
            continue
            
        vreturns = calculate_vreturns(prices_clean, trends)
        if len(vreturns) > 0:
            all_vreturns.extend(vreturns)
    
    return np.array(all_vreturns)

# KL Divergence Calculation

def calculate_kl_divergence(positive_data, negative_data_abs, method='kde', bins=50):
    """
    Calculate Kullback-Leibler divergence between positive and absolute negative distributions.
    
    Parameters:
    -----------
    positive_data : array-like
        Positive VReturns values
    negative_data_abs : array-like
        Absolute values of negative VReturns
    method : str
        Method to estimate probability densities ('kde' or 'histogram')
    bins : int
        Number of bins for histogram method
        
    Returns:
    --------
    kl_divergence : float
        KL divergence D(P||Q) where P is positive distribution and Q is absolute negative distribution
    """
    if len(positive_data) == 0 or len(negative_data_abs) == 0:
        return np.nan
    
    try:
        if method == 'kde':
            # Use KDE to estimate probability densities
            kde_pos = gaussian_kde(positive_data)
            kde_neg = gaussian_kde(negative_data_abs)
            
            # Define common support range
            x_min = min(np.min(positive_data), np.min(negative_data_abs))
            x_max = max(np.max(positive_data), np.max(negative_data_abs))
            
            # Create evaluation points
            x_eval = np.linspace(x_min, x_max, 1000)
            
            # Evaluate PDFs
            p_pos = kde_pos(x_eval)
            p_neg = kde_neg(x_eval)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            p_pos = np.maximum(p_pos, epsilon)
            p_neg = np.maximum(p_neg, epsilon)
            
            # Normalize to ensure they are proper probability distributions
            p_pos = p_pos / np.trapz(p_pos, x_eval)
            p_neg = p_neg / np.trapz(p_neg, x_eval)
            
            # Calculate KL divergence using trapezoidal integration
            # D(P||Q) = ∫ P(x) * log(P(x)/Q(x)) dx
            integrand = p_pos * np.log(p_pos / p_neg)
            kl_divergence = np.trapz(integrand, x_eval)
            
        elif method == 'histogram':
            # Use histogram to estimate probability densities
            x_min = min(np.min(positive_data), np.min(negative_data_abs))
            x_max = max(np.max(positive_data), np.max(negative_data_abs))
            
            # Create common bins
            bin_edges = np.linspace(x_min, x_max, bins + 1)
            
            # Calculate histograms
            hist_pos, _ = np.histogram(positive_data, bins=bin_edges, density=True)
            hist_neg, _ = np.histogram(negative_data_abs, bins=bin_edges, density=True)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            hist_pos = np.maximum(hist_pos, epsilon)
            hist_neg = np.maximum(hist_neg, epsilon)
            
            # Normalize histograms
            bin_width = bin_edges[1] - bin_edges[0]
            hist_pos = hist_pos / (np.sum(hist_pos) * bin_width)
            hist_neg = hist_neg / (np.sum(hist_neg) * bin_width)
            
            # Calculate KL divergence
            # D(P||Q) = Σ P(i) * log(P(i)/Q(i)) * bin_width
            kl_divergence = np.sum(hist_pos * np.log(hist_pos / hist_neg)) * bin_width
            
        else:
            raise ValueError("Method must be 'kde' or 'histogram'")
            
        return kl_divergence
        
    except Exception as e:
        return np.nan

def calculate_goodness_of_fit_metrics(data, fitted_distribution):
    """
    Calculate three goodness-of-fit metrics for a fitted distribution.
    
    Args:
        data: numpy array of observed data
        fitted_distribution: scipy.stats distribution object (fitted)
        
    Returns:
        dict: Dictionary containing chi-squared, Cramér-von Mises, and Kolmogorov-Smirnov test results
    """
    metrics = {}
    
    try:
        # 1. Chi-squared test
        # Create bins for chi-squared test
        n_bins = min(20, max(5, len(data) // 10))  # Adaptive number of bins
        observed_freq, bin_edges = np.histogram(data, bins=n_bins)
        
        # Calculate expected frequencies
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        expected_freq = []
        
        for i in range(len(bin_edges) - 1):
            prob = fitted_distribution.cdf(bin_edges[i+1]) - fitted_distribution.cdf(bin_edges[i])
            expected_freq.append(prob * len(data))
        
        expected_freq = np.array(expected_freq)
        
        # Remove bins with very low expected frequency
        mask = expected_freq >= 5
        if np.sum(mask) >= 3:  # Need at least 3 bins
            chi2_stat, chi2_pvalue = chisquare(observed_freq[mask], expected_freq[mask])
            metrics['chi_squared'] = {'statistic': chi2_stat, 'pvalue': chi2_pvalue}
        else:
            metrics['chi_squared'] = {'statistic': np.nan, 'pvalue': np.nan}
            
    except Exception as e:
        metrics['chi_squared'] = {'statistic': np.nan, 'pvalue': np.nan}
    
    try:
        # 2. Cramér-von Mises test
        cvm_result = cramervonmises(data, fitted_distribution.cdf)
        metrics['cramer_von_mises'] = {'statistic': cvm_result.statistic, 'pvalue': cvm_result.pvalue}
        
    except Exception as e:
        metrics['cramer_von_mises'] = {'statistic': np.nan, 'pvalue': np.nan}
    
    try:
        # 3. Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.kstest(data, fitted_distribution.cdf)
        metrics['kolmogorov_smirnov'] = {'statistic': ks_stat, 'pvalue': ks_pvalue}
        
    except Exception as e:
        metrics['kolmogorov_smirnov'] = {'statistic': np.nan, 'pvalue': np.nan}
    
    try:
        # 4. MSE-based metrics (comparing empirical vs theoretical PDF)
        # Create histogram for empirical PDF
        n_bins = min(50, max(10, len(data) // 20))
        hist_counts, bin_edges = np.histogram(data, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate theoretical PDF at bin centers
        theoretical_pdf = fitted_distribution.pdf(bin_centers)
        
        # Remove bins with zero counts to avoid issues
        mask = hist_counts > 0
        if np.sum(mask) >= 3:
            empirical_pdf = hist_counts[mask]
            theoretical_pdf_masked = theoretical_pdf[mask]
            
            # Mean Squared Error (MSE)
            mse = np.mean((empirical_pdf - theoretical_pdf_masked)**2)
            
            # Root Mean Squared Error (RMSE)
            rmse = np.sqrt(mse)
            
            # Mean Absolute Error (MAE)
            mae = np.mean(np.abs(empirical_pdf - theoretical_pdf_masked))
            
            # R-squared (coefficient of determination)
            ss_res = np.sum((empirical_pdf - theoretical_pdf_masked)**2)
            ss_tot = np.sum((empirical_pdf - np.mean(empirical_pdf))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            metrics['mse_metrics'] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r_squared': r_squared
            }
        else:
            metrics['mse_metrics'] = {
                'mse': np.nan,
                'rmse': np.nan,
                'mae': np.nan,
                'r_squared': np.nan
            }
            
    except Exception as e:
        metrics['mse_metrics'] = {
            'mse': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'r_squared': np.nan
        }
    
    return metrics

# Plotting Functions

def plot_vreturns_histogram_linear_separated(vreturns_by_index, save_path='VReturns_Analysis_Plots'):
    """
    Generate separate histogram plots of VReturns with bars for each index in linear-linear scale,
    separated into positive and negative parts with KL divergence calculation.
    """
    hist_save_path = save_path
    os.makedirs(hist_save_path, exist_ok=True)
    
    # Define colors for each index (consistent with other plots)
    colors = {
        'DJIA': '#1f77b4',    # Dark blue
        'DAX': '#d62728',     # Dark red  
        'IPC': '#2ca02c',     # Dark green
        'Nikkei': '#ff7f0e'   # Dark orange
    }
    
    plot_filenames = []
    kl_results = {}
    
    
    # Generate separate plot for each index
    for index_name, vreturns in vreturns_by_index.items():
        if len(vreturns) > 0:
            
            # Separate positive and negative VReturns
            positive_vreturns = vreturns[vreturns > 0]
            negative_vreturns = vreturns[vreturns < 0]
            
            
            # Calculate KL divergence between positive and negative distributions
            if len(positive_vreturns) > 0 and len(negative_vreturns) > 0:
                # Using absolute values for negative part to compare distributions properly
                negative_abs = np.abs(negative_vreturns)
                kl_div_pos_neg = calculate_kl_divergence(positive_vreturns, negative_abs, method='kde')
                kl_div_neg_pos = calculate_kl_divergence(negative_abs, positive_vreturns, method='kde')
                
                
                kl_results[index_name] = {
                    'kl_pos_neg': kl_div_pos_neg,
                    'kl_neg_pos': kl_div_neg_pos
                }
            else:
                kl_results[index_name] = {'kl_pos_neg': np.nan, 'kl_neg_pos': np.nan}
            
            # Create figure with two subplots (side by side)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot positive VReturns histogram (left subplot)
            if len(positive_vreturns) > 0:
                ax1.hist(positive_vreturns, bins=50, density=False, alpha=0.7, 
                        color='#1f77b4', 
                        edgecolor='black', linewidth=0.5, label='Positive Data')
                
                # Calculate statistics for positive part
                mean_pos = np.mean(positive_vreturns)
                median_pos = np.median(positive_vreturns)
                std_pos = np.std(positive_vreturns, ddof=1)
                skew_pos = stats.skew(positive_vreturns)
                kurt_pos = stats.kurtosis(positive_vreturns, fisher=True)
                
                n_pos = len(positive_vreturns)
                mean_se_pos = std_pos / np.sqrt(n_pos)  # Standard error of the mean
                skew_se_pos = np.sqrt(6 * n_pos * (n_pos - 1) / ((n_pos - 2) * (n_pos + 1) * (n_pos + 3)))
                kurt_se_pos = np.sqrt(24 * n_pos * (n_pos - 1)**2 / ((n_pos - 3) * (n_pos - 2) * (n_pos + 3) * (n_pos + 5)))
                
                # Create statistics text box for positive (including KL divergence)
                stats_text_pos = f'n = {n_pos}\n'
                stats_text_pos += f'Mean = {mean_pos:.4f} ± {mean_se_pos:.4f}\n'
                stats_text_pos += f'Median = {median_pos:.4f}\n'
                stats_text_pos += f'Std = {std_pos:.4f}\n'
                stats_text_pos += f'Skewness = {float(skew_pos):.4f} ± {float(skew_se_pos):.4f}\n'
                stats_text_pos += f'Kurtosis = {float(kurt_pos):.4f} ± {float(kurt_se_pos):.4f}\n'
                # Add KL divergence information
                if index_name in kl_results and not np.isnan(kl_results[index_name]['kl_pos_neg']):
                    stats_text_pos += f'KL(Pos||Neg) = {kl_results[index_name]["kl_pos_neg"]:.4f}'
                
                ax1.text(0.72, 0.88, stats_text_pos, transform=ax1.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='none', 
                        edgecolor='black', linewidth=1, alpha=1.0))
                
                # Set y-axis to log scale
                ax1.set_yscale('log')
                
                # Ensure all data is visible on x-axis
                data_min_pos = np.min(positive_vreturns)
                data_max_pos = np.max(positive_vreturns)
                x_margin_pos = (data_max_pos - data_min_pos) * 0.05
                ax1.set_xlim(data_min_pos - x_margin_pos, data_max_pos + x_margin_pos)
                
                # Labels and formatting
                ax1.set_xlabel('VReturns (Positive)', fontsize=12)
                ax1.set_ylabel('Frequency (log)', fontsize=12)
                ax1.set_title('Positive VReturns')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Plot negative VReturns histogram (right subplot) - using absolute values
            if len(negative_vreturns) > 0:
                negative_abs_for_plot = np.abs(negative_vreturns)
                ax2.hist(negative_abs_for_plot, bins=50, density=False, alpha=0.7, 
                        color='#1f77b4', 
                        edgecolor='black', linewidth=0.5, label='Negative Data (|values|)')
                
                # Calculate statistics for negative part (using absolute values)
                mean_neg = np.mean(negative_abs_for_plot)
                median_neg = np.median(negative_abs_for_plot)
                std_neg = np.std(negative_abs_for_plot, ddof=1)
                skew_neg = stats.skew(negative_abs_for_plot)
                kurt_neg = stats.kurtosis(negative_abs_for_plot, fisher=True)
                
                n_neg = len(negative_abs_for_plot)
                mean_se_neg = std_neg / np.sqrt(n_neg)  # Standard error of the mean
                skew_se_neg = np.sqrt(6 * n_neg * (n_neg - 1) / ((n_neg - 2) * (n_neg + 1) * (n_neg + 3)))
                kurt_se_neg = np.sqrt(24 * n_neg * (n_neg - 1)**2 / ((n_neg - 3) * (n_neg - 2) * (n_neg + 3) * (n_neg + 5)))
                
                # Create statistics text box for negative (including KL divergence)
                stats_text_neg = f'n = {n_neg}\n'
                stats_text_neg += f'Mean = {mean_neg:.4f} ± {mean_se_neg:.4f}\n'
                stats_text_neg += f'Median = {median_neg:.4f}\n'
                stats_text_neg += f'Std = {std_neg:.4f}\n'
                stats_text_neg += f'Skewness = {float(skew_neg):.4f} ± {float(skew_se_neg):.4f}\n'
                stats_text_neg += f'Kurtosis = {float(kurt_neg):.4f} ± {float(kurt_se_neg):.4f}\n'
                # Add KL divergence information
                if index_name in kl_results and not np.isnan(kl_results[index_name]['kl_neg_pos']):
                    stats_text_neg += f'KL(Neg||Pos) = {kl_results[index_name]["kl_neg_pos"]:.4f}'
                
                ax2.text(0.72, 0.88, stats_text_neg, transform=ax2.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='none', 
                        edgecolor='black', linewidth=1, alpha=1.0))
                
                # Set y-axis to log scale
                ax2.set_yscale('log')
                
                # Ensure all data is visible on x-axis
                data_min_neg = np.min(negative_abs_for_plot)
                data_max_neg = np.max(negative_abs_for_plot)
                x_margin_neg = (data_max_neg - data_min_neg) * 0.05
                ax2.set_xlim(data_min_neg - x_margin_neg, data_max_neg + x_margin_neg)
                
                # Labels and formatting
                ax2.set_xlabel('|VReturns| (Negative)', fontsize=12)
                ax2.set_ylabel('Frequency (log)', fontsize=12)
                ax2.set_title('Negative VReturns (Absolute Values)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save individual plot
            plot_filename = os.path.join(hist_save_path, f'vreturns_histogram_linear_separated_{index_name.replace(" ", "_").replace("/", "_")}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_filenames.append(plot_filename)
    
    return plot_filenames, kl_results

# Processing Functions

def process_single_experiment(experiment_folder, experiment_name):
    """Process a single experiment and generate VReturns histogram and CDF plots."""
    
    # Load repetitions data
    repetitions_data = load_repetitions_for_experiment(experiment_folder)
    if repetitions_data is None:
        return None
    
    
    # Extract pooled VReturns from all repetitions
    pooled_vreturns = extract_vreturns_from_repetitions(repetitions_data)
    if len(pooled_vreturns) == 0:
        return None
    
    # Create output folder if it doesn't exist
    save_path = os.path.join(experiment_folder, 'vreturns_cdf_plots')
    os.makedirs(save_path, exist_ok=True)
    
    # Generate the original histogram plot
    # Note: The original function expects a dictionary with index names as keys
    vreturns_by_index = {experiment_name: pooled_vreturns}
    plot_filenames_original, kl_results = plot_vreturns_histogram_linear_separated(
        vreturns_by_index, save_path
    )
    
    # Generate the new CDF plot with gamma CDF fits (using all data)
    plot_filename_cdf = plot_vreturns_cdf_separated_with_gamma_fits(
        pooled_vreturns, experiment_name, save_path
    )
    
    return [plot_filenames_original, plot_filename_cdf]

def process_experiment_worker(args):
    """Worker function for parallel processing."""
    experiment_folder, experiment_name = args
    try:
        return process_single_experiment(experiment_folder, experiment_name)
    except Exception as e:
        return None

def process_all_experiments_parallel(max_workers=None, experiments_root='experiments_multi'):
    """
    Process all experiments in experiments_multi folder in parallel using multiple CPU cores.
    
    Args:
        max_workers (int): Maximum number of parallel workers
        experiments_root (str): Root folder containing experiments (can be a single experiment folder)
    """
    if not os.path.exists(experiments_root):
        return
    
    # Check if this is a single experiment folder (has rep_ subfolders)
    has_reps = any(item.startswith('rep_') for item in os.listdir(experiments_root) if os.path.isdir(os.path.join(experiments_root, item)))
    
    if has_reps:
        # This is a single experiment folder, process it directly
        exp_name = os.path.basename(experiments_root)
        result = process_single_experiment(experiments_root, exp_name)
        return
    
    # Otherwise, look for exp_ subfolders (legacy behavior)
    experiment_folders = []
    for item in os.listdir(experiments_root):
        if item.startswith('exp_') and os.path.isdir(os.path.join(experiments_root, item)):
            experiment_folders.append(item)
    
    if not experiment_folders:
        return
    
    experiment_folders.sort(key=lambda x: int(x.split('_')[1]))
    
    
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(experiment_folders))
    
    
    # Prepare arguments for parallel processing
    args_list = []
    for exp_folder in experiment_folders:
        full_path = os.path.join(experiments_root, exp_folder)
        args_list.append((full_path, exp_folder))
    
    # Process experiments in parallel
    successful_plots = 0
    failed_experiments = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_exp = {executor.submit(process_experiment_worker, args): args[1] 
                        for args in args_list}
        
        # Collect results as they complete
        for future in as_completed(future_to_exp):
            exp_name = future_to_exp[future]
            try:
                result = future.result()
                if result is not None:
                    successful_plots += 1
                else:
                    failed_experiments.append(exp_name)
            except Exception as e:
                failed_experiments.append(exp_name)


def process_all_experiments(experiments_root='experiments_multi'):
    """Process all experiments sequentially (fallback method).
    
    Args:
        experiments_root (str): Root folder containing experiments
    """
    if not os.path.exists(experiments_root):
        return
    
    # Get list of experiment folders
    experiment_folders = []
    for item in os.listdir(experiments_root):
        if item.startswith('exp_') and os.path.isdir(os.path.join(experiments_root, item)):
            experiment_folders.append(item)
    
    experiment_folders.sort(key=lambda x: int(x.split('_')[1]))
    
    
    successful_plots = 0
    failed_experiments = []
    
    for exp_folder in experiment_folders:
        full_path = os.path.join(experiments_root, exp_folder)
        try:
            result = process_single_experiment(full_path, exp_folder)
            if result is not None:
                successful_plots += 1
            else:
                failed_experiments.append(exp_folder)
        except Exception as e:
            failed_experiments.append(exp_folder)


# Main Function

def main(experiments_root='experiments_multi'):
    """Main function to run the VReturns analysis for all experiments.
    
    Args:
        experiments_root (str): Root folder containing experiments
    """
    
    try:
        # Try parallel processing first
        process_all_experiments_parallel(experiments_root=experiments_root)
    except Exception as e:
        process_all_experiments(experiments_root=experiments_root)
    

def fit_pareto_to_data(data):
    """Fit a Pareto distribution to data using MLE.
    
    For Pareto distribution: F(x) = 1 - (x_m / x)^alpha
    where x_m is the minimum value and alpha is the shape parameter.
    
    Returns:
        tuple: (pareto_dist, alpha, x_m, alpha_se, x_m_se)
    """
    if len(data) < 2:
        return None, None, None, None, None
    
    try:
        x_m = np.min(data)
        n = len(data)
        # MLE for alpha: alpha = n / sum(log(x_i / x_m))
        alpha = n / np.sum(np.log(data / x_m))
        
        # Standard error of alpha: SE(alpha) = alpha / sqrt(n)
        alpha_se = alpha / np.sqrt(n)
        
        # Standard error of x_m: SE(x_m) = x_m / (n+1) * sqrt(n)
        # This is an approximation for the minimum order statistic
        x_m_se = x_m / np.sqrt(n)
        
        # Create a Pareto distribution object
        # scipy.stats.pareto is parameterized as: F(x) = 1 - (1/(1+x/scale))^shape
        # We need to adjust: our x_m corresponds to scale, our alpha to shape
        pareto_dist = stats.pareto(b=alpha, scale=x_m)
        
        return pareto_dist, alpha, x_m, alpha_se, x_m_se
    except Exception as e:
        return None, None, None, None, None

def find_optimal_percentile_cuts(data, percentile_range=np.arange(5, 46, 5)):
    """Find optimal left and right percentile cuts to minimize KS(D) statistic.
    
    Returns:
        tuple: (trimmed_data, (left_pct, right_pct), best_ks_stat, left_threshold, right_threshold)
    """
    best_ks_stat = np.inf
    best_cuts = (0, 100)
    best_data = data
    best_left_val = np.min(data)
    best_right_val = np.max(data)
    
    for left_pct in percentile_range:
        for right_pct in percentile_range:
            left_val = np.percentile(data, left_pct)
            right_val = np.percentile(data, 100 - right_pct)
            
            if left_val >= right_val:
                continue
            
            trimmed = data[(data >= left_val) & (data <= right_val)]
            if len(trimmed) < 10:
                continue
            
            try:
                shape, loc, scale = stats.gamma.fit(trimmed, floc=0)
                fitted = stats.gamma(a=shape, loc=loc, scale=scale)
                metrics = calculate_goodness_of_fit_metrics(trimmed, fitted)
                
                ks_stat = metrics.get('kolmogorov_smirnov', {}).get('statistic', np.inf)
                
                if ks_stat < best_ks_stat:
                    best_ks_stat = ks_stat
                    best_cuts = (left_pct, right_pct)
                    best_data = trimmed
                    best_left_val = left_val
                    best_right_val = right_val
            except:
                continue
    
    return best_data, best_cuts, best_ks_stat, best_left_val, best_right_val

def save_fit_metrics_csv(data_dict, save_path, experiment_name):
    """Save fit metrics to CSV in compact format."""
    rows = []
    for part, metrics in data_dict.items():
        row = {'Part': part, 'Experiment': experiment_name}
        row.update(metrics)
        rows.append(row)
    
    csv_path = os.path.join(save_path, f'fit_metrics_{experiment_name}.csv')
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

def plot_vreturns_cdf_separated_with_gamma_fits(vreturns, experiment_name, save_path):
    """Generate CDF plots with optimized gamma fits using percentile cuts and Pareto fits for discarded tails."""
    if len(vreturns) == 0:
        return None
    
    positive_vreturns = vreturns[vreturns > 0]
    negative_vreturns = vreturns[vreturns < 0]
    
    # Initialize metrics
    pos_cuts = (0, 0)
    pos_ks_stat = 0
    vreturns_rmse_pos = 0
    neg_cuts = (0, 0)
    neg_ks_stat = 0
    vreturns_rmse_neg = 0
    
    # Initialize data for CSV
    csv_data = {}
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    data_color = '#1f77b4'
    fit_color = '#ff7f0e'
    
    if len(positive_vreturns) > 0:
        pos_trimmed, pos_cuts, pos_ks_stat, pos_left_val, pos_right_val = find_optimal_percentile_cuts(positive_vreturns)
        
        # Extract right tail data (discarded from gamma fit)
        pos_right_tail = positive_vreturns[positive_vreturns > pos_right_val]
        
        # Plot empirical CDF for truncated positive data only (ax1 - Gamma fit)
        positive_sorted = np.sort(pos_trimmed)
        positive_cdf = np.arange(1, len(positive_sorted) + 1) / len(positive_sorted)
        ax1.plot(positive_sorted, positive_cdf, color=data_color, linewidth=2, label='Empirical CDF', alpha=0.8)
        
        try:
            shape_pos, loc_pos, scale_pos = stats.gamma.fit(pos_trimmed, floc=0)
            fitted_gamma_pos = stats.gamma(a=shape_pos, loc=loc_pos, scale=scale_pos)
            metrics_pos = calculate_goodness_of_fit_metrics(pos_trimmed, fitted_gamma_pos)
            
            x_pos = np.linspace(np.min(pos_trimmed), np.max(pos_trimmed), 1000)
            gamma_cdf_pos = fitted_gamma_pos.cdf(x_pos)
            ax1.plot(x_pos, gamma_cdf_pos, color=fit_color, linewidth=2, label='Gamma CDF Fit', linestyle='--')
            
            mean_pos = np.mean(pos_trimmed)
            median_pos = np.median(pos_trimmed)
            std_pos = np.std(pos_trimmed, ddof=1)
            skew_pos = stats.skew(pos_trimmed)
            kurt_pos = stats.kurtosis(pos_trimmed, fisher=True)
            n_pos = len(pos_trimmed)
            se_mean_pos = std_pos / np.sqrt(n_pos)
            
            # Standard errors for gamma parameters
            # SE(shape) ≈ shape / sqrt(n)
            # SE(scale) ≈ scale / sqrt(n)
            # SE(rate) = SE(1/scale) ≈ rate / sqrt(n)
            se_shape_pos = shape_pos / np.sqrt(n_pos)
            se_scale_pos = scale_pos / np.sqrt(n_pos)
            rate_pos = 1.0 / scale_pos  # Rate = 1/Scale
            se_rate_pos = rate_pos / np.sqrt(n_pos)
            
            ks_stat_pos = metrics_pos.get('kolmogorov_smirnov', {}).get('statistic', np.nan)
            rmse_pos = metrics_pos.get('mse_metrics', {}).get('rmse', np.nan)
            r2_pos = metrics_pos.get('mse_metrics', {}).get('r_squared', np.nan)
            mae_pos = metrics_pos.get('mse_metrics', {}).get('mae', np.nan)
            vreturns_rmse_pos = rmse_pos if not np.isnan(rmse_pos) else 0
            
            csv_data['Positive_Gamma'] = {
                'n': n_pos,
                'Shape_alpha': f"{shape_pos:.6f}±{se_shape_pos:.6f}",
                'Scale_theta': f"{scale_pos:.6f}±{se_scale_pos:.6f}",
                'Rate_beta': f"{rate_pos:.6f}±{se_rate_pos:.6f}",
                'KS_D': f"{ks_stat_pos:.6f}" if not np.isnan(ks_stat_pos) else "",
                'RMSE': f"{rmse_pos:.6f}" if not np.isnan(rmse_pos) else "",
                'R2': f"{r2_pos:.6f}" if not np.isnan(r2_pos) else "",
                'MAE': f"{mae_pos:.6f}" if not np.isnan(mae_pos) else ""
            }
            
            stats_text_pos = f'n = {n_pos}\n'
            stats_text_pos += f'Cuts: ({pos_cuts[0]:.0f}%, {pos_cuts[1]:.0f}%)\n'
            stats_text_pos += f'Mean = {mean_pos:.4f} ± {se_mean_pos:.4f}\n'
            stats_text_pos += f'Median = {median_pos:.4f}\n'
            stats_text_pos += f'Std = {std_pos:.4f}\n'
            stats_text_pos += f'Skewness = {skew_pos:.4f}\n'
            stats_text_pos += f'Kurtosis = {kurt_pos:.4f}\n\n'
            stats_text_pos += f'Shape (α) = {shape_pos:.4f} ± {se_shape_pos:.4f}\n'
            stats_text_pos += f'Scale (θ) = {scale_pos:.4f} ± {se_scale_pos:.4f}\n'
            stats_text_pos += f'Rate (β) = {rate_pos:.4f} ± {se_rate_pos:.4f}\n\n'
            if not np.isnan(ks_stat_pos):
                stats_text_pos += f'KS(D) = {ks_stat_pos:.4f}\n'
            if not np.isnan(rmse_pos):
                stats_text_pos += f'RMSE = {rmse_pos:.4f}'
            
            ax1.text(0.65, 0.95, stats_text_pos, transform=ax1.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor='black', linewidth=1, alpha=0.9))
        except Exception as e:
            mean_pos = np.mean(pos_trimmed)
            median_pos = np.median(pos_trimmed)
            std_pos = np.std(pos_trimmed, ddof=1)
            stats_text_pos = f"n = {len(pos_trimmed)}\nCuts: ({pos_cuts[0]:.0f}%, {pos_cuts[1]:.0f}%)\nMean = {mean_pos:.4f}\nMedian = {median_pos:.4f}\nStd = {std_pos:.4f}\n\nGamma fit failed"
            ax1.text(0.65, 0.95, stats_text_pos, transform=ax1.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor='black', linewidth=1, alpha=0.9))
        
        ax1.set_xlabel('Positive VReturns')
        ax1.set_ylabel('Cumulative Probability')
        ax1.set_title('Positive VReturns - Gamma Fit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot Pareto fit for right tail (ax3)
        if len(pos_right_tail) > 0:
            pos_right_sorted = np.sort(pos_right_tail)
            pos_right_cdf = np.arange(1, len(pos_right_sorted) + 1) / len(pos_right_sorted)
            ax3.plot(pos_right_sorted, pos_right_cdf, color=data_color, linewidth=2, label='Empirical CDF (Right Tail)', alpha=0.8)
            
            try:
                pareto_pos, alpha_pos, x_m_pos, alpha_se_pos, x_m_se_pos = fit_pareto_to_data(pos_right_tail)
                
                if pareto_pos is not None:
                    x_pos_pareto = np.linspace(np.min(pos_right_tail), np.max(pos_right_tail), 1000)
                    pareto_cdf_pos = pareto_pos.cdf(x_pos_pareto)
                    ax3.plot(x_pos_pareto, pareto_cdf_pos, color=fit_color, linewidth=2, label='Pareto CDF Fit', linestyle='--')
                    
                    # Calculate KS statistic and RMSE for Pareto fit
                    ks_stat_pareto_pos, _ = stats.kstest(pos_right_tail, pareto_pos.cdf)
                    
                    # Calculate RMSE, R2, and MAE for Pareto fit
                    n_bins_pareto = min(20, max(5, len(pos_right_tail) // 10))
                    hist_counts_pareto, bin_edges_pareto = np.histogram(pos_right_tail, bins=n_bins_pareto, density=True)
                    bin_centers_pareto = (bin_edges_pareto[:-1] + bin_edges_pareto[1:]) / 2
                    theoretical_pdf_pareto = pareto_pos.pdf(bin_centers_pareto)
                    
                    mask_pareto = hist_counts_pareto > 0
                    if np.sum(mask_pareto) >= 3:
                        empirical_pdf_pareto = hist_counts_pareto[mask_pareto]
                        theoretical_pdf_pareto_masked = theoretical_pdf_pareto[mask_pareto]
                        rmse_pareto = np.sqrt(np.mean((empirical_pdf_pareto - theoretical_pdf_pareto_masked)**2))
                        mae_pareto = np.mean(np.abs(empirical_pdf_pareto - theoretical_pdf_pareto_masked))
                        ss_res = np.sum((empirical_pdf_pareto - theoretical_pdf_pareto_masked)**2)
                        ss_tot = np.sum((empirical_pdf_pareto - np.mean(empirical_pdf_pareto))**2)
                        r2_pareto = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                    else:
                        rmse_pareto = np.nan
                        mae_pareto = np.nan
                        r2_pareto = np.nan
                    
                    mean_tail_pos = np.mean(pos_right_tail)
                    median_tail_pos = np.median(pos_right_tail)
                    std_tail_pos = np.std(pos_right_tail, ddof=1)
                    n_tail_pos = len(pos_right_tail)
                    se_mean_tail_pos = std_tail_pos / np.sqrt(n_tail_pos)
                    
                    csv_data['Positive_Pareto'] = {
                        'n': n_tail_pos,
                        'Shape_alpha': f"{alpha_pos:.6f}±{alpha_se_pos:.6f}",
                        'x_m': f"{x_m_pos:.6f}±{x_m_se_pos:.6f}",
                        'KS_D': f"{ks_stat_pareto_pos:.6f}",
                        'RMSE': f"{rmse_pareto:.6f}" if not np.isnan(rmse_pareto) else "",
                        'R2': f"{r2_pareto:.6f}" if not np.isnan(r2_pareto) else "",
                        'MAE': f"{mae_pareto:.6f}" if not np.isnan(mae_pareto) else ""
                    }
                    
                    stats_text_tail_pos = f'n = {n_tail_pos}\n'
                    stats_text_tail_pos += f'Mean = {mean_tail_pos:.4f} ± {se_mean_tail_pos:.4f}\n'
                    stats_text_tail_pos += f'Median = {median_tail_pos:.4f}\n'
                    stats_text_tail_pos += f'Std = {std_tail_pos:.4f}\n\n'
                    stats_text_tail_pos += f'Shape (α) = {alpha_pos:.4f} ± {alpha_se_pos:.4f}\n'
                    stats_text_tail_pos += f'$x_m$ = {x_m_pos:.4f} ± {x_m_se_pos:.4f}\n\n'
                    stats_text_tail_pos += f'KS(D) = {ks_stat_pareto_pos:.4f}\n'
                    if not np.isnan(rmse_pareto):
                        stats_text_tail_pos += f'RMSE = {rmse_pareto:.4f}'
                    
                    ax3.text(0.65, 0.95, stats_text_tail_pos, transform=ax3.transAxes, fontsize=8,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
                            edgecolor='black', linewidth=1, alpha=0.9))
                else:
                    ax3.text(0.65, 0.95, f'n = {len(pos_right_tail)}\nPareto fit failed', 
                            transform=ax3.transAxes, fontsize=8,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
                            edgecolor='black', linewidth=1, alpha=0.9))
            except Exception as e:
                ax3.text(0.65, 0.95, f'n = {len(pos_right_tail)}\nPareto fit failed', 
                        transform=ax3.transAxes, fontsize=8,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='black', linewidth=1, alpha=0.9))
            
            ax3.set_xlabel('Positive VReturns (Right Tail)')
            ax3.set_ylabel('Cumulative Probability')
            ax3.set_title('Positive VReturns - Right Tail Pareto Fit')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    if len(negative_vreturns) > 0:
        negative_abs = np.abs(negative_vreturns)
        neg_trimmed, neg_cuts, neg_ks_stat, neg_left_val, neg_right_val = find_optimal_percentile_cuts(negative_abs)
        
        # Extract right tail data (discarded from gamma fit)
        neg_right_tail = negative_abs[negative_abs > neg_right_val]
        
        # Plot empirical CDF for truncated negative data only (ax2 - Gamma fit)
        negative_abs_sorted = np.sort(neg_trimmed)
        negative_cdf = np.arange(1, len(negative_abs_sorted) + 1) / len(negative_abs_sorted)
        ax2.plot(negative_abs_sorted, negative_cdf, color=data_color, linewidth=2, label='Empirical CDF (|negative|)', alpha=0.8)
        
        try:
            shape_neg, loc_neg, scale_neg = stats.gamma.fit(neg_trimmed, floc=0)
            fitted_gamma_neg = stats.gamma(a=shape_neg, loc=loc_neg, scale=scale_neg)
            metrics_neg = calculate_goodness_of_fit_metrics(neg_trimmed, fitted_gamma_neg)
            
            x_neg = np.linspace(np.min(neg_trimmed), np.max(neg_trimmed), 1000)
            gamma_cdf_neg = fitted_gamma_neg.cdf(x_neg)
            ax2.plot(x_neg, gamma_cdf_neg, color=fit_color, linewidth=2, label='Gamma CDF Fit', linestyle='--')
            
            mean_neg = np.mean(neg_trimmed)
            median_neg = np.median(neg_trimmed)
            std_neg = np.std(neg_trimmed, ddof=1)
            skew_neg = stats.skew(neg_trimmed)
            kurt_neg = stats.kurtosis(neg_trimmed, fisher=True)
            n_neg = len(neg_trimmed)
            se_mean_neg = std_neg / np.sqrt(n_neg)
            
            # Standard errors for gamma parameters
            # SE(shape) ≈ shape / sqrt(n)
            # SE(scale) ≈ scale / sqrt(n)
            # SE(rate) = SE(1/scale) ≈ rate / sqrt(n)
            se_shape_neg = shape_neg / np.sqrt(n_neg)
            se_scale_neg = scale_neg / np.sqrt(n_neg)
            rate_neg = 1.0 / scale_neg  # Rate = 1/Scale
            se_rate_neg = rate_neg / np.sqrt(n_neg)
            
            ks_stat_neg = metrics_neg.get('kolmogorov_smirnov', {}).get('statistic', np.nan)
            rmse_neg = metrics_neg.get('mse_metrics', {}).get('rmse', np.nan)
            r2_neg = metrics_neg.get('mse_metrics', {}).get('r_squared', np.nan)
            mae_neg = metrics_neg.get('mse_metrics', {}).get('mae', np.nan)
            vreturns_rmse_neg = rmse_neg if not np.isnan(rmse_neg) else 0
            
            csv_data['Negative_Gamma'] = {
                'n': n_neg,
                'Shape_alpha': f"{shape_neg:.6f}±{se_shape_neg:.6f}",
                'Scale_theta': f"{scale_neg:.6f}±{se_scale_neg:.6f}",
                'Rate_beta': f"{rate_neg:.6f}±{se_rate_neg:.6f}",
                'KS_D': f"{ks_stat_neg:.6f}" if not np.isnan(ks_stat_neg) else "",
                'RMSE': f"{rmse_neg:.6f}" if not np.isnan(rmse_neg) else "",
                'R2': f"{r2_neg:.6f}" if not np.isnan(r2_neg) else "",
                'MAE': f"{mae_neg:.6f}" if not np.isnan(mae_neg) else ""
            }
            
            stats_text_neg = f'n = {n_neg}\n'
            stats_text_neg += f'Cuts: ({neg_cuts[0]:.0f}%, {neg_cuts[1]:.0f}%)\n'
            stats_text_neg += f'Mean = {mean_neg:.4f} ± {se_mean_neg:.4f}\n'
            stats_text_neg += f'Median = {median_neg:.4f}\n'
            stats_text_neg += f'Std = {std_neg:.4f}\n'
            stats_text_neg += f'Skewness = {skew_neg:.4f}\n'
            stats_text_neg += f'Kurtosis = {kurt_neg:.4f}\n\n'
            stats_text_neg += f'Shape (α) = {shape_neg:.4f} ± {se_shape_neg:.4f}\n'
            stats_text_neg += f'Scale (θ) = {scale_neg:.4f} ± {se_scale_neg:.4f}\n'
            stats_text_neg += f'Rate (β) = {rate_neg:.4f} ± {se_rate_neg:.4f}\n\n'
            if not np.isnan(ks_stat_neg):
                stats_text_neg += f'KS(D) = {ks_stat_neg:.4f}\n'
            if not np.isnan(rmse_neg):
                stats_text_neg += f'RMSE = {rmse_neg:.4f}'
            
            ax2.text(0.65, 0.95, stats_text_neg, transform=ax2.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor='black', linewidth=1, alpha=0.9))
        except Exception as e:
            mean_neg = np.mean(neg_trimmed)
            median_neg = np.median(neg_trimmed)
            std_neg = np.std(neg_trimmed, ddof=1)
            stats_text_neg = f"n = {len(neg_trimmed)}\nCuts: ({neg_cuts[0]:.0f}%, {neg_cuts[1]:.0f}%)\nMean = {mean_neg:.4f}\nMedian = {median_neg:.4f}\nStd = {std_neg:.4f}\n\nGamma fit failed"
            ax2.text(0.65, 0.95, stats_text_neg, transform=ax2.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor='black', linewidth=1, alpha=0.9))
        
        ax2.set_xlabel('|Negative VReturns|')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Negative VReturns - Gamma Fit')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot Pareto fit for right tail (ax4)
        if len(neg_right_tail) > 0:
            neg_right_sorted = np.sort(neg_right_tail)
            neg_right_cdf = np.arange(1, len(neg_right_sorted) + 1) / len(neg_right_sorted)
            ax4.plot(neg_right_sorted, neg_right_cdf, color=data_color, linewidth=2, label='Empirical CDF (Right Tail)', alpha=0.8)
            
            try:
                pareto_neg, alpha_neg, x_m_neg, alpha_se_neg, x_m_se_neg = fit_pareto_to_data(neg_right_tail)
                
                if pareto_neg is not None:
                    x_neg_pareto = np.linspace(np.min(neg_right_tail), np.max(neg_right_tail), 1000)
                    pareto_cdf_neg = pareto_neg.cdf(x_neg_pareto)
                    ax4.plot(x_neg_pareto, pareto_cdf_neg, color=fit_color, linewidth=2, label='Pareto CDF Fit', linestyle='--')
                    
                    # Calculate KS statistic and RMSE for Pareto fit
                    ks_stat_pareto_neg, _ = stats.kstest(neg_right_tail, pareto_neg.cdf)
                    
                    # Calculate RMSE, R2, and MAE for Pareto fit
                    n_bins_pareto = min(20, max(5, len(neg_right_tail) // 10))
                    hist_counts_pareto, bin_edges_pareto = np.histogram(neg_right_tail, bins=n_bins_pareto, density=True)
                    bin_centers_pareto = (bin_edges_pareto[:-1] + bin_edges_pareto[1:]) / 2
                    theoretical_pdf_pareto = pareto_neg.pdf(bin_centers_pareto)
                    
                    mask_pareto = hist_counts_pareto > 0
                    if np.sum(mask_pareto) >= 3:
                        empirical_pdf_pareto = hist_counts_pareto[mask_pareto]
                        theoretical_pdf_pareto_masked = theoretical_pdf_pareto[mask_pareto]
                        rmse_pareto_neg = np.sqrt(np.mean((empirical_pdf_pareto - theoretical_pdf_pareto_masked)**2))
                        mae_pareto_neg = np.mean(np.abs(empirical_pdf_pareto - theoretical_pdf_pareto_masked))
                        ss_res = np.sum((empirical_pdf_pareto - theoretical_pdf_pareto_masked)**2)
                        ss_tot = np.sum((empirical_pdf_pareto - np.mean(empirical_pdf_pareto))**2)
                        r2_pareto_neg = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                    else:
                        rmse_pareto_neg = np.nan
                        mae_pareto_neg = np.nan
                        r2_pareto_neg = np.nan
                    
                    mean_tail_neg = np.mean(neg_right_tail)
                    median_tail_neg = np.median(neg_right_tail)
                    std_tail_neg = np.std(neg_right_tail, ddof=1)
                    n_tail_neg = len(neg_right_tail)
                    se_mean_tail_neg = std_tail_neg / np.sqrt(n_tail_neg)
                    
                    csv_data['Negative_Pareto'] = {
                        'n': n_tail_neg,
                        'Shape_alpha': f"{alpha_neg:.6f}±{alpha_se_neg:.6f}",
                        'x_m': f"{x_m_neg:.6f}±{x_m_se_neg:.6f}",
                        'KS_D': f"{ks_stat_pareto_neg:.6f}",
                        'RMSE': f"{rmse_pareto_neg:.6f}" if not np.isnan(rmse_pareto_neg) else "",
                        'R2': f"{r2_pareto_neg:.6f}" if not np.isnan(r2_pareto_neg) else "",
                        'MAE': f"{mae_pareto_neg:.6f}" if not np.isnan(mae_pareto_neg) else ""
                    }
                    
                    stats_text_tail_neg = f'n = {n_tail_neg}\n'
                    stats_text_tail_neg += f'Mean = {mean_tail_neg:.4f} ± {se_mean_tail_neg:.4f}\n'
                    stats_text_tail_neg += f'Median = {median_tail_neg:.4f}\n'
                    stats_text_tail_neg += f'Std = {std_tail_neg:.4f}\n\n'
                    stats_text_tail_neg += f'Shape (α) = {alpha_neg:.4f} ± {alpha_se_neg:.4f}\n'
                    stats_text_tail_neg += f'$x_m$ = {x_m_neg:.4f} ± {x_m_se_neg:.4f}\n\n'
                    stats_text_tail_neg += f'KS(D) = {ks_stat_pareto_neg:.4f}\n'
                    if not np.isnan(rmse_pareto):
                        stats_text_tail_neg += f'RMSE = {rmse_pareto:.4f}'
                    
                    ax4.text(0.65, 0.95, stats_text_tail_neg, transform=ax4.transAxes, fontsize=8,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
                            edgecolor='black', linewidth=1, alpha=0.9))
                else:
                    ax4.text(0.65, 0.95, f'n = {len(neg_right_tail)}\nPareto fit failed', 
                            transform=ax4.transAxes, fontsize=8,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
                            edgecolor='black', linewidth=1, alpha=0.9))
            except Exception as e:
                ax4.text(0.65, 0.95, f'n = {len(neg_right_tail)}\nPareto fit failed', 
                        transform=ax4.transAxes, fontsize=8,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='black', linewidth=1, alpha=0.9))
            
            ax4.set_xlabel('|Negative VReturns| (Right Tail)')
            ax4.set_ylabel('Cumulative Probability')
            ax4.set_title('Negative VReturns - Right Tail Pareto Fit')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'vreturns_cdf_separated_with_gamma_fits_{experiment_name}.png'
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    save_fit_metrics_csv(csv_data, save_path, experiment_name)
    
    return filename

if __name__ == "__main__":
    import sys
    
    # Check if custom folder was provided as command line argument
    if len(sys.argv) > 1:
        experiments_root = sys.argv[1]
    else:
        experiments_root = 'experiments_multi'
    
    main(experiments_root=experiments_root)


