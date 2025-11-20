import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import pandas as pd
import pickle
import warnings
import yfinance as yf
from scipy.stats import geom
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='invalid value encountered in log')
warnings.filterwarnings('ignore', message='invalid value encountered in sqrt')
warnings.filterwarnings('ignore', message='divide by zero encountered in divide')
warnings.filterwarnings('ignore', message='Optimal parameters not found')

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

def download_financial_data():
    indices = {'DJIA': '^DJI', 'DAX': '^GDAXI', 'IPC': '^MXX', 'Nikkei': '^N225'}
    date_ranges = {
        'DJIA': ('1991-08-11', '2023-12-31'), 'DAX': ('1991-08-11', '2023-12-30'),
        'IPC': ('1991-08-11', '2023-12-31'), 'Nikkei': ('1991-10-11', '2023-12-30')
    }
    
    # Load cached data
    data, all_prices_raw = {}, {}
    for index_name in indices.keys():
        close_fn = os.path.join('financial_data', f"{index_name}_close.pkl")
        full_fn = os.path.join('financial_data', f"{index_name}_full.pkl")
        if os.path.exists(close_fn) and os.path.exists(full_fn):
            try:
                with open(close_fn, 'rb') as f:
                    data[index_name] = pickle.load(f)
                with open(full_fn, 'rb') as f:
                    all_prices_raw[index_name] = pickle.load(f)
            except Exception:
                pass
    
    missing_indices = [idx for idx in indices.keys() if idx not in data or idx not in all_prices_raw]
    if not missing_indices:
        pass
    else:
        for index in missing_indices:
            ticker, (start_date, end_date) = indices[index], date_ranges[index]
            try:
                df_yf = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df_yf.empty:
                    data[index] = df_yf['Close'].dropna()
                    all_prices_raw[index] = df_yf
            except Exception:
                pass
        
        # Save downloaded data
        if not os.path.exists('financial_data'):
            os.makedirs('financial_data')
        for index_name, price_series in data.items():
            try:
                with open(os.path.join('financial_data', f"{index_name}_close.pkl"), 'wb') as f:
                    pickle.dump(price_series, f)
            except Exception:
                pass
        for index_name, price_df in all_prices_raw.items():
            try:
                with open(os.path.join('financial_data', f"{index_name}_full.pkl"), 'wb') as f:
                    pickle.dump(price_df, f)
            except Exception:
                pass
    
    # Synchronize dates
    if data:
        start_dates = [price_series.index.min() for price_series in data.values() if not price_series.empty]
        end_dates = [price_series.index.max() for price_series in data.values() if not price_series.empty]
        if start_dates and end_dates:
            common_start_date = max(start_dates)
            common_end_date = min(end_dates)
            synchronized_data = {}
            synchronized_all_prices_raw = {}
            for index_name, price_series in data.items():
                mask = (price_series.index >= common_start_date) & (price_series.index <= common_end_date)
                synchronized_data[index_name] = price_series[mask]
                if index_name in all_prices_raw:
                    price_df = all_prices_raw[index_name]
                    mask_df = (price_df.index >= common_start_date) & (price_df.index <= common_end_date)
                    synchronized_all_prices_raw[index_name] = price_df[mask_df]
            data = synchronized_data
            all_prices_raw = synchronized_all_prices_raw
    
    return data, all_prices_raw

# --- Trend Analysis Functions ---

def identify_trends(prices):
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


def precompute_trend_data(all_prices_data):
    all_trends_by_index = {}
    all_trend_sequences_by_index = {}
    for index_name, prices_series in all_prices_data.items():
        prices_series_cleaned = prices_series.dropna()
        if prices_series_cleaned.empty:
            continue
        prices_values_cleaned = prices_series_cleaned.values
        prices_cleaned = prices_values_cleaned[~np.isnan(prices_values_cleaned)]
        if len(prices_cleaned) < 2:
            continue
        trends = identify_trends(prices_cleaned)
        if not trends:
            continue
        enriched_trends = enrich_trend_data(prices_cleaned, trends)
        if not enriched_trends:
            continue
        df_trends_current_index = pd.DataFrame(enriched_trends)
        df_trends_current_index['index_name'] = index_name
        all_trends_by_index[index_name] = df_trends_current_index
    return all_trends_by_index, all_trend_sequences_by_index

# --- Auxiliary Functions ---

def calculate_rmse(observed_freq, expected_freq):
    if len(observed_freq) == 0 or len(expected_freq) == 0:
        return np.nan
    valid_indices = ~(np.isnan(observed_freq) | np.isinf(observed_freq) | np.isnan(expected_freq) | np.isinf(expected_freq))
    observed_freq = observed_freq[valid_indices]
    expected_freq = expected_freq[valid_indices]
    if len(observed_freq) == 0:
        return np.nan
    return np.sqrt(np.mean((observed_freq - expected_freq)**2))

# --- Plotting Functions ---

def _add_statistical_summary_to_plot(ax, data, geom_fit_params=None, geom_fit_rmse=None, geom_chi2=None, geom_chi2_p=None):
    if data is None or len(data) < 2:
        return
    
    mean_val = np.mean(data)
    median_val = np.median(data)
    n = len(data)
    std = np.std(data, ddof=1) if n > 1 else 0
    mean_error = std / np.sqrt(n) if n > 0 else np.nan
    median_error = 1.253 * std / np.sqrt(n) if n > 0 else np.nan

    ax.axvline(mean_val, color='green', linestyle='--', linewidth=1.5, label=f"Mean: {mean_val:.2f}±{mean_error:.2f}")
    ax.axvline(median_val, color='red', linestyle=':', linewidth=1.5, label=f"Median: {median_val:.2f}")
    
    if geom_fit_params is not None and geom_chi2 is not None and geom_chi2_p is not None:
        goodness_text = f"Geometric Fit:\nχ² = {geom_chi2:.3f}\np-value = {geom_chi2_p:.3f}\nRMSE = {geom_fit_rmse:.3f}"
        ax.text(0.78, 0.65, goodness_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, loc='best', fontsize=8)


def exponential_model(x, a, b):
    """Exponential model: y = a * exp(-b * x)"""
    return a * np.exp(-b * x)

def fit_exponential_to_duration(data):
    """
    Fits an exponential model y = a * exp(-b * x) to duration data.
    Returns parameters, errors, goodness-of-fit metrics including R².
    """
    try:
        # Create histogram data
        bins = np.arange(data.min(), data.max() + 2) - 0.5
        counts, bin_edges = np.histogram(data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Remove zero counts for fitting
        non_zero_mask = counts > 0
        x_data = bin_centers[non_zero_mask]
        y_data = counts[non_zero_mask]
        
        if len(x_data) < 3:  # Need at least 3 points for exponential fit
            return None, None, np.nan, np.nan, np.nan, np.nan, None
        
        # Initial parameter guess
        p0 = [y_data[0], 0.1]  # a = first y value, b = small positive value
        
        # Fit exponential model
        popt, pcov = curve_fit(exponential_model, x_data, y_data, p0=p0, maxfev=5000)
        a, b = popt
        
        # Calculate parameter errors
        param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan, np.nan]
        
        # Calculate fitted values
        y_fitted = exponential_model(x_data, a, b)
        
        # Calculate R²
        r2 = r2_score(y_data, y_fitted)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_data - y_fitted) ** 2))
        
        # Calculate chi-square test
        chi2_stat = np.sum((y_data - y_fitted) ** 2 / y_fitted)
        dof = len(y_data) - 2  # degrees of freedom = n_points - n_parameters
        chi2_p_value = 1 - stats.chi2.cdf(chi2_stat, dof) if dof > 0 else np.nan
        
        return (a, b), param_errors, chi2_stat, chi2_p_value, rmse, r2, {'x': x_data, 'y': y_data, 'y_fitted': y_fitted}
        
    except Exception as e:
        print(f"    Error in exponential fit: {e}")
        return None, None, np.nan, np.nan, np.nan, np.nan, None

def fit_geometric_to_duration(data):
    """
    Fits a geometric distribution to discrete duration data and computes goodness-of-fit metrics.
    Returns parameters, parameter errors, chi-squared statistics, and RMSE.
    """
    if len(data) < 2 or not np.all(data > 0):
        print(f"  Warning: Insufficient or invalid data for geometric fit (len={len(data)}, all_positive={np.all(data > 0)})")
        return None, None, None, None, None, None
    try:
        # Ensure data is integer
        data = np.round(data).astype(int)
        
        # Estimate geometric distribution parameter using method of moments
        # For geometric distribution starting at 1: E[X] = 1/p, so p = 1/mean(X)
        sample_mean = np.mean(data)
        if sample_mean <= 0:
            return None, None, None, None, None, None
        
        p_est = 1.0 / sample_mean
        # Ensure p is in valid range (0, 1]
        p_est = min(max(p_est, 1e-10), 1.0)
        
        # Geometric distribution in scipy.stats uses parameterization where X starts at 1
        # geom.pmf(k, p) gives P(X = k) for k = 1, 2, 3, ...
        params = (0, p_est)  # (loc, p) where loc=0 means starts at 1
        
        # Compute chi-squared test for goodness of fit
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
        
        # Relaxed chi-squared requirement: use bins with expected frequency >= 1
        mask = expected_freq >= 1
        if np.sum(mask) >= 2:  # Require at least 2 valid bins for chi-squared test
            observed_freq_filtered = observed_freq[mask]
            expected_freq_filtered = expected_freq[mask]
            
            # Normalize expected frequencies to match observed sum (required by chisquare)
            observed_sum = np.sum(observed_freq_filtered)
            expected_sum = np.sum(expected_freq_filtered)
            if expected_sum > 0:
                expected_freq_filtered = expected_freq_filtered * (observed_sum / expected_sum)
            
            from scipy.stats import chisquare
            chi2_stat, chi2_p_value = chisquare(observed_freq_filtered, expected_freq_filtered)
        else:
            chi2_stat, chi2_p_value = np.nan, np.nan
        
        # Compute RMSE using all data points
        rmse_val = calculate_rmse(observed_freq, expected_freq)
        
        # Bootstrap for parameter errors
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
        print(f"  Error in fit_geometric_to_duration: {e}")
        return None, None, None, None, None, None

def plot_trend_duration_distribution(all_trends_by_index, save_path='plots_trend_acceleration', index_name="", plot_number=1):
    """
    Generates a single plot with two subplots showing the duration distribution of
    ascending and descending trends with geometric fits.
    """
    output_folder_idx = os.path.join(save_path, index_name)
    if not os.path.exists(output_folder_idx):
        os.makedirs(output_folder_idx)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_up = axes[0]
    up_trends = all_trends_by_index.get(index_name, pd.DataFrame()).query('direction == 1')
    up_durations = up_trends['duration'].values
    ax_down = axes[1]
    down_trends = all_trends_by_index.get(index_name, pd.DataFrame()).query('direction == -1')
    down_durations = down_trends['duration'].values
    if len(up_durations) > 0:
        # Create histogram for ascending trends (discrete bins)
        bins_up = np.arange(up_durations.min(), up_durations.max() + 2) - 0.5
        counts_up, bins_up, _ = ax_up.hist(up_durations, bins=bins_up, density=False, alpha=0.7, color=ASCENDING_COLOR, edgecolor='black', label='Uptrends')
        # Fit geometric distribution to ascending durations
        geom_params_up, geom_errors_up, chi2_up, chi2_p_up, rmse_up, hist_data_up = fit_geometric_to_duration(up_durations)
        if geom_params_up is not None:
            loc, p = geom_params_up
            p_error_up = geom_errors_up[1] if geom_errors_up is not None and len(geom_errors_up) > 1 else np.nan
            label_up = f'Geometric Fit (p={p:.3f}±{p_error_up:.3f})' if pd.notna(p_error_up) else f'Geometric Fit (p={p:.3f})'
            discrete_x = np.arange(1, int(up_durations.max()) + 1)
            pmf_values = geom.pmf(discrete_x, p) * len(up_durations)  # Scale to frequency
            ax_up.bar(discrete_x, pmf_values, alpha=0.6, color='red', width=0.4, label=label_up)
            
        _add_statistical_summary_to_plot(ax_up, up_durations, geom_fit_params=geom_params_up, geom_fit_rmse=rmse_up, geom_chi2=chi2_up, geom_chi2_p=chi2_p_up)
    else:
        ax_up.text(0.5, 0.5, 'No ascending trends found', ha='center', va='center', transform=ax_up.transAxes)
    ax_up.set_title('(a) Uptrends')
    ax_up.set_xlabel('Trend Duration (Days)')
    ax_up.set_ylabel('Frequency (log)')
    ax_up.set_yscale('log')
    ax_up.legend()
    ax_up.grid(True, which="both", ls="--", alpha=0.3)
    if len(down_durations) > 0:
        bins_down = np.arange(down_durations.min(), down_durations.max() + 2) - 0.5
        counts_down, bins_down, _ = ax_down.hist(down_durations, bins=bins_down, density=False, alpha=0.7, color=DESCENDING_COLOR, edgecolor='black', label='Downtrends')
        geom_params_down, geom_errors_down, chi2_down, chi2_p_down, rmse_down, hist_data_down = fit_geometric_to_duration(down_durations)
        if geom_params_down is not None:
            loc, p = geom_params_down
            p_error_down = geom_errors_down[1] if geom_errors_down is not None and len(geom_errors_down) > 1 else np.nan
            label_down = f'Geometric Fit (p={p:.3f}±{p_error_down:.3f})' if pd.notna(p_error_down) else f'Geometric Fit (p={p:.3f})'
            discrete_x = np.arange(1, int(down_durations.max()) + 1)
            pmf_values = geom.pmf(discrete_x, p) * len(down_durations)
            ax_down.bar(discrete_x, pmf_values, alpha=0.6, color='red', width=0.4, label=label_down)
            
        _add_statistical_summary_to_plot(ax_down, down_durations, geom_fit_params=geom_params_down, geom_fit_rmse=rmse_down, geom_chi2=chi2_down, geom_chi2_p=chi2_p_down)
    else:
        ax_down.text(0.5, 0.5, 'No descending trends found', ha='center', va='center', transform=ax_down.transAxes)
    ax_down.set_title('(b) Downtrends')
    ax_down.set_xlabel('Trend Duration (Days)')
    ax_down.set_ylabel('Frequency (log)')
    ax_down.set_yscale('log')
    ax_down.legend()
    ax_down.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f'plot_{plot_number:02d}_trend_duration_distribution_{index_name}.png'
    plt.savefig(f'{output_folder_idx}/{plot_filename}', dpi=300, bbox_inches='tight')
    plt.close()

def plot_trend_duration_distribution_exponential_fit(all_trends_by_index, save_path='plots_trend_acceleration', index_name="", plot_number=2):
    """
    Generates a single plot with two subplots showing the duration distribution of
    ascending and descending trends as scatter points with exponential fits.
    """
    output_folder_idx = os.path.join(save_path, index_name)
    if not os.path.exists(output_folder_idx):
        os.makedirs(output_folder_idx)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_up = axes[0]
    up_trends = all_trends_by_index.get(index_name, pd.DataFrame()).query('direction == 1')
    up_durations = up_trends['duration'].values
    
    ax_down = axes[1]
    down_trends = all_trends_by_index.get(index_name, pd.DataFrame()).query('direction == -1')
    down_durations = down_trends['duration'].values
    
    if len(up_durations) > 0:
        # Create histogram data for scatter points
        bins_up = np.arange(up_durations.min(), up_durations.max() + 2) - 0.5
        counts_up, bin_edges_up = np.histogram(up_durations, bins=bins_up)
        bin_centers_up = (bin_edges_up[:-1] + bin_edges_up[1:]) / 2
        
        # Plot as scatter points
        ax_up.scatter(bin_centers_up, counts_up, color=ASCENDING_COLOR, alpha=0.7, s=50, label='Uptrends', edgecolors='black')
        
        # Fit exponential distribution to ascending durations
        exp_params_up, exp_errors_up, chi2_up, chi2_p_up, rmse_up, r2_up, fit_data_up = fit_exponential_to_duration(up_durations)
        
        if exp_params_up is not None:
            a, b = exp_params_up
            a_error_up = exp_errors_up[0] if exp_errors_up is not None and len(exp_errors_up) > 0 else np.nan
            b_error_up = exp_errors_up[1] if exp_errors_up is not None and len(exp_errors_up) > 1 else np.nan
            
            # Plot exponential fit
            x_fit = np.linspace(1, int(up_durations.max()), 100)
            y_fit = exponential_model(x_fit, a, b)
            
            label_up = f'Exponential Fit (a={a:.2f}±{a_error_up:.2f}, b={b:.3f}±{b_error_up:.3f})' if pd.notna(a_error_up) and pd.notna(b_error_up) else f'Exponential Fit (a={a:.2f}, b={b:.3f})'
            ax_up.plot(x_fit, y_fit, 'r-', linewidth=2, label=label_up)
            
        else:
            print(f"  Warning: Exponential fit failed for ascending trends in {index_name}")
    else:
        ax_up.text(0.5, 0.5, 'No ascending trends found', ha='center', va='center', transform=ax_up.transAxes)
    
    ax_up.set_title('(a) Uptrends')
    ax_up.set_xlabel('Trend Duration (Days)')
    ax_up.set_ylabel('Frequency (log)')
    ax_up.set_yscale('log')
    ax_up.legend()
    ax_up.grid(True, which="both", ls="--", alpha=0.3)
    
    if len(down_durations) > 0:
        # Create histogram data for scatter points
        bins_down = np.arange(down_durations.min(), down_durations.max() + 2) - 0.5
        counts_down, bin_edges_down = np.histogram(down_durations, bins=bins_down)
        bin_centers_down = (bin_edges_down[:-1] + bin_edges_down[1:]) / 2
        
        # Plot as scatter points
        ax_down.scatter(bin_centers_down, counts_down, color=DESCENDING_COLOR, alpha=0.7, s=50, label='Downtrends', edgecolors='black')
        
        # Fit exponential distribution to descending durations
        exp_params_down, exp_errors_down, chi2_down, chi2_p_down, rmse_down, r2_down, fit_data_down = fit_exponential_to_duration(down_durations)
        
        if exp_params_down is not None:
            a, b = exp_params_down
            a_error_down = exp_errors_down[0] if exp_errors_down is not None and len(exp_errors_down) > 0 else np.nan
            b_error_down = exp_errors_down[1] if exp_errors_down is not None and len(exp_errors_down) > 1 else np.nan
            
            # Plot exponential fit
            x_fit = np.linspace(1, int(down_durations.max()), 100)
            y_fit = exponential_model(x_fit, a, b)
            
            label_down = f'Exponential Fit (a={a:.2f}±{a_error_down:.2f}, b={b:.3f}±{b_error_down:.3f})' if pd.notna(a_error_down) and pd.notna(b_error_down) else f'Exponential Fit (a={a:.2f}, b={b:.3f})'
            ax_down.plot(x_fit, y_fit, 'r-', linewidth=2, label=label_down)
            
    else:
        ax_down.text(0.5, 0.5, 'No descending trends found', ha='center', va='center', transform=ax_down.transAxes)
    
    ax_down.set_title('(b) Downtrends')
    ax_down.set_xlabel('Trend Duration (Days)')
    ax_down.set_ylabel('Frequency (log)')
    ax_down.set_yscale('log')
    ax_down.legend()
    ax_down.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f'plot_{plot_number:02d}_trend_duration_distribution_exponential_fit_{index_name}.png'
    plt.savefig(f'{output_folder_idx}/{plot_filename}', dpi=300, bbox_inches='tight')
    plt.close()


def generate_plots(all_trends_by_index, all_trend_sequences_by_index, save_path='plots_trend_acceleration'):
    for index_name, df_trends_current_index in all_trends_by_index.items():
        output_folder_for_index = os.path.join(save_path, index_name)
        if not os.path.exists(output_folder_for_index):
            os.makedirs(output_folder_for_index)
        plot_trend_duration_distribution(all_trends_by_index, save_path, index_name, plot_number=1)
        plot_trend_duration_distribution_exponential_fit(all_trends_by_index, save_path, index_name, plot_number=2)

def main():
    all_prices_data, _ = download_financial_data()
    if not all_prices_data:
        return
    all_trends_by_index, all_trend_sequences_by_index = precompute_trend_data(all_prices_data)
    if not all_trends_by_index and not all_trend_sequences_by_index:
        return
    save_directory = 'plots_trend_acceleration'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    generate_plots(all_trends_by_index, all_trend_sequences_by_index, save_directory)

if __name__ == "__main__":
    main()
