import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.stats import geom, chisquare, gaussian_kde
from statsmodels.graphics.tsaplots import plot_acf
import warnings
from scipy.stats import gamma, pareto, kstest
from scipy.special import polygamma
from scipy.integrate import trapezoid
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Configuration of warnings and style
warnings.filterwarnings('ignore')

# Matplotlib configuration
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

def load_garch_prices(csv_filename='garch_simulated_prices.csv'):
    """Load GARCH price series from CSV"""
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"File {csv_filename} not found")
    return pd.read_csv(csv_filename)

def identify_trends(prices, min_trend_length=1):
    """Identify uptrends and downtrends"""
    if len(prices) < 2:
        return []
    
    trends = []
    i = 0
    
    while i < len(prices) - 1:
        # Search for uptrend
        j = i + 1
        while j < len(prices) and prices[j] >= prices[j-1]:
            j += 1
        
        # If we found a valid uptrend
        if j - 1 > i:
            duration = j - 1 - i
            if duration >= min_trend_length:
                trends.append({
                    'type': 'uptrend',
                    'start': i,
                    'end': j - 1,
                    'duration': duration,
                    'start_price': prices[i],
                    'end_price': prices[j - 1]
                })
            i = j - 1
            continue
        
        # Search for downtrend
        k = i + 1
        while k < len(prices) and prices[k] <= prices[k-1]:
            k += 1
        
        # If we found a valid downtrend
        if k - 1 > i:
            duration = k - 1 - i
            if duration >= min_trend_length:
                trends.append({
                    'type': 'downtrend',
                    'start': i,
                    'end': k - 1,
                    'duration': duration,
                    'start_price': prices[i],
                    'end_price': prices[k - 1]
                })
            i = k - 1
            continue
        
        # If no trend, advance
        i += 1
    
    return trends

def calculate_tvreturns(prices, trends):
    """Calculate TVReturns = ln(P_end / P_start)"""
    tvreturns = []
    
    for trend in trends:
        start_price = trend['start_price']
        end_price = trend['end_price']
        
        if start_price > 0 and end_price > 0:
            tvreturn = np.log(end_price / start_price)
            tvreturns.append(tvreturn)
    
    return np.array(tvreturns)

def calculate_log_returns(price_series):
    """Calculate daily logarithmic returns"""
    return np.diff(np.log(price_series))

def fit_geometric_to_duration(data):
    """Fit geometric distribution to duration data"""
    if len(data) < 2 or not np.all(data > 0):
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
        min_val = unique_values.min()
        max_val = unique_values.max()
        
        observed_freq = []
        expected_freq = []
        values_range = np.arange(min_val, max_val + 1)
        
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
        return None, None, None, None, None, None

def calculate_rmse(observed_freq, expected_freq):
    """Calculate RMSE"""
    return np.sqrt(np.mean((observed_freq - expected_freq)**2))

def calculate_kl_divergence(positive_data, negative_data_abs, method='kde', bins=50):
    """Calculate KL divergence D(P||Q)"""
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
            
            p_pos = p_pos / trapezoid(p_pos, x_eval)
            p_neg = p_neg / trapezoid(p_neg, x_eval)
            
            # Calculate KL divergence using trapezoidal integration
            # D(P||Q) = ∫ P(x) * log(P(x)/Q(x)) dx
            integrand = p_pos * np.log(p_pos / p_neg)
            kl_divergence = trapezoid(integrand, x_eval)
            
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

def fit_gamma_distribution(data):
    """Fit gamma distribution"""
    if len(data) < 10:
        return None, None, None, None, None, None, None
    
    try:
        shape, loc, scale = gamma.fit(data, floc=0)
        n = len(data)
        I_aa = polygamma(1, shape)
        I_tt = shape / (scale**2)
        I_at = 1 / scale
        
        fisher_matrix = np.array([[I_aa, I_at], [I_at, I_tt]])
        try:
            cov_matrix = np.linalg.inv(fisher_matrix)
            shape_se = np.sqrt(cov_matrix[0, 0] / n)
            scale_se = np.sqrt(cov_matrix[1, 1] / n)
        except np.linalg.LinAlgError:
            shape_se = np.sqrt(1 / (n * I_aa))
            scale_se = np.sqrt(1 / (n * I_tt))
        ks_stat, ks_p_value = kstest(data, lambda x: gamma.cdf(x, shape, loc=loc, scale=scale))
        hist_counts, bin_edges = np.histogram(data, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        theoretical_hist = gamma.pdf(bin_centers, shape, loc=loc, scale=scale)
        mse = mean_squared_error(hist_counts, theoretical_hist)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(hist_counts, theoretical_hist)
        
        # Calculate R²
        ss_res = np.sum((hist_counts - theoretical_hist) ** 2)
        ss_tot = np.sum((hist_counts - np.mean(hist_counts)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return (shape, loc, scale, shape_se, scale_se), ks_p_value, mse, rmse, mae, r2
        
    except Exception as e:
        return None, None, None, None, None, None, None

def fit_pareto_distribution(data):
    """Fit Pareto distribution"""
    if len(data) < 10:
        return None, None, None, None, None, None, None
    
    try:
        shape, loc, scale = pareto.fit(data, floc=0)
        n = len(data)
        I_aa = 1 / (shape**2)
        I_xx = shape / (scale**2)
        I_ax = 1 / (shape * scale)
        
        fisher_matrix = np.array([[I_aa, I_ax], [I_ax, I_xx]])
        try:
            cov_matrix = np.linalg.inv(fisher_matrix)
            shape_se = np.sqrt(cov_matrix[0, 0] / n)
            scale_se = np.sqrt(cov_matrix[1, 1] / n)
        except np.linalg.LinAlgError:
            shape_se = np.sqrt(1 / (n * I_aa))
            scale_se = np.sqrt(1 / (n * I_xx))
        ks_stat, ks_p_value = kstest(data, lambda x: pareto.cdf(x, shape, loc=loc, scale=scale))
        hist_counts, bin_edges = np.histogram(data, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        theoretical_hist = pareto.pdf(bin_centers, shape, loc=loc, scale=scale)
        mse = mean_squared_error(hist_counts, theoretical_hist)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(hist_counts, theoretical_hist)
        
        # Calculate R²
        ss_res = np.sum((hist_counts - theoretical_hist) ** 2)
        ss_tot = np.sum((hist_counts - np.mean(hist_counts)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return (shape, loc, scale, shape_se, scale_se), ks_p_value, mse, rmse, mae, r2
        
    except Exception as e:
        return None, None, None, None, None, None, None

def plot_trend_duration_distribution(trends_df, save_path='TVReturns_Analysis_Plots/GARCH_plots'):
    """Plot trend duration distribution with geometric fit"""
    os.makedirs(save_path, exist_ok=True)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_up = axes[0]
    ax_down = axes[1]
    
    # Filter trends by type
    up_trends = trends_df[trends_df['type'] == 'uptrend']
    down_trends = trends_df[trends_df['type'] == 'downtrend']
    
    up_durations = up_trends['duration'].values
    down_durations = down_trends['duration'].values
    
    # Exact colors from trend_acceleration_plots.py
    ASCENDING_COLOR = '#2ca02c'  # Green
    DESCENDING_COLOR = '#d62728' # Red
    
    # Create histogram for uptrends
    if len(up_durations) > 0:
        # Create histogram for uptrends
        bins_up = np.arange(up_durations.min(), up_durations.max() + 2) - 0.5
        counts_up, bins_up, _ = ax_up.hist(up_durations, bins=bins_up, density=False, alpha=0.7, 
                                          color=ASCENDING_COLOR, edgecolor='black', label='Uptrends')
        
        # Fit geometric distribution
        geom_params_up, geom_errors_up, chi2_up, chi2_p_up, rmse_up, hist_data_up = fit_geometric_to_duration(up_durations)
        
        if geom_params_up is not None:
            loc, p = geom_params_up
            p_error_up = geom_errors_up[1] if geom_errors_up is not None and len(geom_errors_up) > 1 else np.nan
            label_up = f'Geometric Fit (p={p:.3f}±{p_error_up:.3f})' if pd.notna(p_error_up) else f'Geometric Fit (p={p:.3f})'
            
            discrete_x = np.arange(1, int(up_durations.max()) + 1)
            pmf_values = geom.pmf(discrete_x, p) * len(up_durations)
            ax_up.bar(discrete_x, pmf_values, alpha=0.6, color='red', width=0.4, label=label_up)
            
            # Add mean and median lines as in trend_acceleration_plots.py
            mean_val = np.mean(up_durations)
            median_val = np.median(up_durations)
            mean_error = np.std(up_durations)/np.sqrt(len(up_durations))
            
            ax_up.axvline(mean_val, color='green', linestyle='--', linewidth=1.5, label=f"Mean: {mean_val:.2f}±{mean_error:.2f}")
            ax_up.axvline(median_val, color='red', linestyle=':', linewidth=1.5, label=f"Median: {median_val:.2f}")
            
            # Goodness of fit text as in trend_acceleration_plots.py
            goodness_text = f"Geometric Fit:\nχ² = {chi2_up:.3f}\np-value = {chi2_p_up:.3f}\nRMSE = {rmse_up:.3f}"
            ax_up.text(0.78, 0.65, goodness_text, transform=ax_up.transAxes, fontsize=9,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax_up.text(0.5, 0.5, 'No uptrends found', ha='center', va='center', transform=ax_up.transAxes)
    
    ax_up.set_title('(a) Uptrends', fontsize=14)
    ax_up.set_xlabel('Trend Duration (Days)', fontsize=12)
    ax_up.set_ylabel('Frequency (log)', fontsize=12)
    ax_up.set_yscale('log')
    ax_up.grid(True, which="both", ls="--", alpha=0.3)
    ax_up.legend(fontsize=8)
    
    # Plot downtrends
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
            
            # Add mean and median lines as in trend_acceleration_plots.py
            mean_val = np.mean(down_durations)
            median_val = np.median(down_durations)
            mean_error = np.std(down_durations)/np.sqrt(len(down_durations))
            
            ax_down.axvline(mean_val, color='green', linestyle='--', linewidth=1.5, label=f"Mean: {mean_val:.2f}±{mean_error:.2f}")
            ax_down.axvline(median_val, color='red', linestyle=':', linewidth=1.5, label=f"Median: {median_val:.2f}")
            
            # Goodness of fit text as in trend_acceleration_plots.py
            goodness_text = f"Geometric Fit:\nχ² = {chi2_down:.3f}\np-value = {chi2_p_down:.3f}\nRMSE = {rmse_down:.3f}"
            ax_down.text(0.78, 0.65, goodness_text, transform=ax_down.transAxes, fontsize=9,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax_down.text(0.5, 0.5, 'No downtrends found', ha='center', va='center', transform=ax_down.transAxes)
    
    ax_down.set_title('(b) Downtrends', fontsize=14)
    ax_down.set_xlabel('Trend Duration (Days)', fontsize=12)
    ax_down.set_ylabel('Frequency (log)', fontsize=12)
    ax_down.set_yscale('log')
    ax_down.grid(True, which="both", ls="--", alpha=0.3)
    ax_down.legend(fontsize=8)
    
    # Finalize plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(save_path, 'plot_01_trend_duration_distribution_GARCH.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename

def plot_tvreturns_histogram_linear(tvreturns, save_path='TVReturns_Analysis_Plots/GARCH_plots'):
    """Plot linear TVReturns histogram"""
    os.makedirs(save_path, exist_ok=True)
    
    if len(tvreturns) == 0:
        return None
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # DJIA blue color as in reference image
    djia_color = '#5B9BD5'
    
    # Create histogram
    ax.hist(tvreturns, bins=50, density=False, alpha=0.8, 
           color=djia_color, edgecolor='black', linewidth=0.3, 
           label='Data')
    
    # Calculate statistical metrics
    mean_val = np.mean(tvreturns)
    median_val = np.median(tvreturns)
    std_val = np.std(tvreturns, ddof=1)
    skew_val = stats.skew(tvreturns)
    kurt_val = stats.kurtosis(tvreturns, fisher=True)
    
    # Calculate standard errors with more precise formulas (as in tvreturns_analysis_3.py)
    n = len(tvreturns)
    mean_se = std_val / np.sqrt(n)
    skew_se = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))) if n > 3 else np.sqrt(6/n)
    kurt_se = np.sqrt(24 * n * (n - 1)**2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5))) if n > 5 else np.sqrt(24/n)
    
    # Create statistics box with format similar to tvreturns_analysis_3.py
    stats_text = f'n = {n}\n'
    stats_text += f'Mean = {mean_val:.4f} ± {mean_se:.4f}\n'
    stats_text += f'Median = {median_val:.4f}\n'
    stats_text += f'Std = {std_val:.4f}\n'
    stats_text += f'Skewness = {float(skew_val):.4f} ± {float(skew_se):.4f}\n'
    stats_text += f'Kurtosis = {float(kurt_val):.4f} ± {float(kurt_se):.4f}'
    
    ax.text(0.72, 0.88, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='none', 
           edgecolor='black', linewidth=1, alpha=1.0))
    
    # Configure logarithmic scale on y
    ax.set_yscale('log')
    
    # Ensure all data is visible on x
    data_min = np.min(tvreturns)
    data_max = np.max(tvreturns)
    x_margin = (data_max - data_min) * 0.05
    ax.set_xlim(data_min - x_margin, data_max + x_margin)
    
    # Labels and formatting
    ax.set_xlabel('TVReturns', fontsize=12)
    ax.set_ylabel('Frequency (log)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_filename = os.path.join(save_path, 'tvreturns_histogram_linear_GARCH.png')
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename

def plot_tvreturns_histogram_separated(tvreturns, save_path='TVReturns_Analysis_Plots/GARCH_plots'):
    """Plot separated TVReturns histogram (positive and negative)"""
    os.makedirs(save_path, exist_ok=True)
    
    if len(tvreturns) == 0:
        return None
    
    positive_tvreturns = tvreturns[tvreturns > 0]
    negative_tvreturns = np.abs(tvreturns[tvreturns < 0])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Blue color as in reference image
    djia_color = '#5B9BD5'
    
    # Plot positive TVReturns (left subplot)
    if len(positive_tvreturns) > 0:
        ax1.hist(positive_tvreturns, bins=50, density=False, alpha=0.8, 
                color=djia_color, edgecolor='black', linewidth=0.3, 
                label='Positive Data')
        
        # Calculate statistics for positive part
        mean_pos = np.mean(positive_tvreturns)
        median_pos = np.median(positive_tvreturns)
        std_pos = np.std(positive_tvreturns, ddof=1)
        skew_pos = stats.skew(positive_tvreturns)
        skew_pos_err = np.sqrt(6/len(positive_tvreturns))  # Approximate standard error
        kurt_pos = stats.kurtosis(positive_tvreturns, fisher=True)
        kurt_pos_err = np.sqrt(24/len(positive_tvreturns))  # Approximate standard error
        
        stats_text_pos = f'n = {len(positive_tvreturns)}\n'
        stats_text_pos += f'Mean = {mean_pos:.4f} ± {std_pos/np.sqrt(len(positive_tvreturns)):.4f}\n'
        stats_text_pos += f'Median = {median_pos:.4f}\n'
        stats_text_pos += f'Std = {std_pos:.4f}\n'
        stats_text_pos += f'Skewness = {float(skew_pos):.3f} ± {skew_pos_err:.4f}\n'
        stats_text_pos += f'Kurtosis = {float(kurt_pos):.3f} ± {kurt_pos_err:.4f}'
        
        # Calculate KL divergence KL(Pos||Neg) for positive subplot
        if len(positive_tvreturns) > 0 and len(negative_tvreturns) > 0:
            kl_div_pos_neg = calculate_kl_divergence(positive_tvreturns, negative_tvreturns, method='kde')
            if not np.isnan(kl_div_pos_neg):
                stats_text_pos += f'\nKL(Pos||Neg) = {kl_div_pos_neg:.4f}'
        
        ax1.text(0.98, 0.98, stats_text_pos, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', 
                edgecolor='black', linewidth=1, alpha=1.0))
    
    ax1.set_title('Positive VReturns', fontsize=14)
    ax1.set_xlabel('VReturns', fontsize=12)
    ax1.set_ylabel('Frequency (log)', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(fontsize=10, loc='center right')
    ax1.grid(True, alpha=0.3)
    
    # Calculate KL divergence KL(Neg||Pos) for negative subplot
    kl_divergence_neg_pos = None
    if len(positive_tvreturns) > 0 and len(negative_tvreturns) > 0:
        kl_divergence_neg_pos = calculate_kl_divergence(negative_tvreturns, positive_tvreturns, method='kde')
    
    # Plot negative TVReturns (right subplot) - absolute values
    if len(negative_tvreturns) > 0:
        ax2.hist(negative_tvreturns, bins=50, density=False, alpha=0.8, 
                color=djia_color, edgecolor='black', linewidth=0.3, 
                label='Negative Data')
        
        # Calculate statistics for negative part (absolute values)
        mean_neg = np.mean(negative_tvreturns)
        median_neg = np.median(negative_tvreturns)
        std_neg = np.std(negative_tvreturns, ddof=1)
        skew_neg = stats.skew(negative_tvreturns)
        skew_neg_err = np.sqrt(6/len(negative_tvreturns))
        kurt_neg = stats.kurtosis(negative_tvreturns, fisher=True)
        kurt_neg_err = np.sqrt(24/len(negative_tvreturns))
        
        stats_text_neg = f'n = {len(negative_tvreturns)}\n'
        stats_text_neg += f'Mean = {mean_neg:.4f} ± {std_neg/np.sqrt(len(negative_tvreturns)):.4f}\n'
        stats_text_neg += f'Median = {median_neg:.4f}\n'
        stats_text_neg += f'Std = {std_neg:.4f}\n'
        stats_text_neg += f'Skewness = {float(skew_neg):.3f} ± {skew_neg_err:.4f}\n'
        stats_text_neg += f'Kurtosis = {float(kurt_neg):.3f} ± {kurt_neg_err:.4f}'
        
        # Add KL divergence KL(Neg||Pos) to negative subplot
        if kl_divergence_neg_pos is not None and not np.isnan(kl_divergence_neg_pos):
            stats_text_neg += f'\nKL(Neg||Pos) = {kl_divergence_neg_pos:.4f}'
        
        ax2.text(0.98, 0.98, stats_text_neg, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', 
                edgecolor='black', linewidth=1, alpha=1.0))
    
    ax2.set_title('Negative VReturns (Absolute Values)', fontsize=14)
    ax2.set_xlabel('|VReturns|', fontsize=12)
    ax2.set_ylabel('Frequency (log)', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=10, loc='center right')
    ax2.grid(True, alpha=0.3)
    
    # Save plot
    plot_filename = os.path.join(save_path, 'tvreturns_histogram_linear_separated_GARCH.png')
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename

def plot_stylized_facts(prices, save_path='TVReturns_Analysis_Plots/GARCH_plots'):
    """Plot stylized facts for GARCH price series"""
    log_returns = calculate_log_returns(prices)
    abs_log_returns = np.abs(log_returns)
    
    # Create subplots with DJIA style
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # DJIA blue color as in reference image
    djia_color = '#5B9BD5'
    
    # 1. Returns distribution (without normal distribution fit)
    hist_counts, bin_edges = np.histogram(log_returns, bins=100, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    axs[0, 0].plot(bin_centers, hist_counts, 'o', color=djia_color, 
                   label='DJIA Data', markersize=3, alpha=0.8)
    
    # Calculate statistics with errors
    n = len(log_returns)
    mean_val = np.mean(log_returns)
    std_val = np.std(log_returns, ddof=1)  # Sample standard deviation
    skew_val = stats.skew(log_returns)
    kurt_val = stats.kurtosis(log_returns, fisher=True)  # Excess kurtosis
    
    # Calculate standard errors
    mean_error = std_val / np.sqrt(n)
    # Standard error for skewness
    skew_error = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
    # Standard error for kurtosis
    kurt_error = 2 * skew_error * np.sqrt((n**2 - 1) / ((n - 3) * (n + 5)))
    
    # Create statistics text box
    stats_text = (f'Mean: {mean_val:.4f} ± {mean_error:.4f}\n'
                  f'Std Dev: {std_val:.4f}\n'
                  f'Skewness: {skew_val:.4f} ± {skew_error:.4f}\n'
                  f'Kurtosis: {kurt_val:.4f} ± {kurt_error:.4f}')
    
    # Add text box with semi-transparent background and border
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1)
    axs[0, 0].text(0.05, 0.95, stats_text, transform=axs[0, 0].transAxes, fontsize=9,
                   verticalalignment='top', bbox=props)
    
    # Apply logarithmic scale to y axis
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_xlabel('Log Returns', fontsize=12)
    axs[0, 0].set_ylabel('Density (log)', fontsize=12)
    axs[0, 0].set_title('(a) Returns Distribution', fontsize=14)
    axs[0, 0].legend(fontsize=10)
    axs[0, 0].grid(True, alpha=0.3)
    
    # 2. Time series of logarithmic returns
    time_steps = np.arange(len(log_returns))
    axs[0, 1].plot(time_steps, log_returns, color=djia_color, linewidth=0.6, alpha=0.8)
    axs[0, 1].set_xlabel('Time Steps', fontsize=12)
    axs[0, 1].set_ylabel('Logarithmic Returns', fontsize=12)
    axs[0, 1].set_title('(b) Logarithmic Returns', fontsize=14)
    axs[0, 1].grid(True, alpha=0.3)
    
    # 3. Returns autocorrelation
    plot_acf(log_returns, lags=40, ax=axs[1, 0], title='(c) Returns Autocorrelation',
             color=djia_color, alpha=0.8)
    axs[1, 0].set_xlabel('Lag', fontsize=12)
    axs[1, 0].set_ylabel('Autocorrelation', fontsize=12)
    axs[1, 0].set_title('(c) Returns Autocorrelation', fontsize=14)
    axs[1, 0].grid(True, alpha=0.3)
    
    # 4. Absolute returns autocorrelation
    plot_acf(abs_log_returns, lags=40, ax=axs[1, 1], title='(d) Absolute Returns Autocorrelation',
             color=djia_color, alpha=0.8)
    axs[1, 1].set_xlabel('Lag', fontsize=12)
    axs[1, 1].set_ylabel('Autocorrelation', fontsize=12)
    axs[1, 1].set_title('(d) Absolute Returns Autocorrelation', fontsize=14)
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    plot_filename = os.path.join(save_path, 'plot2_stylized_facts_GARCH.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename

def plot_price_series(prices, save_path='TVReturns_Analysis_Plots/GARCH_plots'):
    """Plot simulated price series"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot price series
    time_index = range(len(prices))
    ax.plot(time_index, prices, color='#1f77b4', linewidth=1.0, alpha=0.8)
    
    # Labels and formatting
    ax.set_xlabel('Time steps', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    initial_price = prices[0]
    final_price = prices[-1]
    total_return = (final_price / initial_price - 1) * 100
    max_price = np.max(prices)
    min_price = np.min(prices)
    
    stats_text = f'Initial price: {initial_price:.2f}\n'
    stats_text += f'Final price: {final_price:.2f}\n'
    stats_text += f'Total return: {total_return:.2f}%\n'
    stats_text += f'Maximum price: {max_price:.2f}\n'
    stats_text += f'Minimum price: {min_price:.2f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', 
           edgecolor='black', linewidth=1, alpha=0.9))
    
    # Save plot
    plot_filename = os.path.join(save_path, 'garch_simulated_price_series.png')
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename

def plot_tvreturns_gamma_pareto_fit(tvreturns, threshold=0.04, save_path='TVReturns_Analysis_Plots/GARCH_plots'):
    """Plot 4-subplot figure with gamma and pareto fits"""
    os.makedirs(save_path, exist_ok=True)
    if len(tvreturns) == 0:
        return None
    
    # Separate positive and negative TVReturns
    positive_tvreturns = tvreturns[tvreturns > 0]
    negative_tvreturns = np.abs(tvreturns[tvreturns < 0])  # Absolute values for negatives
    
    # Split data by threshold
    pos_gamma_data = positive_tvreturns[positive_tvreturns <= threshold]
    pos_pareto_data = positive_tvreturns[positive_tvreturns > threshold]
    neg_gamma_data = negative_tvreturns[negative_tvreturns <= threshold]
    neg_pareto_data = negative_tvreturns[negative_tvreturns > threshold]
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color scheme
    djia_color = '#5B9BD5'
    fit_color = 'red'
    
    # Top left: Positive Gamma fit (0-0.04)
    if len(pos_gamma_data) > 0:
        # Create histogram data for scatter plot
        hist_counts, bin_edges = np.histogram(pos_gamma_data, bins=30, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot as scatter points instead of histogram bars
        ax1.scatter(bin_centers, hist_counts, color=djia_color, alpha=0.7, 
                   s=30, edgecolors='black', linewidth=0.3, label='Positive Data')
        
        # Fit gamma distribution
        gamma_params, ks_p, mse, rmse, mae, r2 = fit_gamma_distribution(pos_gamma_data)
        
        if gamma_params is not None:
            shape, loc, scale, shape_se, scale_se = gamma_params
            # Calculate rate and its propagated error
            rate = 1/scale
            rate_se = scale_se / (scale**2)  # Error propagation for β = 1/θ
            
            x_range = np.linspace(pos_gamma_data.min(), pos_gamma_data.max(), 100)
            fitted_pdf = gamma.pdf(x_range, shape, loc=loc, scale=scale)
            # Scale PDF to match frequency histogram
            bin_width = (pos_gamma_data.max() - pos_gamma_data.min()) / 30
            fitted_freq = fitted_pdf * len(pos_gamma_data) * bin_width
            ax1.plot(x_range, fitted_freq, color=fit_color, linewidth=2, 
                    label=f'Gamma Fit (α={shape:.3f}, θ={scale:.3f})')
            
            # Statistics box
            n = len(pos_gamma_data)
            mean_val = np.mean(pos_gamma_data)
            mean_err = np.std(pos_gamma_data, ddof=1) / np.sqrt(n)
            median_val = np.median(pos_gamma_data)
            std_val = np.std(pos_gamma_data, ddof=1)
            skew_val = stats.skew(pos_gamma_data)
            skew_err = np.sqrt(6/n)
            kurt_val = stats.kurtosis(pos_gamma_data, fisher=True)
            kurt_err = np.sqrt(24/n)
            
            stats_text = f'n = {n}\n'
            stats_text += f'Mean = {mean_val:.4f} ± {mean_err:.4f}\n'
            stats_text += f'Median = {median_val:.4f}\n'
            stats_text += f'Std = {std_val:.4f}\n'
            stats_text += f'Skewness = {skew_val:.4f} ± {skew_err:.4f}\n'
            stats_text += f'Kurtosis = {kurt_val:.4f} ± {kurt_err:.4f}\n\n'
            stats_text += f'Gamma Fit (0-{threshold}):\n'
            stats_text += f'Shape (α) = {shape:.4f} ± {shape_se:.4f}\n'
            stats_text += f'Scale (θ) = {scale:.4f} ± {scale_se:.4f}\n'
            stats_text += f'Rate (β) = {rate:.4f} ± {rate_se:.4f}\n\n'
            stats_text += f'MSE-based metrics:\n'
            stats_text += f'MSE = {mse:.6f}\n'
            stats_text += f'RMSE = {rmse:.6f}\n'
            stats_text += f'MAE = {mae:.6f}\n'
            stats_text += f'R² = {r2:.4f}\n'
            if ks_p is not None:
                stats_text += f'KS p-value = {ks_p:.4f}'
            
            ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor='black', linewidth=1, alpha=1.0))
    
    ax1.set_title('Positive VReturns - Gamma Fit', fontsize=14)
    ax1.set_xlabel('VReturns', fontsize=12)
    ax1.set_ylabel('Frequency (log)', fontsize=12)
    ax1.legend(fontsize=10, loc='upper center')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Top right: Negative Gamma fit (0-0.04)
    if len(neg_gamma_data) > 0:
        # Create histogram data for scatter plot
        hist_counts, bin_edges = np.histogram(neg_gamma_data, bins=30, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot as scatter points instead of histogram bars
        ax2.scatter(bin_centers, hist_counts, color=djia_color, alpha=0.7, 
                   s=30, edgecolors='black', linewidth=0.3, label='Negative Data')
        
        # Fit gamma distribution
        gamma_params, ks_p, mse, rmse, mae, r2 = fit_gamma_distribution(neg_gamma_data)
        
        if gamma_params is not None:
            shape, loc, scale, shape_se, scale_se = gamma_params
            # Calculate rate and its propagated error
            rate = 1/scale
            rate_se = scale_se / (scale**2)  # Error propagation for β = 1/θ
            
            x_range = np.linspace(neg_gamma_data.min(), neg_gamma_data.max(), 100)
            fitted_pdf = gamma.pdf(x_range, shape, loc=loc, scale=scale)
            # Scale PDF to match frequency histogram
            bin_width = (neg_gamma_data.max() - neg_gamma_data.min()) / 30
            fitted_freq = fitted_pdf * len(neg_gamma_data) * bin_width
            ax2.plot(x_range, fitted_freq, color=fit_color, linewidth=2, 
                    label=f'Gamma Fit (α={shape:.3f}, θ={scale:.3f})')
            
            # Statistics box
            n = len(neg_gamma_data)
            mean_val = np.mean(neg_gamma_data)
            mean_err = np.std(neg_gamma_data, ddof=1) / np.sqrt(n)
            median_val = np.median(neg_gamma_data)
            std_val = np.std(neg_gamma_data, ddof=1)
            skew_val = stats.skew(neg_gamma_data)
            skew_err = np.sqrt(6/n)
            kurt_val = stats.kurtosis(neg_gamma_data, fisher=True)
            kurt_err = np.sqrt(24/n)
            
            stats_text = f'n = {n}\n'
            stats_text += f'Mean = {mean_val:.4f} ± {mean_err:.4f}\n'
            stats_text += f'Median = {median_val:.4f}\n'
            stats_text += f'Std = {std_val:.4f}\n'
            stats_text += f'Skewness = {skew_val:.4f} ± {skew_err:.4f}\n'
            stats_text += f'Kurtosis = {kurt_val:.4f} ± {kurt_err:.4f}\n\n'
            stats_text += f'Gamma Fit (0-{threshold}):\n'
            stats_text += f'Shape (α) = {shape:.4f} ± {shape_se:.4f}\n'
            stats_text += f'Scale (θ) = {scale:.4f} ± {scale_se:.4f}\n'
            stats_text += f'Rate (β) = {rate:.4f} ± {rate_se:.4f}\n\n'
            stats_text += f'MSE-based metrics:\n'
            stats_text += f'MSE = {mse:.6f}\n'
            stats_text += f'RMSE = {rmse:.6f}\n'
            stats_text += f'MAE = {mae:.6f}\n'
            stats_text += f'R² = {r2:.4f}\n'
            if ks_p is not None:
                stats_text += f'KS p-value = {ks_p:.4f}'
            
            ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor='black', linewidth=1, alpha=1.0))
    
    ax2.set_title('Negative VReturns - Gamma Fit', fontsize=14)
    ax2.set_xlabel('|VReturns|', fontsize=12)
    ax2.set_ylabel('Frequency (log)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Bottom left: Positive Pareto fit (>0.04)
    if len(pos_pareto_data) > 0:
        # Create histogram data for scatter plot
        hist_counts, bin_edges = np.histogram(pos_pareto_data, bins=30, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot as scatter points instead of histogram bars
        ax3.scatter(bin_centers, hist_counts, color=djia_color, alpha=0.7, 
                   s=30, edgecolors='black', linewidth=0.3, label='Positive Data')
        
        # Fit Pareto distribution
        pareto_params, ks_p, mse, rmse, mae, r2 = fit_pareto_distribution(pos_pareto_data)
        
        if pareto_params is not None:
            shape, loc, scale, shape_se, scale_se = pareto_params
            x_range = np.linspace(pos_pareto_data.min(), pos_pareto_data.max(), 100)
            fitted_pdf = pareto.pdf(x_range, shape, loc=loc, scale=scale)
            # Scale PDF to match frequency histogram
            bin_width = (pos_pareto_data.max() - pos_pareto_data.min()) / 30
            fitted_freq = fitted_pdf * len(pos_pareto_data) * bin_width
            ax3.plot(x_range, fitted_freq, color=fit_color, linewidth=2, 
                    label=f'Pareto Fit (α={shape:.3f}, xₘ={scale:.3f})')
            
            # Statistics box
            n = len(pos_pareto_data)
            mean_val = np.mean(pos_pareto_data)
            mean_err = np.std(pos_pareto_data, ddof=1) / np.sqrt(n)
            median_val = np.median(pos_pareto_data)
            std_val = np.std(pos_pareto_data, ddof=1)
            skew_val = stats.skew(pos_pareto_data)
            skew_err = np.sqrt(6/n)
            kurt_val = stats.kurtosis(pos_pareto_data, fisher=True)
            kurt_err = np.sqrt(24/n)
            
            stats_text = f'n = {n}\n'
            stats_text += f'Mean = {mean_val:.4f} ± {mean_err:.4f}\n'
            stats_text += f'Median = {median_val:.4f}\n'
            stats_text += f'Std = {std_val:.4f}\n'
            stats_text += f'Skewness = {skew_val:.4f} ± {skew_err:.4f}\n'
            stats_text += f'Kurtosis = {kurt_val:.4f} ± {kurt_err:.4f}\n\n'
            stats_text += f'Pareto Fit (>{threshold}):\n'
            stats_text += f'Shape (α) = {shape:.4f} ± {shape_se:.4f}\n'
            stats_text += f'Scale (x₀) = {scale:.4f} ± {scale_se:.4f}\n\n'
            stats_text += f'MSE-based metrics:\n'
            stats_text += f'MSE = {mse:.6f}\n'
            stats_text += f'RMSE = {rmse:.6f}\n'
            stats_text += f'MAE = {mae:.6f}\n'
            stats_text += f'R² = {r2:.4f}\n'
            if ks_p is not None:
                stats_text += f'KS p-value = {ks_p:.4f}'
            
            ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor='black', linewidth=1, alpha=1.0))
    
    ax3.set_title('Positive VReturns - Pareto Fit', fontsize=14)
    ax3.set_xlabel('VReturns', fontsize=12)
    ax3.set_ylabel('Frequency (log)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Bottom right: Negative Pareto fit (>0.04)
    if len(neg_pareto_data) > 0:
        # Create histogram data for scatter plot
        hist_counts, bin_edges = np.histogram(neg_pareto_data, bins=30, density=False)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Plot as scatter points instead of histogram bars
        ax4.scatter(bin_centers, hist_counts, color=djia_color, alpha=0.7, 
                   s=30, edgecolors='black', linewidth=0.3, label='Negative Data')
        
        # Fit Pareto distribution
        pareto_params, ks_p, mse, rmse, mae, r2 = fit_pareto_distribution(neg_pareto_data)
        
        if pareto_params is not None:
            shape, loc, scale, shape_se, scale_se = pareto_params
            x_range = np.linspace(neg_pareto_data.min(), neg_pareto_data.max(), 100)
            fitted_pdf = pareto.pdf(x_range, shape, loc=loc, scale=scale)
            # Scale PDF to match frequency histogram
            bin_width = (neg_pareto_data.max() - neg_pareto_data.min()) / 30
            fitted_freq = fitted_pdf * len(neg_pareto_data) * bin_width
            ax4.plot(x_range, fitted_freq, color=fit_color, linewidth=2, 
                    label=f'Pareto Fit (α={shape:.3f}, xₘ={scale:.3f})')
            
            # Statistics box
            n = len(neg_pareto_data)
            mean_val = np.mean(neg_pareto_data)
            mean_err = np.std(neg_pareto_data, ddof=1) / np.sqrt(n)
            median_val = np.median(neg_pareto_data)
            std_val = np.std(neg_pareto_data, ddof=1)
            skew_val = stats.skew(neg_pareto_data)
            skew_err = np.sqrt(6/n)
            kurt_val = stats.kurtosis(neg_pareto_data, fisher=True)
            kurt_err = np.sqrt(24/n)
            
            stats_text = f'n = {n}\n'
            stats_text += f'Mean = {mean_val:.4f} ± {mean_err:.4f}\n'
            stats_text += f'Median = {median_val:.4f}\n'
            stats_text += f'Std = {std_val:.4f}\n'
            stats_text += f'Skewness = {skew_val:.4f} ± {skew_err:.4f}\n'
            stats_text += f'Kurtosis = {kurt_val:.4f} ± {kurt_err:.4f}\n\n'
            stats_text += f'Pareto Fit (>{threshold}):\n'
            stats_text += f'Shape (α) = {shape:.4f} ± {shape_se:.4f}\n'
            stats_text += f'Scale (x₀) = {scale:.4f} ± {scale_se:.4f}\n\n'
            stats_text += f'MSE-based metrics:\n'
            stats_text += f'MSE = {mse:.6f}\n'
            stats_text += f'RMSE = {rmse:.6f}\n'
            stats_text += f'MAE = {mae:.6f}\n'
            stats_text += f'R² = {r2:.4f}\n'
            if ks_p is not None:
                stats_text += f'KS p-value = {ks_p:.4f}'
            
            ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', 
                    edgecolor='black', linewidth=1, alpha=1.0))
    
    ax4.set_title('Negative VReturns - Pareto Fit', fontsize=14)
    ax4.set_xlabel('|VReturns|', fontsize=12)
    ax4.set_ylabel('Frequency (log)', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Save plot
    plot_filename = os.path.join(save_path, 'tvreturns_gamma_pareto_fit_GARCH.png')
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename


def main():
    """Main function to generate all 6 plots"""
    csv_filename = 'garch_simulated_prices.csv'
    try:
        df = load_garch_prices(csv_filename)
        prices = df['price'].values
    except FileNotFoundError:
        return
    
    trends = identify_trends(prices, min_trend_length=1)
    if len(trends) == 0:
        return
    
    trends_df = pd.DataFrame(trends)
    tvreturns = calculate_tvreturns(prices, trends)
    
    save_path = 'TVReturns_Analysis_Plots/GARCH_plots'
    os.makedirs(save_path, exist_ok=True)
    
    # Generate all 6 plots
    print('Generating plot 1: Trend duration distribution...')
    plot_trend_duration_distribution(trends_df, save_path)
    print('Generating plot 2: TVReturns histogram linear...')
    plot_tvreturns_histogram_linear(tvreturns, save_path)
    print('Generating plot 3: TVReturns histogram separated...')
    plot_tvreturns_histogram_separated(tvreturns, save_path)
    print('Generating plot 4: Stylized facts...')
    plot_stylized_facts(prices, save_path)
    print('Generating plot 5: Price series...')
    plot_price_series(prices, save_path)
    print('Generating plot 6: Gamma-Pareto fit...')
    plot_tvreturns_gamma_pareto_fit(tvreturns, save_path=save_path)
    print('All plots generated successfully!')

if __name__ == '__main__':
    main()
