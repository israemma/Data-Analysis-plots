import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import pandas as pd
import pickle
import warnings
import yfinance as yf
from scipy.stats import gaussian_kde, gamma, chisquare, chi2, cramervonmises, pareto
from datetime import datetime
from scipy.optimize import minimize
from sklearn.metrics import r2_score

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

# --- Standard Error Calculation Functions ---

def calculate_gamma_standard_errors(data, shape, scale):
    """
    Calculate standard errors for Gamma distribution MLE parameters using Fisher Information Matrix.
    
    Args:
        data: numpy array of positive values used for fitting
        shape: fitted shape parameter (α)
        scale: fitted scale parameter (θ)
        
    Returns:
        tuple: (shape_se, scale_se) - standard errors for shape and scale parameters
    """
    try:
        n = len(data)
        if n < 2:
            return np.nan, np.nan
        
        # Calculate Fisher Information Matrix elements for Gamma distribution
        # For Gamma(α, θ) with PDF: f(x) = (1/Γ(α)θ^α) * x^(α-1) * exp(-x/θ)
        
        # Second derivatives of log-likelihood
        from scipy.special import polygamma
        
        # I_αα = n * ψ'(α) where ψ'(α) is the trigamma function
        I_alpha_alpha = n * polygamma(1, shape)
        
        # I_θθ = n * α / θ²
        I_theta_theta = n * shape / (scale ** 2)
        
        # I_αθ = I_θα = n / θ
        I_alpha_theta = n / scale
        
        # Fisher Information Matrix
        fisher_matrix = np.array([[I_alpha_alpha, I_alpha_theta],
                                 [I_alpha_theta, I_theta_theta]])
        
        # Standard errors are square roots of diagonal elements of inverse Fisher matrix
        try:
            inv_fisher = np.linalg.inv(fisher_matrix)
            shape_se = np.sqrt(inv_fisher[0, 0])
            scale_se = np.sqrt(inv_fisher[1, 1])
            
            # Validate results
            if np.isnan(shape_se) or np.isnan(scale_se) or shape_se <= 0 or scale_se <= 0:
                return np.nan, np.nan
                
            return shape_se, scale_se
            
        except np.linalg.LinAlgError:
            # Matrix is singular, use asymptotic approximation
            shape_se = np.sqrt(1.0 / I_alpha_alpha) if I_alpha_alpha > 0 else np.nan
            scale_se = np.sqrt(1.0 / I_theta_theta) if I_theta_theta > 0 else np.nan
            return shape_se, scale_se
            
    except Exception as e:
        return np.nan, np.nan


def calculate_pareto_standard_errors(data, b_param, x_min):
    """
    Calculate standard errors for Pareto distribution MLE parameters.
    
    Args:
        data: numpy array of positive values used for fitting (must be >= x_min)
        b_param: fitted shape parameter (α or b)
        x_min: fitted scale parameter (x₀)
        
    Returns:
        tuple: (b_param_se, x_min_se) - standard errors for shape and scale parameters
    """
    try:
        n = len(data)
        if n < 2:
            return np.nan, np.nan
        
        # For Pareto Type I distribution: f(x) = (b * x_min^b) / x^(b+1) for x >= x_min
        # MLE estimators:
        # b_hat = n / sum(ln(x_i/x_min))
        # x_min is typically fixed at min(data)
        
        # Standard error for shape parameter b using asymptotic theory
        # Var(b_hat) ≈ b² / n (asymptotic variance)
        b_param_se = b_param / np.sqrt(n)
        
        # For scale parameter x_min (when it's the minimum of the data)
        # This is more complex, but for practical purposes when x_min = min(data),
        # we can use order statistics theory
        # For the minimum of Pareto distribution, the standard error is approximately:
        x_min_se = x_min / (np.sqrt(n) * b_param)
        
        # Validate results
        if np.isnan(b_param_se) or np.isnan(x_min_se) or b_param_se <= 0 or x_min_se <= 0:
            return np.nan, np.nan
            
        return b_param_se, x_min_se
        
    except Exception as e:
        return np.nan, np.nan

def calculate_rate_parameter_error(scale, scale_se):
    """
    Calculate propagated error for Rate parameter (β = 1/θ) derived from Scale parameter.
    
    Args:
        scale: fitted scale parameter (θ)
        scale_se: standard error of scale parameter
        
    Returns:
        tuple: (rate, rate_se) - rate parameter and its standard error
    """
    try:
        if scale <= 0 or np.isnan(scale) or np.isnan(scale_se):
            return np.nan, np.nan
        
        # Rate parameter β = 1/θ
        rate = 1.0 / scale
        
        # Error propagation: if β = 1/θ, then SE(β) = SE(θ)/θ²
        rate_se = scale_se / (scale ** 2)
        
        # Validate results
        if np.isnan(rate) or np.isnan(rate_se) or rate_se <= 0:
            return np.nan, np.nan
            
        return rate, rate_se
        
    except Exception as e:
        return np.nan, np.nan

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

def download_financial_data():
    """Download financial data for specified indices."""
    indices = {
        'DJIA': '^DJI',
        'DAX': '^GDAXI', 
        'IPC': '^MXX',
        'Nikkei': '^N225'
    }
    
    start_date = '1992-01-02'
    end_date = '2023-12-29'
    
    # Try to load from pickle files first
    all_prices_data, all_prices_raw = load_financial_data(indices.keys())
    
    if all_prices_data:
        return all_prices_data, all_prices_raw
    
    # If pickle files don't exist, download from Yahoo Finance
    print("Downloading financial data from Yahoo Finance...")
    all_prices_data = {}
    all_prices_raw = {}
    
    for index_name, ticker in indices.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                all_prices_data[index_name] = data['Close']
                all_prices_raw[index_name] = data
            else:
                all_prices_data[index_name] = pd.Series(dtype=float)
                all_prices_raw[index_name] = pd.DataFrame()
        except Exception as e:
            all_prices_data[index_name] = pd.Series(dtype=float)
            all_prices_raw[index_name] = pd.DataFrame()
    
    return all_prices_data, all_prices_raw

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
    
    filtered_data = {}
    filtered_all_prices_raw = {}
    
    for index_name, price_series in data.items():
        # Use index-specific start date
        index_start = pd.to_datetime(index_start_dates.get(index_name, start_date))
        
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

# --- Trend Analysis Functions ---

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

def calculate_VReturns(prices, trends):
    """Calculate VReturns for given trends."""
    VReturns = []
    
    for trend in trends:
        start_idx = trend['start']
        end_idx = trend['end']
        duration = end_idx - start_idx
        
        if duration < 1:  # Need at least 1 day duration
            continue
            
        if start_idx >= len(prices) or prices[start_idx] <= 0:
            continue
        
        log_p_start = np.log(prices[start_idx])
        
        # Calculate VReturns for this trend
        for i in range(1, duration + 1):
            price_idx = start_idx + i
            if price_idx < len(prices):
                p_i = prices[price_idx]
                if p_i > 0:
                    log_p_i = np.log(p_i)
                    vreturn = (log_p_i - log_p_start) / i
                    VReturns.append(vreturn)
    
    return np.array(VReturns)

def extract_VReturns_by_index(all_prices_data):
    """Extract VReturns for all indices."""
    VReturns_by_index = {}
    
    for index_name, prices in all_prices_data.items():
        if prices.empty:
            VReturns_by_index[index_name] = np.array([])
            continue
            
        trends = identify_trends(prices.values)
        VReturns = calculate_VReturns(prices.values, trends)
        VReturns_by_index[index_name] = VReturns
        
    return VReturns_by_index

# --- Gamma Distribution Fitting Functions ---

def ad_statistic_custom(data, cdf_func):
    """Calculate Anderson-Darling statistic manually for any distribution."""
    n = len(data)
    data_sorted = np.sort(data)
    cdf_values = cdf_func(data_sorted)
    
    # Avoid log(0) by clipping values
    cdf_values = np.clip(cdf_values, 1e-10, 1-1e-10)
    
    i = np.arange(1, n+1)
    ad = -n - np.sum((2*i - 1)/n * (np.log(cdf_values) + np.log(1 - cdf_values[::-1])))
    return ad

def find_optimal_gamma_range(data, min_percentile=5, max_percentile=95, step=2.5):
    """
    Find the optimal data range that maximizes Gamma distribution fit quality.
    Excludes the initial 5% of data (left tail) and optimizes the right tail cutoff.
    
    Args:
        data: numpy array of positive values
        min_percentile: minimum percentile to start testing (default 5%)
        max_percentile: maximum percentile to start testing (default 95%)
        step: step size for percentile search (default 2.5%)
        
    Returns:
        tuple: (optimal_lower_percentile, optimal_upper_percentile, best_ks_pvalue, optimal_range_data)
    """
    if len(data) < 20:
        return 5, 100, 0, data
    
    best_ks_pvalue = 0
    best_lower = 5  # Always exclude the initial 5% of data
    best_upper = 100
    best_range_data = data
    
    # Test different upper percentile cutoffs (exclude right tail)
    for upper_p in np.arange(max_percentile, 100 + step, step):
        if upper_p - 5 < 65:  # Ensure we keep at least 65% of data (from 5% to upper_p)
            continue
        
        # Get data from 5% to upper_p percentile (exclude both left 5% and right tail)
        lower_bound = np.percentile(data, 5)
        upper_bound = np.percentile(data, upper_p)
        range_data = data[(data >= lower_bound) & (data <= upper_bound)]
        
        if len(range_data) < 20:  # Need sufficient data
            continue
            
        try:
            # Fit Gamma to this range
            shape, loc, scale = stats.gamma.fit(range_data, floc=0)
            
            # Test goodness of fit with KS test
            ks_stat, ks_pvalue = stats.kstest(range_data, 
                                            lambda x: stats.gamma.cdf(x, shape, loc=loc, scale=scale))
            
            # Higher p-value means better fit (we want to maximize p-value)
            if ks_pvalue > best_ks_pvalue:
                best_ks_pvalue = ks_pvalue
                best_lower = 5  # Always exclude the initial 5%
                best_upper = upper_p
                best_range_data = range_data
                
        except:
            continue
    
    return best_lower, best_upper, best_ks_pvalue, best_range_data

def fit_pareto_distribution(data, gamma_fit_bounds=None):
    """
    Fit a Pareto distribution to the data excluded from gamma fit range.
    Completely excludes the left 5% of data from both fitting and visualization.
    
    Args:
        data: numpy array of positive values
        gamma_fit_bounds: tuple (lower_bound, upper_bound) from gamma fit to exclude
        
    Returns:
        dict: Contains fitted parameters and goodness-of-fit metrics for Pareto distribution
    """
    if len(data) < 10:
        return None
    
    try:
        # First, exclude the left 5% of data completely
        left_5_percentile = np.percentile(data, 5)
        data_without_left_5 = data[data >= left_5_percentile]
        
        # Identify excluded data (only right tail for Pareto fitting)
        if gamma_fit_bounds is not None:
            lower_bound, upper_bound = gamma_fit_bounds
            # Only use right tail (data > upper_bound) for Pareto fitting
            # But exclude the left 5% completely
            excluded_data = data_without_left_5[data_without_left_5 > upper_bound]
            right_tail = excluded_data
            left_tail = np.array([])  # No left tail for Pareto
        else:
            excluded_data = data_without_left_5
            left_tail = np.array([])
            right_tail = data_without_left_5
        
        if len(excluded_data) < 5:
            return None
        
        # For Pareto fitting, use ALL excluded data points (already without left 5%)
        pareto_data = excluded_data
        
        # Set x_min to the minimum of excluded data
        x_min = np.min(pareto_data)
        
        if len(pareto_data) < 5:
            return None
        
        # Calculate Pareto shape parameter using MLE
        try:
            # Manual MLE for Pareto distribution
            # For Pareto Type I: f(x) = (b * x_min^b) / x^(b+1) for x >= x_min
            # MLE estimator: b = n / sum(ln(x_i/x_min))
            log_ratios = np.log(pareto_data / x_min)
            b_param = len(pareto_data) / np.sum(log_ratios)
            
            # For our implementation, we'll use x_min as the scale parameter
            scale_param = x_min
            
            # Validate parameters
            if np.isnan(b_param) or np.isinf(b_param) or b_param <= 0:
                return None
                
        except Exception as fit_error:
            return None
        
        # Calculate goodness of fit metrics and visualization data
        try:
            # Determine visualization range - only cover the right tail data
            if gamma_fit_bounds is not None:
                # For Pareto fitting, we only use right tail (data > upper_bound)
                lower_bound, upper_bound = gamma_fit_bounds
                
                # Only visualize the right tail data
                if len(pareto_data) > 0:
                    vis_x_min = np.min(pareto_data)
                    vis_x_max = np.max(pareto_data)
                else:
                    vis_x_min = x_min
                    vis_x_max = x_min * 2  # fallback
            else:
                vis_x_min = x_min
                vis_x_max = np.max(pareto_data)
            
            # Create x values for visualization - only for right tail
            x_theory = np.linspace(vis_x_min, vis_x_max, 100)
            
            # Calculate Pareto PDF: f(x) = (b * x_min^b) / x^(b+1)
            # Note: x_min is the fitted parameter, vis_x_min is for visualization
            y_theory = (b_param * (x_min ** b_param)) / (x_theory ** (b_param + 1))
            
            # Scale the theoretical curve to match the empirical density
            # Use a more robust approach: match the curve to multiple empirical points
            try:
                # Create histogram for the Pareto data to get empirical density
                hist_counts, bin_edges = np.histogram(pareto_data, bins=max(15, int(len(pareto_data)**0.5)), density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Find bins that overlap with our theoretical curve range
                x_min_theory = x_theory[0]
                x_max_theory = x_theory[-1]
                
                # Get bins within the theoretical range
                valid_bins = (bin_centers >= x_min_theory) & (bin_centers <= x_max_theory)
                
                if np.sum(valid_bins) > 0:
                    # Get empirical densities for valid bins
                    empirical_densities = hist_counts[valid_bins]
                    empirical_x = bin_centers[valid_bins]
                    
                    # Calculate theoretical densities at the same x points
                    theoretical_densities = (b_param * (x_min ** b_param)) / (empirical_x ** (b_param + 1))
                    
                    # Use least squares to find the best scaling factor
                    # that minimizes the difference between empirical and scaled theoretical
                    valid_mask = (empirical_densities > 0) & (theoretical_densities > 0)
                    
                    if np.sum(valid_mask) > 0:
                        emp_valid = empirical_densities[valid_mask]
                        theo_valid = theoretical_densities[valid_mask]
                        
                        # Calculate scaling factor using least squares
                        scaling_factor = np.sum(emp_valid * theo_valid) / np.sum(theo_valid ** 2)
                        
                        # Apply scaling
                        y_theory = y_theory * scaling_factor
                    else:
                        # Fallback to single point matching at the beginning
                        closest_bin_idx = np.argmin(np.abs(bin_centers - x_min_theory))
                        empirical_density_at_start = hist_counts[closest_bin_idx]
                        theoretical_density_at_start = (b_param * (x_min ** b_param)) / (x_min_theory ** (b_param + 1))
                        
                        if theoretical_density_at_start > 0 and empirical_density_at_start > 0:
                            scaling_factor = empirical_density_at_start / theoretical_density_at_start
                            y_theory = y_theory * scaling_factor
                else:
                    # Fallback to single point matching at the beginning
                    closest_bin_idx = np.argmin(np.abs(bin_centers - x_min_theory))
                    empirical_density_at_start = hist_counts[closest_bin_idx]
                    theoretical_density_at_start = (b_param * (x_min ** b_param)) / (x_min_theory ** (b_param + 1))
                    
                    if theoretical_density_at_start > 0 and empirical_density_at_start > 0:
                        scaling_factor = empirical_density_at_start / theoretical_density_at_start
                        y_theory = y_theory * scaling_factor
            
            except Exception as scaling_error:
                pass
            
            # Store for visualization
            x_vis_original = x_theory
            y_vis_original = y_theory
            
            # Kolmogorov-Smirnov test using custom Pareto CDF
            def pareto_cdf(x):
                """Custom Pareto CDF: F(x) = 1 - (x_min/x)^b for x >= x_min"""
                return np.where(x >= x_min, 1 - (x_min / x) ** b_param, 0)
            
            try:
                ks_stat, ks_p_value = stats.kstest(pareto_data, pareto_cdf)
            except Exception as ks_error:
                ks_stat, ks_p_value = np.nan, np.nan
        
        except Exception as e:
            ks_stat, ks_p_value = np.nan, np.nan
        
        # Calculate R-squared by comparing empirical and theoretical densities
        try:
            # Create histogram for pareto data
            hist_counts, bin_edges = np.histogram(pareto_data, bins=max(10, int(len(pareto_data)**0.5)), density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Get theoretical densities at bin centers using Pareto PDF
            theoretical_densities = (b_param * (x_min ** b_param)) / (bin_centers ** (b_param + 1))
            
            # Calculate R-squared
            mask = (hist_counts > 0) & (bin_centers >= x_min)
            if np.sum(mask) > 2:
                r2 = r2_score(hist_counts[mask], theoretical_densities[mask])
                rmse = np.sqrt(np.mean((hist_counts[mask] - theoretical_densities[mask]) ** 2))
            else:
                r2, rmse = np.nan, np.nan
        except Exception as r2_error:
            r2, rmse = np.nan, np.nan
        # Calculate standard errors for Pareto parameters
        b_param_se, x_min_se = calculate_pareto_standard_errors(pareto_data, b_param, x_min)
        
        return {
            'b_param': b_param,  # Pareto shape parameter
            'scale_param': scale_param,  # Pareto scale parameter
            'x_min': x_min,  # Minimum value for Pareto distribution
            'b_param_se': b_param_se,  # Standard error for shape parameter
            'x_min_se': x_min_se,  # Standard error for scale parameter
            'r2': r2,
            'rmse': rmse,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p_value,
            'n_samples': len(excluded_data),  # Use excluded_data count to match subplot n
            'excluded_data': excluded_data,
            'pareto_data': pareto_data,  # Data actually used for fitting (same as excluded_data now)
            'pareto_fit_n': len(pareto_data),  # Same as n_samples now
            'vis_x_min': vis_x_min,
            'vis_x_max': vis_x_max,
            'x_vis_original': x_vis_original,
            'y_vis_original': y_vis_original
        }
        
    except Exception as e:
        return None


def fit_gamma_distribution(data, index_name=None):
    """
    Fit a Gamma distribution to the data with optimal range selection and return parameters and goodness-of-fit metrics.
    
    Args:
        data: numpy array of positive values
        index_name: optional string to identify the index (for DJIA-specific parameters)
        
    Returns:
        dict: Contains fitted parameters, Chi-squared GoF, Cramér-von Mises, and other metrics
    """
    if len(data) < 10:
        return None
    
    try:
        # Find optimal range for fitting (excluding extreme tails)
        # Use different parameters for DJIA and Nikkei to test wider range (75% to 100% instead of 95% to 100%) with 0.5% steps
        if index_name == 'DJIA':
            optimal_lower_p, optimal_upper_p, optimal_ks_p, optimal_data = find_optimal_gamma_range(data, max_percentile=75, step=0.5)
        elif index_name == 'Nikkei':
            optimal_lower_p, optimal_upper_p, optimal_ks_p, optimal_data = find_optimal_gamma_range(data, max_percentile=75, step=0.5)
        else:
            optimal_lower_p, optimal_upper_p, optimal_ks_p, optimal_data = find_optimal_gamma_range(data)
        
        # Fit gamma distribution using MLE on optimal range
        shape, loc, scale = stats.gamma.fit(optimal_data, floc=0)  # Fix location to 0
        
        # Calculate fit range bounds for visualization
        fit_lower_bound = np.percentile(data, optimal_lower_p)
        fit_upper_bound = np.percentile(data, optimal_upper_p)
        
        # Generate fitted distribution for comparison (limited to fitting range for better visualization)
        x_fit = np.linspace(fit_lower_bound, fit_upper_bound, 1000)
        y_fit = stats.gamma.pdf(x_fit, shape, loc=loc, scale=scale)
        
        # Chi-squared Goodness of Fit test (on optimal range)
        chi2_stat = np.nan
        chi2_p = np.nan
        chi2_df = np.nan
        
        try:
            # Use a more conservative approach with fewer bins
            n_bins = max(8, min(20, int(np.ceil(len(optimal_data)**(1/3)))))
            
            # Create histogram with observed counts (on optimal range)
            hist_counts, bin_edges = np.histogram(optimal_data, bins=n_bins)
            
            # Calculate expected counts using the fitted Gamma distribution
            expected = np.zeros(len(hist_counts))
            for i in range(len(bin_edges)-1):
                # Calculate probability for each bin using CDF differences
                p_low = stats.gamma.cdf(bin_edges[i], shape, loc=loc, scale=scale)
                p_high = stats.gamma.cdf(bin_edges[i+1], shape, loc=loc, scale=scale)
                expected[i] = len(optimal_data) * (p_high - p_low)  # Use optimal_data length
            
            # Remove bins with very low expected counts (< 1)
            mask = expected >= 1.0
            observed_filtered = hist_counts[mask]
            expected_filtered = expected[mask]
            
            # Ensure we have enough bins for a valid test
            if len(observed_filtered) >= 3 and np.sum(expected_filtered) > 0:
                # Normalize expected to match observed total (handles numerical precision)
                expected_filtered = expected_filtered * (np.sum(observed_filtered) / np.sum(expected_filtered))
                
                # Perform Chi-squared test
                chi2_stat, chi2_p = chisquare(observed_filtered, expected_filtered)
                chi2_df = max(1, len(observed_filtered) - 3)  # df = bins - 1 - parameters
                
        except Exception as e:
            pass
        
        # Cramér-von Mises test (on optimal range)
        cvm_statistic = np.nan
        cvm_pvalue = np.nan
        
        try:
            cvm_result = cramervonmises(optimal_data, 'gamma', args=(shape, loc, scale))
            cvm_statistic = cvm_result.statistic
            cvm_pvalue = cvm_result.pvalue
        except Exception as e:
            pass
        
        # Kolmogorov-Smirnov test (on optimal range)
        ks_statistic, ks_p_value = stats.kstest(optimal_data, lambda x: stats.gamma.cdf(x, shape, loc=loc, scale=scale))
        
        # Anderson-Darling test (on optimal range)
        ad_statistic = np.nan
        try:
            ad_statistic = ad_statistic_custom(optimal_data, lambda x: stats.gamma.cdf(x, shape, loc=loc, scale=scale))
        except Exception as e:
            pass
        
        # Calculate standard errors for Gamma parameters
        shape_se, scale_se = calculate_gamma_standard_errors(optimal_data, shape, scale)
        
        # Calculate rate parameter and its error
        rate, rate_se = calculate_rate_parameter_error(scale, scale_se)
        
        return {
            'shape': shape,
            'loc': loc,
            'scale': scale,
            'shape_se': shape_se,
            'scale_se': scale_se,
            'rate': rate,
            'rate_se': rate_se,
            'x_fit': x_fit,
            'y_fit': y_fit,
            'chi2_stat': chi2_stat,
            'chi2_p': chi2_p,
            'chi2_df': chi2_df,
            'cvm_statistic': cvm_statistic,
            'cvm_pvalue': cvm_pvalue,
            'ks_statistic': ks_statistic,
            'ks_p_value': ks_p_value,
            'ad_statistic': ad_statistic,
            'n_samples': len(data),
            'mean_data': np.mean(data),
            'std_data': np.std(data),
            'optimal_lower_percentile': optimal_lower_p,
            'optimal_upper_percentile': optimal_upper_p,
            'optimal_ks_pvalue': optimal_ks_p,
            'fit_lower_bound': fit_lower_bound,
            'fit_upper_bound': fit_upper_bound,
            'n_samples_used_for_fit': len(optimal_data)
        }
        
    except Exception as e:
        return None

def calculate_kl_divergence(data1, data2, method='kde'):
    """Calculate KL divergence between two datasets using KDE."""
    try:
        if len(data1) < 5 or len(data2) < 5:
            return np.nan
        
        # Create KDE for both datasets
        kde1 = gaussian_kde(data1)
        kde2 = gaussian_kde(data2)
        
        # Create common support
        x_min = min(np.min(data1), np.min(data2))
        x_max = max(np.max(data1), np.max(data2))
        x = np.linspace(x_min, x_max, 1000)
        
        # Evaluate PDFs
        p = kde1(x)
        q = kde2(x)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p = np.maximum(p, epsilon)
        q = np.maximum(q, epsilon)
        
        # Normalize to ensure they are proper probability distributions
        p = p / np.trapz(p, x)
        q = q / np.trapz(q, x)
        
        # Calculate KL divergence: KL(P||Q) = ∫ p(x) log(p(x)/q(x)) dx
        kl_div = np.trapz(p * np.log(p / q), x)
        
        return kl_div
        
    except Exception as e:
        return np.nan

# --- Plotting Functions ---

def plot_VReturns_histogram_4_subplots_separated(VReturns_by_index, save_path='TVReturns_Analysis_Plots'):
    """
    Generate plots with 4 subplots per index:
    - Top 2 subplots: Data fitted with Gamma distribution only
    - Bottom 2 subplots: Data excluded from Gamma, fitted with Pareto distribution
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
    gamma_results = {}
    
    # Generate separate plot for each index
    for index_name, VReturns in VReturns_by_index.items():
        if len(VReturns) > 0:
            # Initialize KL results for this index first
            kl_results[index_name] = {
                'kl_pos_neg': np.nan, 'kl_neg_pos': np.nan,
                'gamma_kl_pos_neg': np.nan, 'gamma_kl_neg_pos': np.nan,
                'pareto_kl_pos_neg': np.nan, 'pareto_kl_neg_pos': np.nan
            }
            
            # Separate positive and negative VReturns
            positive_VReturns = VReturns[VReturns > 0]
            negative_VReturns = VReturns[VReturns < 0]
            
            # Fit Gamma distributions
            gamma_fit_pos = None
            gamma_fit_neg = None
            pareto_fit_pos = None
            pareto_fit_neg = None
            
            # Fit positive data
            if len(positive_VReturns) > 10:
                gamma_fit_pos = fit_gamma_distribution(positive_VReturns, index_name)
                if gamma_fit_pos:
                    # Fit Pareto to excluded data (outside gamma fit range)
                    gamma_bounds = (gamma_fit_pos['fit_lower_bound'], gamma_fit_pos['fit_upper_bound'])
                    pareto_fit_pos = fit_pareto_distribution(positive_VReturns, gamma_bounds)
            # Fit negative data
            if len(negative_VReturns) > 10:
                negative_abs = np.abs(negative_VReturns)
                gamma_fit_neg = fit_gamma_distribution(negative_abs, index_name)
                if gamma_fit_neg:
                    # Recalculate KS p-value for Nikkei using manual cut
                    if index_name == 'Nikkei':
                        manual_cut = 0.06
                        gamma_lower_neg = gamma_fit_neg['fit_lower_bound']
                        manual_gamma_data = negative_abs[(negative_abs >= gamma_lower_neg) & (negative_abs <= manual_cut)]
                        if len(manual_gamma_data) > 0:
                            # Recalculate KS test with manual cut data
                            shape = gamma_fit_neg['shape']
                            loc = gamma_fit_neg['loc']
                            scale = gamma_fit_neg['scale']
                            ks_statistic, ks_p_value_manual = stats.kstest(manual_gamma_data, lambda x: stats.gamma.cdf(x, shape, loc=loc, scale=scale))
                            gamma_fit_neg['ks_p_value'] = ks_p_value_manual

                    # Fit Pareto to excluded data (outside gamma fit range)
                    gamma_bounds = (gamma_fit_neg['fit_lower_bound'], gamma_fit_neg['fit_upper_bound'])

                    # Special manual cut for Nikkei negative VReturns at 0.06 (same as plotting)
                    if index_name == 'Nikkei':
                        manual_cut = 0.06
                        gamma_bounds = (gamma_fit_neg['fit_lower_bound'], manual_cut)

                    pareto_fit_neg = fit_pareto_distribution(negative_abs, gamma_bounds)

            # Calculate KL divergences separately for Gamma and Pareto data
            # KL divergence for Gamma data (data within gamma fit range)
            if gamma_fit_pos and gamma_fit_neg:
                # Get gamma data for positive
                gamma_lower_pos = gamma_fit_pos['fit_lower_bound']
                gamma_upper_pos = gamma_fit_pos['fit_upper_bound']
                gamma_data_pos = positive_VReturns[(positive_VReturns >= gamma_lower_pos) & (positive_VReturns <= gamma_upper_pos)]

                # Get gamma data for negative (absolute values)
                gamma_lower_neg = gamma_fit_neg['fit_lower_bound']
                gamma_upper_neg = gamma_fit_neg['fit_upper_bound']
                negative_abs = np.abs(negative_VReturns)
                gamma_data_neg = negative_abs[(negative_abs >= gamma_lower_neg) & (negative_abs <= gamma_upper_neg)]

                gamma_kl_pos_neg = calculate_kl_divergence(gamma_data_pos, gamma_data_neg, method='kde')
                gamma_kl_neg_pos = calculate_kl_divergence(gamma_data_neg, gamma_data_pos, method='kde')

                kl_results[index_name]['gamma_kl_pos_neg'] = gamma_kl_pos_neg
                kl_results[index_name]['gamma_kl_neg_pos'] = gamma_kl_neg_pos

            # KL divergence for Pareto data (excluded data from gamma fit)
            if gamma_fit_pos and gamma_fit_neg:
                # Get excluded data for positive (Pareto data) - only right tail to match subplot
                gamma_upper_pos = gamma_fit_pos['fit_upper_bound']
                # Remove left 5% first, then get right tail data (same as subplot)
                data_without_left_5_pos = positive_VReturns[positive_VReturns >= np.percentile(positive_VReturns, 5)]
                excluded_data_pos = data_without_left_5_pos[data_without_left_5_pos > gamma_upper_pos]

                # Get excluded data for negative (Pareto data, absolute values) - only right tail to match subplot
                gamma_upper_neg = gamma_fit_neg['fit_upper_bound']
                negative_abs = np.abs(negative_VReturns)
                # Remove left 5% first, then get right tail data (same as subplot)
                data_without_left_5_neg = negative_abs[negative_abs >= np.percentile(negative_abs, 5)]

                # Special manual cut for Nikkei negative VReturns at 0.06 (same as plotting section)
                if index_name == 'Nikkei':
                    manual_cut = 0.06
                    excluded_data_neg = data_without_left_5_neg[data_without_left_5_neg > manual_cut]
                else:
                    excluded_data_neg = data_without_left_5_neg[data_without_left_5_neg > gamma_upper_neg]

                if len(excluded_data_pos) > 0 and len(excluded_data_neg) > 0:
                    pareto_kl_pos_neg = calculate_kl_divergence(excluded_data_pos, excluded_data_neg, method='kde')
                    pareto_kl_neg_pos = calculate_kl_divergence(excluded_data_neg, excluded_data_pos, method='kde')

                    kl_results[index_name]['pareto_kl_pos_neg'] = pareto_kl_pos_neg
                    kl_results[index_name]['pareto_kl_neg_pos'] = pareto_kl_neg_pos

            # Calculate KL divergence between positive and negative distributions (legacy)
            if len(positive_VReturns) > 0 and len(negative_VReturns) > 0:
                negative_abs = np.abs(negative_VReturns)
                kl_div_pos_neg = calculate_kl_divergence(positive_VReturns, negative_abs, method='kde')
                kl_div_neg_pos = calculate_kl_divergence(negative_abs, positive_VReturns, method='kde')

                kl_results[index_name]['kl_pos_neg'] = kl_div_pos_neg
                kl_results[index_name]['kl_neg_pos'] = kl_div_neg_pos
            else:
                kl_results[index_name]['kl_pos_neg'] = np.nan
                kl_results[index_name]['kl_neg_pos'] = np.nan

            # Store gamma results
            gamma_results[index_name] = {
                'positive': gamma_fit_pos,
                'negative': gamma_fit_neg
            }

            # Create figure with 4 subplots (2x2 grid)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

            # TOP LEFT: Positive VReturns with Gamma fit only (data within gamma range)
            if len(positive_VReturns) > 0 and gamma_fit_pos:
                # Filter data to gamma fit range only
                gamma_lower = gamma_fit_pos['fit_lower_bound']
                gamma_upper = gamma_fit_pos['fit_upper_bound']
                gamma_data_pos = positive_VReturns[(positive_VReturns >= gamma_lower) & (positive_VReturns <= gamma_upper)]

                if len(gamma_data_pos) > 0:
                    # Calculate histogram for gamma data only
                    hist_counts, bin_edges = np.histogram(gamma_data_pos, bins=30, density=False)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    # Create scatter plot
                    ax1.scatter(bin_centers, hist_counts, alpha=0.7, 
                               color=colors.get(index_name, '#000000'), 
                               s=30, label=f'{index_name} Positive')

                    # Plot Gamma fit - only in optimal range, scaled for frequency
                    fit_mask = (gamma_fit_pos['x_fit'] >= gamma_lower) & (gamma_fit_pos['x_fit'] <= gamma_upper)
                    # Scale PDF to frequency: multiply by bin width and total count
                    bin_width = (bin_edges[1] - bin_edges[0])
                    scaled_y_fit = gamma_fit_pos['y_fit'][fit_mask] * bin_width * len(gamma_data_pos)
                    ax1.plot(gamma_fit_pos['x_fit'][fit_mask], scaled_y_fit, 
                            'r-', linewidth=2, label=f'Gamma Fit')

                    # Statistics for gamma data
                    mean_gamma_pos = np.mean(gamma_data_pos)
                    median_gamma_pos = np.median(gamma_data_pos)
                    std_gamma_pos = np.std(gamma_data_pos, ddof=1)
                    skew_gamma_pos = stats.skew(gamma_data_pos)
                    kurt_gamma_pos = stats.kurtosis(gamma_data_pos, fisher=True)
                    n_gamma_pos = len(gamma_data_pos)

                    # Calculate standard errors
                    mean_se_gamma_pos = std_gamma_pos / np.sqrt(n_gamma_pos)
                    skew_se_gamma_pos = np.sqrt(6 * n_gamma_pos * (n_gamma_pos - 1) / ((n_gamma_pos - 2) * (n_gamma_pos + 1) * (n_gamma_pos + 3)))
                    kurt_se_gamma_pos = np.sqrt(24 * n_gamma_pos * (n_gamma_pos - 1)**2 / ((n_gamma_pos - 3) * (n_gamma_pos - 2) * (n_gamma_pos + 3) * (n_gamma_pos + 5)))

                    stats_text = f'n = {n_gamma_pos}\n'
                    stats_text += f'Mean = {mean_gamma_pos:.4f} ± {mean_se_gamma_pos:.4f}\n'
                    stats_text += f'Median = {median_gamma_pos:.4f}\n'
                    stats_text += f'Std = {std_gamma_pos:.4f}\n'
                    stats_text += f'Skewness = {float(skew_gamma_pos):.4f} ± {skew_se_gamma_pos:.4f}\n'
                    stats_text += f'Kurtosis = {float(kurt_gamma_pos):.4f} ± {kurt_se_gamma_pos:.4f}\n'
                    stats_text += f'\n'
                    stats_text += f'Gamma Fit:\n'
                    # Display Shape and Scale with standard errors
                    if not np.isnan(gamma_fit_pos.get("shape_se", np.nan)):
                        stats_text += f'Shape (α) = {gamma_fit_pos["shape"]:.4f} ± {gamma_fit_pos["shape_se"]:.4f}\n'
                    else:
                        stats_text += f'Shape (α) = {gamma_fit_pos["shape"]:.4f}\n'

                    if not np.isnan(gamma_fit_pos.get("scale_se", np.nan)):
                        stats_text += f'Scale (θ) = {gamma_fit_pos["scale"]:.4f} ± {gamma_fit_pos["scale_se"]:.4f}\n'
                    else:
                        stats_text += f'Scale (θ) = {gamma_fit_pos["scale"]:.4f}\n'

                    # Display Rate parameter with standard error if available
                    if not np.isnan(gamma_fit_pos.get("rate", np.nan)) and not np.isnan(gamma_fit_pos.get("rate_se", np.nan)):
                        stats_text += f'Rate (β) = {gamma_fit_pos["rate"]:.4f} ± {gamma_fit_pos["rate_se"]:.4f}\n'
                    else:
                        stats_text += f'Rate (β) = {1/gamma_fit_pos["scale"]:.4f}\n'
                    stats_text += f'\n'
                    stats_text += f'Fit Range = {gamma_fit_pos["optimal_lower_percentile"]:.1f}%-{gamma_fit_pos["optimal_upper_percentile"]:.1f}%\n'
                    stats_text += f'Fit n = {gamma_fit_pos["n_samples_used_for_fit"]}\n'
                    # Calculate excluded points (both left and right tails)
                    excluded_left_count_pos = np.sum(positive_VReturns < gamma_fit_pos["fit_lower_bound"])
                    excluded_right_count_pos = np.sum(positive_VReturns > gamma_fit_pos["fit_upper_bound"])
                    total_excluded_pos = excluded_left_count_pos + excluded_right_count_pos

                    stats_text += f'n exc = {total_excluded_pos} (L {excluded_left_count_pos}, R {excluded_right_count_pos})\n'
                    stats_text += f'\n'
                    stats_text += f'KS p-value = {gamma_fit_pos["ks_p_value"]:.4f}\n'

                    # Add KL divergence information for Gamma
                    if index_name in kl_results and not np.isnan(kl_results[index_name]['gamma_kl_pos_neg']):
                        stats_text += f'KL(Pos||Neg) = {kl_results[index_name]["gamma_kl_pos_neg"]:.4f}'

                    ax1.text(0.65, 0.95, stats_text, transform=ax1.transAxes, fontsize=8,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='none', 
                            edgecolor='black', linewidth=1, alpha=1.0))

            ax1.set_yscale('log')
            ax1.set_xlabel('VReturns', fontsize=12)
            ax1.set_ylabel('Frequency (log)', fontsize=12)
            ax1.set_title(f'{index_name} - Positive VReturns')
            ax1.legend(loc='lower left')
            ax1.grid(True, alpha=0.3)

            # TOP RIGHT: Negative VReturns with Gamma fit only (data within gamma range)
            if len(negative_VReturns) > 0 and gamma_fit_neg:
                negative_abs_for_plot = np.abs(negative_VReturns)

                # Calculate n_neg for the total negative data
                n_neg = len(negative_abs_for_plot)
                
                # Filter data to gamma fit range only
                gamma_lower = gamma_fit_neg['fit_lower_bound']
                gamma_upper = gamma_fit_neg['fit_upper_bound']
                
                # Special manual cut for Nikkei negative VReturns at 0.06 (Gamma part: <= 0.06)
                if index_name == 'Nikkei':
                    manual_cut = 0.06
                    gamma_data_neg = negative_abs_for_plot[(negative_abs_for_plot >= gamma_lower) & (negative_abs_for_plot <= manual_cut)]
                else:
                    gamma_data_neg = negative_abs_for_plot[(negative_abs_for_plot >= gamma_lower) & (negative_abs_for_plot <= gamma_upper)]
                
                if len(gamma_data_neg) > 0:
                    # Calculate histogram for gamma data only
                    hist_counts, bin_edges = np.histogram(gamma_data_neg, bins=30, density=False)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Create scatter plot
                    ax2.scatter(bin_centers, hist_counts, alpha=0.7, 
                               color=colors.get(index_name, '#000000'), 
                               s=30, label=f'{index_name} Negative')
                    
                    # Plot Gamma fit - only in optimal range, scaled for frequency
                    # Special handling for Nikkei to limit fit line to manual cut
                    if index_name == 'Nikkei':
                        fit_mask = (gamma_fit_neg['x_fit'] >= gamma_lower) & (gamma_fit_neg['x_fit'] <= manual_cut)
                    else:
                        fit_mask = (gamma_fit_neg['x_fit'] >= gamma_lower) & (gamma_fit_neg['x_fit'] <= gamma_upper)
                    # Scale PDF to frequency: multiply by bin width and total count
                    bin_width = (bin_edges[1] - bin_edges[0])
                    scaled_y_fit = gamma_fit_neg['y_fit'][fit_mask] * bin_width * len(gamma_data_neg)
                    ax2.plot(gamma_fit_neg['x_fit'][fit_mask], scaled_y_fit, 
                            'r-', linewidth=2, label=f'Gamma Fit')
                    
                    # Statistics for gamma data
                    mean_gamma_neg = np.mean(gamma_data_neg)
                    median_gamma_neg = np.median(gamma_data_neg)
                    std_gamma_neg = np.std(gamma_data_neg, ddof=1)
                    skew_gamma_neg = stats.skew(gamma_data_neg)
                    kurt_gamma_neg = stats.kurtosis(gamma_data_neg, fisher=True)
                    n_gamma_neg = len(gamma_data_neg)
                    
                    # For Nikkei, recalculate gamma points using manual cut to match exclusion calculation
                    if index_name == 'Nikkei':
                        manual_cut = 0.06
                        gamma_points_neg = np.sum((negative_abs_for_plot >= gamma_fit_neg["fit_lower_bound"]) & (negative_abs_for_plot <= manual_cut))
                    else:
                        gamma_points_neg = gamma_fit_neg["n_samples_used_for_fit"]
                    
                    # Calculate standard errors
                    mean_se_gamma_neg = std_gamma_neg / np.sqrt(n_gamma_neg)
                    skew_se_gamma_neg = np.sqrt(6 * n_gamma_neg * (n_gamma_neg - 1) / ((n_gamma_neg - 2) * (n_gamma_neg + 1) * (n_gamma_neg + 3)))
                    kurt_se_gamma_neg = np.sqrt(24 * n_gamma_neg * (n_gamma_neg - 1)**2 / ((n_gamma_neg - 3) * (n_gamma_neg - 2) * (n_gamma_neg + 3) * (n_gamma_neg + 5)))
                    
                    stats_text = f'n = {gamma_points_neg}\n'
                    stats_text += f'Mean = {mean_gamma_neg:.4f} ± {mean_se_gamma_neg:.4f}\n'
                    stats_text += f'Median = {median_gamma_neg:.4f}\n'
                    stats_text += f'Std = {std_gamma_neg:.4f}\n'
                    stats_text += f'Skewness = {float(skew_gamma_neg):.4f} ± {skew_se_gamma_neg:.4f}\n'
                    stats_text += f'Kurtosis = {float(kurt_gamma_neg):.4f} ± {kurt_se_gamma_neg:.4f}\n'
                    stats_text += f'\n'
                    stats_text += f'Gamma Fit:\n'
                    # Display Shape and Scale with standard errors
                    if not np.isnan(gamma_fit_neg.get("shape_se", np.nan)):
                        stats_text += f'Shape (α) = {gamma_fit_neg["shape"]:.4f} ± {gamma_fit_neg["shape_se"]:.4f}\n'
                    else:
                        stats_text += f'Shape (α) = {gamma_fit_neg["shape"]:.4f}\n'
                    
                    if not np.isnan(gamma_fit_neg.get("scale_se", np.nan)):
                        stats_text += f'Scale (θ) = {gamma_fit_neg["scale"]:.4f} ± {gamma_fit_neg["scale_se"]:.4f}\n'
                    else:
                        stats_text += f'Scale (θ) = {gamma_fit_neg["scale"]:.4f}\n'
                    
                    # Display Rate parameter with standard error if available
                    if not np.isnan(gamma_fit_neg.get("rate", np.nan)) and not np.isnan(gamma_fit_neg.get("rate_se", np.nan)):
                        stats_text += f'Rate (β) = {gamma_fit_neg["rate"]:.4f} ± {gamma_fit_neg["rate_se"]:.4f}\n'
                    else:
                        stats_text += f'Rate (β) = {1/gamma_fit_neg["scale"]:.4f}\n'
                    stats_text += f'\n'
                    stats_text += f'Fit Range = {gamma_fit_neg["optimal_lower_percentile"]:.1f}%-{gamma_fit_neg["optimal_upper_percentile"]:.1f}%\n'
                    
                    # For Nikkei, recalculate gamma points using manual cut to match exclusion calculation
                    if index_name == 'Nikkei':
                        manual_cut = 0.06
                        gamma_points_neg = np.sum((negative_abs_for_plot >= gamma_fit_neg["fit_lower_bound"]) & (negative_abs_for_plot <= manual_cut))
                    else:
                        gamma_points_neg = gamma_fit_neg["n_samples_used_for_fit"]
                    
                    stats_text += f'Fit n = {gamma_points_neg}\n'
                    # Calculate excluded points (both left and right tails)
                    excluded_left_count_neg = np.sum(negative_abs_for_plot < gamma_fit_neg["fit_lower_bound"])
                    
                    # Special manual cut for Nikkei negative VReturns at 0.04 (same as Gamma fit)
                    if index_name == 'Nikkei':
                        manual_cut = 0.04
                        excluded_right_count_neg = np.sum(negative_abs_for_plot > manual_cut)
                    else:
                        excluded_right_count_neg = np.sum(negative_abs_for_plot > gamma_fit_neg["fit_upper_bound"])
                    
                    total_excluded_neg = excluded_left_count_neg + excluded_right_count_neg
                    
                    stats_text += f'n exc = {total_excluded_neg} (L {excluded_left_count_neg}, R {excluded_right_count_neg})\n'
                    stats_text += f'\n'
                    stats_text += f'KS p-value = {gamma_fit_neg["ks_p_value"]:.4f}\n'
                    
                    # Add KL divergence information for Gamma
                    if index_name in kl_results and not np.isnan(kl_results[index_name]['gamma_kl_neg_pos']):
                        stats_text += f'KL(Neg||Pos) = {kl_results[index_name]["gamma_kl_neg_pos"]:.4f}'
                    
                    ax2.text(0.65, 0.95, stats_text, transform=ax2.transAxes, fontsize=8,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='none', 
                            edgecolor='black', linewidth=1, alpha=1.0))
                
                ax2.set_yscale('log')
                ax2.set_xlabel('|VReturns|', fontsize=12)
                ax2.set_ylabel('Frequency (log)', fontsize=12)
                ax2.set_title(f'{index_name} - Negative VReturns')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # BOTTOM LEFT: Positive VReturns excluded from Gamma with Pareto fit
            if len(positive_VReturns) > 0 and gamma_fit_pos and pareto_fit_pos:
                # First, exclude the left 5% of data completely
                left_5_percentile_pos = np.percentile(positive_VReturns, 5)
                data_without_left_5_pos = positive_VReturns[positive_VReturns >= left_5_percentile_pos]
                
                # Filter data to excluded range (only right tail for Pareto)
                gamma_lower = gamma_fit_pos['fit_lower_bound']
                gamma_upper = gamma_fit_pos['fit_upper_bound']
                excluded_data_pos = data_without_left_5_pos[data_without_left_5_pos > gamma_upper]
                
                if len(excluded_data_pos) > 0:
                    # Calculate histogram for excluded data only
                    hist_counts, bin_edges = np.histogram(excluded_data_pos, bins=30, density=False)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Create scatter plot
                    ax3.scatter(bin_centers, hist_counts, alpha=0.7, 
                               color=colors.get(index_name, '#000000'), 
                               s=30, label=f'{index_name} Positive')
                    
                    # Plot Pareto fit, scaled for frequency
                    # Scale PDF to frequency: multiply by bin width and total count
                    bin_width = (bin_edges[1] - bin_edges[0])
                    scaled_pareto_y = pareto_fit_pos['y_vis_original'] * bin_width * len(excluded_data_pos)
                    ax3.plot(pareto_fit_pos['x_vis_original'], scaled_pareto_y, 
                            'b-', linewidth=2, alpha=0.8, label=f'Pareto Fit')
                    
                    # Statistics for excluded data
                    mean_excl_pos = np.mean(excluded_data_pos)
                    median_excl_pos = np.median(excluded_data_pos)
                    std_excl_pos = np.std(excluded_data_pos, ddof=1)
                    skew_excl_pos = stats.skew(excluded_data_pos)
                    kurt_excl_pos = stats.kurtosis(excluded_data_pos, fisher=True)
                    n_excl_pos = len(excluded_data_pos)
                    
                    # Calculate standard errors
                    mean_se_excl_pos = std_excl_pos / np.sqrt(n_excl_pos)
                    skew_se_excl_pos = np.sqrt(6 * n_excl_pos * (n_excl_pos - 1) / ((n_excl_pos - 2) * (n_excl_pos + 1) * (n_excl_pos + 3)))
                    kurt_se_excl_pos = np.sqrt(24 * n_excl_pos * (n_excl_pos - 1)**2 / ((n_excl_pos - 3) * (n_excl_pos - 2) * (n_excl_pos + 3) * (n_excl_pos + 5)))
                    
                    stats_text = f'n = {n_excl_pos}\n'
                    stats_text += f'Mean = {mean_excl_pos:.4f} ± {mean_se_excl_pos:.4f}\n'
                    stats_text += f'Median = {median_excl_pos:.4f}\n'
                    stats_text += f'Std = {std_excl_pos:.4f}\n'
                    stats_text += f'Skewness = {float(skew_excl_pos):.4f} ± {float(skew_se_excl_pos):.4f}\n'
                    stats_text += f'Kurtosis = {float(kurt_excl_pos):.4f} ± {float(kurt_se_excl_pos):.4f}\n'
                    stats_text += f'\n'
                    stats_text += f'Pareto Fit:\n'
                    # Display Shape parameter with standard error
                    if not np.isnan(pareto_fit_pos.get("b_param_se", np.nan)):
                        stats_text += f'Shape (α) = {pareto_fit_pos["b_param"]:.4f} ± {pareto_fit_pos["b_param_se"]:.4f}\n'
                    else:
                        stats_text += f'Shape (α) = {pareto_fit_pos["b_param"]:.4f}\n'
                    stats_text += f'\n'
                    stats_text += f'Fit n = {pareto_fit_pos["pareto_fit_n"]}\n'
                    stats_text += f'\n'
                    stats_text += f'KS p-value = {pareto_fit_pos["ks_p_value"]:.4f}\n'
                    
                    # Add KL divergence information for Pareto
                    if index_name in kl_results and not np.isnan(kl_results[index_name]['pareto_kl_pos_neg']):
                        stats_text += f'KL(Pos||Neg) = {kl_results[index_name]["pareto_kl_pos_neg"]:.4f}'
                    
                    ax3.text(0.65, 0.95, stats_text, transform=ax3.transAxes, fontsize=8,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='none', 
                            edgecolor='black', linewidth=1, alpha=1.0))
                
                ax3.set_yscale('log')
                ax3.set_xlabel('VReturns', fontsize=12)
                ax3.set_ylabel('Frequency (log)', fontsize=12)
                ax3.set_title(f'{index_name} - Positive VReturns')
                ax3.legend(loc='lower left')
                ax3.grid(True, alpha=0.3)
            
            # BOTTOM RIGHT: Negative VReturns excluded from Gamma with Pareto fit
            if len(negative_VReturns) > 0 and gamma_fit_neg and pareto_fit_neg:
                negative_abs_for_plot = np.abs(negative_VReturns)
                
                # First, exclude the left 5% of data completely
                left_5_percentile_neg = np.percentile(negative_abs_for_plot, 5)
                data_without_left_5_neg = negative_abs_for_plot[negative_abs_for_plot >= left_5_percentile_neg]
                
                # Filter data to excluded range (only right tail for Pareto)
                gamma_lower = gamma_fit_neg['fit_lower_bound']
                gamma_upper = gamma_fit_neg['fit_upper_bound']
                
                # Special manual cut for Nikkei negative VReturns at 0.06
                if index_name == 'Nikkei':
                    manual_cut = 0.06
                    excluded_data_neg = data_without_left_5_neg[data_without_left_5_neg > manual_cut]
                else:
                    excluded_data_neg = data_without_left_5_neg[data_without_left_5_neg > gamma_upper]
                
                if len(excluded_data_neg) > 0:
                    # Calculate histogram for excluded data only
                    hist_counts, bin_edges = np.histogram(excluded_data_neg, bins=30, density=False)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Create scatter plot
                    ax4.scatter(bin_centers, hist_counts, alpha=0.7, 
                               color=colors.get(index_name, '#000000'), 
                               s=30, label=f'{index_name} Negative')
                    
                    # Plot Pareto fit, scaled for frequency
                    # Scale PDF to frequency: multiply by bin width and total count
                    bin_width = (bin_edges[1] - bin_edges[0])
                    scaled_pareto_y = pareto_fit_neg['y_vis_original'] * bin_width * len(excluded_data_neg)
                    ax4.plot(pareto_fit_neg['x_vis_original'], scaled_pareto_y, 
                            'b-', linewidth=2, alpha=0.8, label=f'Pareto Fit')
                    
                    # Statistics for excluded data
                    mean_excl_neg = np.mean(excluded_data_neg)
                    median_excl_neg = np.median(excluded_data_neg)
                    std_excl_neg = np.std(excluded_data_neg, ddof=1)
                    skew_excl_neg = stats.skew(excluded_data_neg)
                    kurt_excl_neg = stats.kurtosis(excluded_data_neg, fisher=True)
                    n_excl_neg = len(excluded_data_neg)
                    
                    # Calculate standard errors
                    mean_se_excl_neg = std_excl_neg / np.sqrt(n_excl_neg)
                    skew_se_excl_neg = np.sqrt(6 * n_excl_neg * (n_excl_neg - 1) / ((n_excl_neg - 2) * (n_excl_neg + 1) * (n_excl_neg + 3)))
                    kurt_se_excl_neg = np.sqrt(24 * n_excl_neg * (n_excl_neg - 1)**2 / ((n_excl_neg - 3) * (n_excl_neg - 2) * (n_excl_neg + 3) * (n_excl_neg + 5)))
                    
                    stats_text = f'n = {n_excl_neg}\n'
                    stats_text += f'Mean = {mean_excl_neg:.4f} ± {mean_se_excl_neg:.4f}\n'
                    stats_text += f'Median = {median_excl_neg:.4f}\n'
                    stats_text += f'Std = {std_excl_neg:.4f}\n'
                    stats_text += f'Skewness = {float(skew_excl_neg):.4f} ± {float(skew_se_excl_neg):.4f}\n'
                    stats_text += f'Kurtosis = {float(kurt_excl_neg):.4f} ± {float(kurt_se_excl_neg):.4f}\n'
                    stats_text += f'\n'
                    stats_text += f'Pareto Fit:\n'
                    # Display Shape parameter with standard error
                    if not np.isnan(pareto_fit_neg.get("b_param_se", np.nan)):
                        stats_text += f'Shape (α) = {pareto_fit_neg["b_param"]:.4f} ± {pareto_fit_neg["b_param_se"]:.4f}\n'
                    else:
                        stats_text += f'Shape (α) = {pareto_fit_neg["b_param"]:.4f}\n'
                    stats_text += f'\n'
                    stats_text += f'Fit n = {pareto_fit_neg["pareto_fit_n"]}\n'
                    stats_text += f'\n'
                    stats_text += f'KS p-value = {pareto_fit_neg["ks_p_value"]:.4f}\n'
                    
                    # Add KL divergence information for Pareto
                    if index_name in kl_results and not np.isnan(kl_results[index_name]['pareto_kl_neg_pos']):
                        stats_text += f'KL(Neg||Pos) = {kl_results[index_name]["pareto_kl_neg_pos"]:.4f}'
                    
                    ax4.text(0.65, 0.95, stats_text, transform=ax4.transAxes, fontsize=8,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='none', 
                            edgecolor='black', linewidth=1, alpha=1.0))
                
                ax4.set_yscale('log')
                ax4.set_xlabel('|VReturns|', fontsize=12)
                ax4.set_ylabel('Frequency (log)', fontsize=12)
                ax4.set_title(f'{index_name} - Negative VReturns')
                ax4.legend(loc='lower left')
                ax4.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save individual plot
            plot_filename = os.path.join(hist_save_path, f'VReturns_histogram_4_subplots_gamma_pareto_{index_name.replace(" ", "_").replace("/", "_")}.png')
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            plot_filenames.append(plot_filename)
            print(f"VReturns 4-subplot histogram plot with Gamma/Pareto separation for {index_name} saved: {plot_filename}")
    
    print("\nAll individual VReturns 4-subplot histogram plots with Gamma/Pareto separation completed.")
    print("=" * 50)
    return plot_filenames, kl_results, gamma_results

# --- Main Function ---

def main():
    """Main function to execute the VReturns analysis with Gamma fits."""
    # Load financial data
    all_prices_data, all_prices_raw = download_financial_data()
    
    if not all_prices_data:
        return
    
    # Filter data by period with index-specific start dates
    all_prices_data, all_prices_raw = filter_data_by_period(all_prices_data, all_prices_raw)
    
    # Extract VReturns for all indices
    VReturns_by_index = extract_VReturns_by_index(all_prices_data)
    
    # Generate VReturns histogram plots with 4 subplots (Gamma + Pareto separated)
    plot_filenames, kl_results, gamma_results = plot_VReturns_histogram_4_subplots_separated(VReturns_by_index)

if __name__ == "__main__":
    main()