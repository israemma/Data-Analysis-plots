import pandas as pd
import numpy as np
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

def load_djia_data():
    """
    Load DJIA data from CSV file
    """
    print("=== Loading DJIA Data ===")
    
    # Load data from CSV
    djia_data = pd.read_csv('csv_data/DJIA_close_prices.csv')
    djia_data['Date'] = pd.to_datetime(djia_data['Date'])
    djia_data.set_index('Date', inplace=True)
    
    print(f"Data loaded: {len(djia_data)} observations")
    print(f"Period: {djia_data.index[0]} to {djia_data.index[-1]}")
    print(f"Initial price: ${djia_data['Close'].iloc[0]:.2f}")
    print(f"Final price: ${djia_data['Close'].iloc[-1]:.2f}")
    
    return djia_data

def calculate_returns(prices):
    """
    Calculate logarithmic returns
    """
    print("\n=== Calculating Logarithmic Returns ===")
    
    returns = np.log(prices['Close'] / prices['Close'].shift(1)).dropna()
    
    print(f"Returns calculated: {len(returns)} observations")
    print(f"Mean: {returns.mean():.6f}")
    print(f"Standard deviation: {returns.std():.6f}")
    print(f"Minimum: {returns.min():.6f}")
    print(f"Maximum: {returns.max():.6f}")
    
    return returns

def calibrate_garch_model(returns):
    """
    Calibrate GARCH(1,1) model using real data
    """
    print("\n=== Calibrating GARCH(1,1) Model ===")
    
    # Convert returns to percentage for better convergence
    returns_pct = returns * 100
    
    # Create and fit GARCH(1,1) model
    model = arch_model(returns_pct, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
    
    print("Fitting model...")
    res = model.fit(disp='off')
    
    print("\n=== Calibration Results ===")
    print(res.summary())
    
    # Extract parameters
    params = res.params
    
    # Parameters are in percentage scale, convert back
    mu = params['mu'] / 100  # Mean
    omega = params['omega'] / (100**2)  # Constant term
    alpha = params['alpha[1]']  # ARCH coefficient
    beta = params['beta[1]']   # GARCH coefficient
    
    print(f"\n=== Calibrated Parameters (Original Scale) ===")
    print(f"mu (μ): {mu:.8f}")
    print(f"omega (ω): {omega:.8e}")
    print(f"alpha (α): {alpha:.6f}")
    print(f"beta (β): {beta:.6f}")
    print(f"α + β = {alpha + beta:.6f} (must be < 1 for stationarity)")
    
    # Model statistics
    print(f"\n=== Model Statistics ===")
    print(f"Log-likelihood: {res.loglikelihood:.2f}")
    print(f"AIC: {res.aic:.2f}")
    print(f"BIC: {res.bic:.2f}")
    
    return {
        'mu': mu,
        'omega': omega,
        'alpha': alpha,
        'beta': beta,
        'model_results': res
    }

def main():
    """
    Main function to calibrate GARCH with DJIA data
    """
    try:
        # Load data
        djia_data = load_djia_data()
        
        # Calculate returns
        returns = calculate_returns(djia_data)
        
        # Calibrate GARCH model
        garch_params = calibrate_garch_model(returns)
        
        print(f"\n=== Final Parameters for Simulation ===")
        print(f"omega = {garch_params['omega']:.8e}")
        print(f"alpha = {garch_params['alpha']:.6f}")
        print(f"beta = {garch_params['beta']:.6f}")
        print(f"mu = {garch_params['mu']:.8f}")
        
        return garch_params
        
    except Exception as e:
        print(f"Error in calibration: {e}")
        return None

if __name__ == "__main__":
    calibrated_params = main()