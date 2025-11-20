import numpy as np
import pandas as pd
import os
from datetime import datetime

class GARCHSimulator:
    """
    GARCH(1,1) model simulator for generating synthetic financial price series
    """
    
    def __init__(self, omega=0.00001, alpha=0.1, beta=0.85, mu=0.0005, initial_price=100.0):
        """
        Initialize the GARCH(1,1) simulator
        
        Parameters:
        - omega: volatility constant term
        - alpha: ARCH coefficient (impact of past shocks)
        - beta: GARCH coefficient (volatility persistence)
        - mu: returns mean
        - initial_price: initial price of the series
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.initial_price = initial_price
        
        # Check stationarity condition
        if alpha + beta >= 1:
            print(f"Warning: α + β = {alpha + beta:.6f} >= 1. The model may not be stationary.")
        else:
            print(f"Stationary GARCH model: α + β = {alpha + beta:.6f} < 1")
    
    def simulate_returns(self, n_periods=1000, initial_variance=None, initial_shock=None):
        """
        Simulate returns using the GARCH(1,1) model
        
        Model:
        r_t = μ + ε_t
        ε_t = σ_t * z_t, where z_t ~ N(0,1)
        σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
        """
        # Initialize arrays
        returns = np.zeros(n_periods)
        variances = np.zeros(n_periods)
        shocks = np.zeros(n_periods)
        
        # Initial values
        if initial_variance is None:
            # Unconditional variance: σ² = ω / (1 - α - β)
            initial_variance = self.omega / (1 - self.alpha - self.beta)
        
        if initial_shock is None:
            initial_shock = np.random.normal(0, np.sqrt(initial_variance))
        
        variances[0] = initial_variance
        shocks[0] = initial_shock
        returns[0] = self.mu + shocks[0]
        
        # Generate time series
        for t in range(1, n_periods):
            # Update conditional variance
            variances[t] = (self.omega + 
                          self.alpha * shocks[t-1]**2 + 
                          self.beta * variances[t-1])
            
            # Generate random shock
            shocks[t] = np.random.normal(0, np.sqrt(variances[t]))
            
            # Calculate return
            returns[t] = self.mu + shocks[t]
        
        return returns, variances, shocks
    
    def construct_price_series(self, returns):
        """
        Construct price series from logarithmic returns
        P_t = P_{t-1} * exp(r_t)
        """
        prices = np.zeros(len(returns) + 1)
        prices[0] = self.initial_price
        
        for t in range(len(returns)):
            prices[t+1] = prices[t] * np.exp(returns[t])
        
        return prices

def generate_garch_prices(n_periods=4000, save_csv=True, csv_filename='garch_prices.csv'):
    """
    Generate GARCH price series and save it to CSV
    """
    print("=== GARCH(1,1) Price Series Generator ===")
    print(f"Generating {n_periods} periods of synthetic data\n")
    
    # Parameters calibrated with real DJIA data (1992-2024)
    # Estimated using Maximum Likelihood with arch_model
    omega = 1.95340182e-06  # Base volatility constant term
    alpha = 0.114017        # Impact of past shocks (α)
    beta = 0.869275         # Volatility persistence (β)
    mu = 0.00062019         # Calibrated daily returns mean
    initial_price = 100.0   # Initial price
    
    print("GARCH(1,1) model parameters:")
    print(f"  ω (omega) = {omega:.2e}")
    print(f"  α (alpha) = {alpha:.6f}")
    print(f"  β (beta)  = {beta:.6f}")
    print(f"  μ (mu)    = {mu:.6f}")
    print(f"  α + β     = {alpha + beta:.6f}")
    print(f"  Initial price = {initial_price}")
    
    # Create simulator
    simulator = GARCHSimulator(omega=omega, alpha=alpha, beta=beta, 
                              mu=mu, initial_price=initial_price)
    
    # Simulate returns
    print(f"\nSimulating {n_periods} returns...")
    returns, variances, shocks = simulator.simulate_returns(n_periods=n_periods)
    
    # Construct price series
    print("Constructing price series...")
    prices = simulator.construct_price_series(returns)
    
    # Simulation statistics
    print(f"\n=== Simulation Statistics ===")
    print(f"Returns - Mean: {np.mean(returns):.6f}, Std: {np.std(returns):.6f}")
    print(f"Variances - Mean: {np.mean(variances):.8f}, Std: {np.std(variances):.8f}")
    print(f"Prices - Initial: {prices[0]:.2f}, Final: {prices[-1]:.2f}")
    print(f"Total return: {(prices[-1]/prices[0] - 1)*100:.2f}%")
    print(f"Maximum price: {np.max(prices):.2f}")
    print(f"Minimum price: {np.min(prices):.2f}")
    
    if save_csv:
        # Create DataFrame with the data
        # Create synthetic dates (daily frequency)
        dates = pd.date_range(start='2000-01-01', periods=len(prices), freq='D')
        
        df = pd.DataFrame({
            'date': dates,
            'price': prices,
            'return': [np.nan] + list(returns),  # First return is NaN
            'variance': [np.nan] + list(variances),  # First variance is NaN
            'shock': [np.nan] + list(shocks)  # First shock is NaN
        })
        
        # Save to CSV
        df.to_csv(csv_filename, index=False)
        print(f"\n=== Data Saved ===")
        print(f"CSV file created: {csv_filename}")
        print(f"Number of records: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        
        # Show first rows
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        return df, csv_filename
    
    return prices, returns, variances, shocks

def main():
    """
    Main function to generate GARCH price series
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate price series
    n_periods = 4000  # Number of days to simulate
    csv_filename = 'garch_simulated_prices.csv'
    
    df, filename = generate_garch_prices(n_periods=n_periods, 
                                       save_csv=True, 
                                       csv_filename=csv_filename)
    
    print(f"\n=== Generation Completed ===")
    print(f"GARCH price series saved to: {filename}")
    print(f"Ready for trend analysis and plot generation.")

if __name__ == "__main__":
    main()