import numpy as np
import pandas as pd
from agents import MovingAverageAgent, FundamentalAgent

class Market:
    """
    Simulates the interaction of agents in a financial market to generate a
    price series, now including a fundamental price and agent-type demand.
    """
    def __init__(self, agents, market_params):
        """
        Initializes the market with agents and parameters.

        Args:
            agents (list): List of agent objects.
            market_params (dict): Dictionary of market parameters.
        """
        self.agents = agents
        self.initial_price = market_params.get('initial_price', 100.0)
        self.price_impact = market_params.get('price_impact', 0.001)
        self.noise_std = market_params.get('noise_std', 0.01)
        
        # Fundamental price parameters
        self.initial_fundamental_value = market_params.get('initial_fundamental_value', 100.0)
        self.long_term_mean = market_params.get('long_term_mean', 100.0)
        self.reversion_theta = market_params.get('reversion_theta', 0.001)
        self.fundamental_sigma = market_params.get('fundamental_sigma', 0.1)

        # Price history tracking
        self.market_prices = [self.initial_price]
        self.fundamental_prices = [self.initial_fundamental_value]
        self.agent_demands = []

    def simulate_step(self):
        """
        Execute one market simulation step.

        Updates fundamental price, collects agent decisions, and updates market price.
        """
        # Update fundamental price with mean reversion and random shock
        current_fundamental_price = self.fundamental_prices[-1]
        random_shock = np.random.normal(0, self.fundamental_sigma)
        new_fundamental_price = current_fundamental_price + self.reversion_theta * (self.long_term_mean - current_fundamental_price) + random_shock
        self.fundamental_prices.append(new_fundamental_price)

        # Collect agent decisions and aggregate demand
        historical_market_prices = pd.Series(self.market_prices)
        current_market_price = self.market_prices[-1]
        
        step_demands = {'fundamental': 0, 'technical': 0}
        for agent in self.agents:
            decision = agent.decide(
                historical_prices=historical_market_prices,
                market_price=current_market_price,
                fundamental_price=new_fundamental_price
            )
            if agent.type in step_demands:
                step_demands[agent.type] += decision
        
        self.agent_demands.append(step_demands)
        net_demand = step_demands['fundamental'] + step_demands['technical']

        # Update market price based on net demand and noise
        proportional_change = (self.price_impact * net_demand) + (np.random.normal(0, self.noise_std))
        new_market_price = current_market_price * (1 + proportional_change)
        
        self.market_prices.append(max(0.01, new_market_price))

    def run_simulation(self, num_steps):
        """
        Execute market simulation for specified number of steps.

        Args:
            num_steps (int): Number of simulation steps.

        Returns:
            pd.DataFrame: DataFrame containing market price and demand data.
        """
        for i in range(num_steps):
            self.simulate_step()
        
        # Compile price data
        df_prices = pd.DataFrame({
            'market_price': self.market_prices,
            'fundamental_price': self.fundamental_prices
        })
        
        # Compile demand data and align indices
        df_demands = pd.DataFrame(self.agent_demands)
        df_demands.index = df_prices.index[1:]
        
        # Merge price and demand data
        df_full = pd.concat([df_prices.iloc[1:].reset_index(drop=True), df_demands.reset_index(drop=True)], axis=1)
        
        return df_full


def generate_market_data(num_agents, num_steps, output_file, market_params):
    """
    Generate market simulation data and save to file.

    Args:
        num_agents (int): Number of agents in the market.
        num_steps (int): Number of simulation steps.
        output_file (str): File path to save the output data.
        market_params (dict): Dictionary of market parameters.
    """
    SHORT_MA_RANGE = (10, 50)
    LONG_MA_RANGE = (60, 300)
    
    fundamental_proportion = market_params.get('fundamental_proportion', 0.2)
    num_fundamental = int(num_agents * fundamental_proportion)
    num_technical = num_agents - num_fundamental

    # Initialize agent populations
    technical_agents = [MovingAverageAgent(SHORT_MA_RANGE, LONG_MA_RANGE) for _ in range(num_technical)]
    threshold_range = market_params.get('fundamental_threshold_range', (0.01, 0.05))
    fundamental_agents = [FundamentalAgent(threshold=np.random.uniform(threshold_range[0], threshold_range[1])) for _ in range(num_fundamental)]

    # Combine and shuffle agents
    all_agents = technical_agents + fundamental_agents
    np.random.shuffle(all_agents)

    market = Market(all_agents, market_params)

    df_results = market.run_simulation(num_steps=num_steps)
    df_results.to_csv(output_file, index_label="step")

if __name__ == "__main__":
    print("Running a test simulation from market_model.py")
    test_params = {
        'price_impact': 0.001, 
        'noise_std': 0.01,
        'fundamental_proportion': 0.2,
        'reversion_theta': 0.005,
        'fundamental_sigma': 0.15,
        'fundamental_threshold_range': (0.01, 0.20)
    }
    generate_market_data(
        num_agents=100,
        num_steps=2000,
        output_file='test_price_series.csv',
        market_params=test_params
    )