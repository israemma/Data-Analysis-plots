import numpy as np
import pandas as pd
from agents import UniformNoiseAgent

class Market:
    """
    Simulates the interaction of agents in a financial market to generate a
    price series. Only uniform noise agents operate.
    """
    def __init__(self, agents, market_params):
        """
        Initializes the market.

        Args:
            agents (list): List of agent objects that will operate in the market.
            market_params (dict): Dictionary with the market parameters.
        """
        self.agents = agents
        self.initial_price = market_params.get('initial_price', 100.0)
        self.price_impact = market_params.get('price_impact', 0.001)
        self.noise_std = market_params.get('noise_std', 0.01)

        self.market_prices = [self.initial_price]
        self.agent_demands = []

    def simulate_step(self):
        """Simulates a single time step in the market."""
        current_market_price = self.market_prices[-1]
        
        step_demands = {'uniform_noise': 0}
        for agent in self.agents:
            decision = agent.decide()
            step_demands[agent.type] += decision
        
        self.agent_demands.append(step_demands)
        net_demand = step_demands['uniform_noise']

        proportional_change = (self.price_impact * net_demand) + (np.random.normal(0, self.noise_std))
        new_market_price = current_market_price * (1 + proportional_change)
        
        self.market_prices.append(max(0.01, new_market_price))

    def run_simulation(self, num_steps):
        """Runs the complete simulation for a number of steps."""
        for i in range(num_steps):
            self.simulate_step()
        
        df_prices = pd.DataFrame({
            'market_price': self.market_prices[1:]
        })
        
        df_demands = pd.DataFrame(self.agent_demands)
        
        df_full = pd.concat([df_prices.reset_index(drop=True), df_demands.reset_index(drop=True)], axis=1)
        
        return df_full


def generate_market_data(num_agents, num_steps, output_file, market_params):
    """
    Sets up and runs the simulation with uniform noise agents and saves the data.
    """
    max_holding = market_params.get('max_holding_period', 14)

    all_agents = [UniformNoiseAgent(max_holding_period=max_holding) for _ in range(num_agents)]
    
    market = Market(all_agents, market_params)

    df_results = market.run_simulation(num_steps=num_steps)

    df_results.to_csv(output_file, index_label="step")

if __name__ == "__main__":
    print("Running a test simulation from market_model.py")
    test_params = {
        'price_impact': 0.001, 
        'noise_std': 0.01,
        'max_holding_period': 14,
    }
    generate_market_data(
        num_agents=100,
        num_steps=2000,
        output_file='test_price_series.csv',
        market_params=test_params
    )