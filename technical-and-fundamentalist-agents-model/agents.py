import numpy as np
import pandas as pd
from collections import deque


class MovingAverageAgent:
    """Technical agent using moving average crossover strategy."""

    def __init__(self, short_ma_range=(10, 50), long_ma_range=(60, 300)):
        """Initialize technical agent with moving average parameters."""
        self.type = 'technical'
        
        self.short_ma_period = np.random.randint(short_ma_range[0], short_ma_range[1] + 1)
        self.long_ma_period = np.random.randint(long_ma_range[0], long_ma_range[1] + 1)
        
        if self.short_ma_period >= self.long_ma_period:
            self.short_ma_period = short_ma_range[0]
        self.position = 0
        
        # Buffers for incremental moving average calculation
        self._short_ma_buffer = deque(maxlen=self.short_ma_period)
        self._long_ma_buffer = deque(maxlen=self.long_ma_period)
        self._short_ma_sum = 0.0
        self._long_ma_sum = 0.0
        self._initialized = False

    def decide(self, historical_prices, **kwargs):
        """Generate trading decision based on moving average crossover."""
        # Extract latest price
        if isinstance(historical_prices, pd.Series):
            if len(historical_prices) == 0:
                return 0
            latest_price = historical_prices.iloc[-1]
        else:
            if len(historical_prices) == 0:
                return 0
            latest_price = historical_prices[-1]
        
        # Update short moving average buffer
        if len(self._short_ma_buffer) == self.short_ma_period:
            self._short_ma_sum -= self._short_ma_buffer[0]
        self._short_ma_buffer.append(latest_price)
        self._short_ma_sum += latest_price
        
        # Update long moving average buffer
        if len(self._long_ma_buffer) == self.long_ma_period:
            self._long_ma_sum -= self._long_ma_buffer[0]
        self._long_ma_buffer.append(latest_price)
        self._long_ma_sum += latest_price
        
        # Require sufficient data for long moving average
        if len(self._long_ma_buffer) < self.long_ma_period:
            return 0
        
        # Calculate moving averages
        short_ma = self._short_ma_sum / len(self._short_ma_buffer)
        long_ma = self._long_ma_sum / len(self._long_ma_buffer)
        
        # Generate trading signal
        if short_ma > long_ma and self.position <= 0:
            self.position = 1
            return 1
        elif short_ma < long_ma and self.position >= 0:
            self.position = -1
            return -1
        else:
            return 0


class FundamentalAgent:
    """Fundamental agent comparing market price to fundamental value."""

    def __init__(self, threshold=0.05):
        """Initialize fundamental agent with deviation threshold.
        
        Args:
            threshold (float): Percentage deviation from fundamental value to trigger action.
        """
        self.type = 'fundamental'
        self.threshold = threshold
        self.position = 0

    def decide(self, market_price, fundamental_price, **kwargs):
        """Generate trading decision based on price deviation from fundamental value."""
        positive_deviation = fundamental_price * (1 + self.threshold)
        negative_deviation = fundamental_price * (1 - self.threshold)

        # Buy signal when market price is below fundamental value
        if market_price < negative_deviation and self.position <= 0:
            self.position = 1
            return 1

        # Sell signal when market price is above fundamental value
        elif market_price > positive_deviation and self.position >= 0:
            self.position = -1
            return -1

        # Close position when price returns to fundamental value range
        elif negative_deviation <= market_price <= positive_deviation:
            if self.position == -1: 
                self.position = 0
                return 1
            elif self.position == 1:
                self.position = 0
                return -1
            
        # Hold current position
        return 0
