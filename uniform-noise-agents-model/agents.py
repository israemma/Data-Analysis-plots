import numpy as np

class UniformNoiseAgent:
    """
    An agent that makes trading decisions randomly.
    The duration of each position (long or short) is determined by a
    uniform distribution.
    """
    def __init__(self, max_holding_period=14):
        """
        Initializes the uniform noise agent.

        Args:
            max_holding_period (int): Maximum holding period.
        """
        self.type = 'uniform_noise'
        self.max_holding_period = max_holding_period

        self.position = np.random.choice([-1, 1])
        self.steps_remaining = self._get_new_duration()

    def _get_new_duration(self):
        """
        Calculates the duration for the next position using a uniform distribution.
        """
        duration = np.random.randint(1, self.max_holding_period + 1)
        return duration

    def decide(self, **kwargs):
        """
        Makes a trading decision.

        Returns:
            int: 0 if holding, or the new position (1 or -1) if it changes.
        """
        self.steps_remaining -= 1

        if self.steps_remaining > 0:
            return 0
        else:
            self.position *= -1
            self.steps_remaining = self._get_new_duration()
            return self.position