import numpy as np


class Sensor:
    """Simulates noisy observations of the ball's position."""

    def __init__(self, noise_std):
        """
        Initialize the sensor with noise standard deviation.

        Args:
            noise_std (float): Standard deviation of the sensor noise
        """
        self.noise_std = noise_std

    def observe(self, true_position):
        """
        Generate a noisy observation of the true position.

        Args:
            true_position (np.ndarray): True position [x, y]

        Returns:
            np.ndarray: Noisy observation [x, y]
        """
        return true_position + np.random.normal(0, self.noise_std, size=2)
