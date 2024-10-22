import numpy as np

class ProbabilitySampler:
    """Class for generating samples from different probability distributions."""
    
    @staticmethod
    def sample_inverse_cdf(size=1):
        """Generate samples using inverse CDF method."""
        lam = 1.0
        u = np.random.uniform(0, 1, size)
        samples = -np.log(1 - u) / lam
        return samples

    @staticmethod
    def sample_rejection(size=1):
        """Generate samples using rejection sampling method."""
        samples = []
        while len(samples) < size:
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(0, 0.4)
            if y < (1/np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2):
                samples.append(x)
        return np.array(samples)
