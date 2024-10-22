import unittest
import numpy as np
from brownian_motion_sim.core.sampling import ProbabilitySampler

class TestProbabilitySampler(unittest.TestCase):
    def test_sample_inverse_cdf(self):
        size = 1000
        samples = ProbabilitySampler.sample_inverse_cdf(size=size)
        # Test if the samples have the expected size
        self.assertEqual(len(samples), size)
        # Test if the samples follow an exponential distribution (mean should be around 1/lambda, where lambda = 1)
        self.assertAlmostEqual(np.mean(samples), 1.0, delta=0.1)
        self.assertGreaterEqual(np.min(samples), 0)

    def test_sample_rejection(self):
        size = 1000
        samples = ProbabilitySampler.sample_rejection(size=size)
        # Test if the samples have the expected size
        self.assertEqual(len(samples), size)
        # Test if the samples follow approximately normal distribution (mean 0, std around 1)
        self.assertAlmostEqual(np.mean(samples), 0.0, delta=0.1)
        self.assertAlmostEqual(np.std(samples), 1.0, delta=0.1)

if __name__ == '__main__':
    unittest.main()
