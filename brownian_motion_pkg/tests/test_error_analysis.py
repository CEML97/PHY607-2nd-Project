import unittest
import numpy as np
from brownian_motion_sim.analysis.error_analysis import SimulationResults

class TestErrorAnalysis(unittest.TestCase):
    def setUp(self):
        self.sim_results = SimulationResults(num_simulations=5, T=10, N=100, x0=0, v0=1, L=10, p=0.1, gamma=0.5, m=1, sigma=0.2)
        self.sim_results.run_all_simulations()

    def test_calculate_statistics(self):
        stats = self.sim_results.calculate_statistics()
        # Ensure correct output structure
        self.assertEqual(len(stats), 8)
        # Check that the means and standard deviations are numpy arrays of the correct length
        self.assertEqual(stats[0].shape[0], self.sim_results.brownian_motion.N)  # x_euler_mean length
        self.assertEqual(stats[4].shape[0], self.sim_results.brownian_motion.N)  # x_euler_std length

    def test_calculate_numerical_error(self):
        error = self.sim_results.calculate_numerical_error()
        # Ensure the output is a numpy array of the correct length
        self.assertEqual(error.shape[0], self.sim_results.brownian_motion.N)
        # Numerical error should not be negative
        self.assertTrue(np.all(error >= 0))

    def test_calculate_statistical_error(self):
        stat_error = self.sim_results.calculate_statistical_error()
        # Ensure the output is a numpy array of the correct length
        self.assertEqual(stat_error.shape[0], self.sim_results.brownian_motion.N)
        # Statistical error should not be negative
        self.assertTrue(np.all(stat_error >= 0))

    def test_calculate_finite_precision_error(self):
        precisions = [32, 64]
        precision_errors = self.sim_results.calculate_finite_precision_error(precisions)
        # Ensure we have the correct number of precision errors
        self.assertEqual(len(precision_errors), len(precisions) - 1)
        for error in precision_errors:
            # Ensure each error is a numpy array of the correct length
            self.assertEqual(error.shape[0], self.sim_results.brownian_motion.N)

if __name__ == '__main__':
    unittest.main()
