import numpy as np
from ..core.simulation import BrownianMotion

class SimulationResults:
    """Class for managing multiple simulation runs and analyzing results."""
    
    def __init__(self, num_simulations, T, N, x0, v0, L, p, gamma, m, sigma):
        """Initialize simulation parameters and storage."""
        self.num_simulations = num_simulations
        self.brownian_motion = BrownianMotion(T, N, x0, v0, L, p, gamma, m, sigma)
        self.x_euler_all = []
        self.v_euler_all = []
        self.x_rk4_all = []
        self.v_rk4_all = []
    def run_all_simulations(self):
        """Run multiple simulations using both Euler and RK4 methods."""
        for _ in range(self.num_simulations):
            x_euler, v_euler = self.brownian_motion.run_simulation(method='euler')
            x_rk4, v_rk4 = self.brownian_motion.run_simulation(method='rk4')
            self.x_euler_all.append(x_euler)
            self.v_euler_all.append(v_euler)
            self.x_rk4_all.append(x_rk4)
            self.v_rk4_all.append(v_rk4)
    def calculate_statistics(self):
        """Calculate means and standard deviations for all simulations."""
        x_euler_mean = np.mean(self.x_euler_all, axis=0)
        v_euler_mean = np.mean(self.v_euler_all, axis=0)
        x_rk4_mean = np.mean(self.x_rk4_all, axis=0)
        v_rk4_mean = np.mean(self.v_rk4_all, axis=0)

        x_euler_std = np.std(self.x_euler_all, axis=0)
        v_euler_std = np.std(self.v_euler_all, axis=0)
        x_rk4_std = np.std(self.x_rk4_all, axis=0)
        v_rk4_std = np.std(self.v_rk4_all, axis=0)

        return (x_euler_mean, v_euler_mean, x_rk4_mean, v_rk4_mean,
                x_euler_std, v_euler_std, x_rk4_std, v_rk4_std)
    def calculate_numerical_error(self):
        """Calculate numerical error between Euler and RK4 methods."""
        x_euler_mean = np.mean(self.x_euler_all, axis=0)
        x_rk4_mean = np.mean(self.x_rk4_all, axis=0)
        return np.abs(x_euler_mean - x_rk4_mean)

    def calculate_statistical_error(self):
        """Calculate statistical error in Euler method."""
        return np.std(self.x_euler_all, axis=0)

    def calculate_finite_precision_error(self, precisions):
        """Calculate errors due to finite precision arithmetic."""
        x_results = []
        for precision in precisions:
            x, _ = self.brownian_motion.run_simulation(method='euler')
            x_results.append(x.astype(np.float64))
        
        errors = []
        for i in range(len(precisions) - 1):
            diff = np.abs(x_results[i] - x_results[-1])
            errors.append(diff)
        return errors
