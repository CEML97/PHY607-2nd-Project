import matplotlib.pyplot as plt
import numpy as np

class SimulationPlotter:
    """Class for visualizing simulation results and analysis."""
    
    @staticmethod
    def plot_simulation_results(time, stats, skip=1000):
        """Plot position and velocity results from both methods."""
        x_euler_mean, v_euler_mean, x_rk4_mean, v_rk4_mean, \
        x_euler_std, v_euler_std, x_rk4_std, v_rk4_std = stats
        
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 1, 1)
        plt.errorbar(time[::skip], x_euler_mean[::skip], yerr=x_euler_std[::skip], 
                    label='Euler', alpha=0.7)
        plt.errorbar(time[::skip], x_rk4_mean[::skip], yerr=x_rk4_std[::skip], 
                    label='RK4', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Position (x)')
        plt.title('Position vs Time: Euler and RK4 Methods')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.errorbar(time[::skip], v_euler_mean[::skip], yerr=v_euler_std[::skip], 
                    label='Euler', alpha=0.7)
        plt.errorbar(time[::skip], v_rk4_mean[::skip], yerr=v_rk4_std[::skip], 
                    label='RK4', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Velocity (v)')
        plt.title('Velocity vs Time: Euler and RK4 Methods')
        plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_error_analysis(time, numerical_error, statistical_error, finite_precision_errors):
        """Plot various types of errors from the simulation."""
        # Plot numerical error
        plt.figure(figsize=(12, 6))
        plt.plot(time, numerical_error, label='Numerical Error', color='orange')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.title('Numerical Error over Time')
        plt.legend()
        plt.show()

        # Plot statistical error
        plt.figure(figsize=(12, 6))
        plt.plot(time, statistical_error, label='Statistical Error', color='blue')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.title('Statistical Error over Time')
        plt.legend()
        plt.show()

        # Plot finite precision error
        plt.figure(figsize=(12, 6))
        for i, error in enumerate(finite_precision_errors):
            plt.plot(time, error, label=f'Precision Error (Type {i})')
        plt.xlabel('Time')
        plt.ylabel('Absolute Error')
        plt.title('Finite Precision Errors')
        plt.legend()
        plt.yscale('log')
        plt.show()

    @staticmethod
    def plot_probability_distributions(inverse_cdf_samples, rejection_samples):
        """Plot histograms of samples from different probability distributions."""
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(inverse_cdf_samples, bins=30, density=True, alpha=0.6, 
                color='b', label='Inverse CDF Samples')
        plt.title('Samples from Inverse CDF')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(rejection_samples, bins=30, density=True, alpha=0.6, 
                color='r', label='Rejection Sampling Samples')
        plt.title('Samples from Rejection Sampling')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

        plt.tight_layout()
        plt.show()