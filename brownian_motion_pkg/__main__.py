from src.brownian_motion_sim.analysis.error_analysis import SimulationResults
from src.brownian_motion_sim.core.sampling import ProbabilitySampler
from src.brownian_motion_sim.visualization.plotters import SimulationPlotter
from src.brownian_motion_sim.core.simulation import BrownianMotion

def main():
    # Set simulation parameters
    T, N = 10, 1000000
    x0, v0, L, p = 0.01, 50.0, 200000, 0
    gamma, m, sigma = 0.5, 1.0, 1.5

    # Run simulations
    print("Running simulations...")
    num_simulations = 10
    sim_results = SimulationResults(num_simulations, T, N, x0, v0, L, p, gamma, m, sigma)
    sim_results.run_all_simulations()
    stats = sim_results.calculate_statistics()

    # Calculate errors
    print("Calculating errors...")
    numerical_error = sim_results.calculate_numerical_error()
    statistical_error = sim_results.calculate_statistical_error()
    finite_precision_errors = sim_results.calculate_finite_precision_error([np.float32, np.float64])

    # Plot results
    print("Plotting results...")
    SimulationPlotter.plot_simulation_results(sim_results.brownian_motion.time, stats)
    
    # Generate and plot probability distributions
    print("Generating probability distributions...")
    n_samples = 1000
    inverse_cdf_samples = ProbabilitySampler.sample_inverse_cdf(size=n_samples)
    rejection_samples = ProbabilitySampler.sample_rejection(size=n_samples)
    SimulationPlotter.plot_probability_distributions(inverse_cdf_samples, rejection_samples)

    # Plot error analysis
    print("Plotting error analysis...")
    SimulationPlotter.plot_error_analysis(
        sim_results.brownian_motion.time,
        numerical_error,
        statistical_error,
        finite_precision_errors
    )
    dt = T/N
    print(f"\nTheoretical Error Estimates:")
    print(f"Euler method global error: {dt}")
    print(f"RK4 method global error: {dt**4}")
    print(f"float32 precision limit: {np.finfo(np.float32).eps}")
    print(f"float64 precision limit: {np.finfo(np.float64).eps}")

if __name__ == "__main__":
    import numpy as np
    main()
    print("\nSimulation completed successfully!")