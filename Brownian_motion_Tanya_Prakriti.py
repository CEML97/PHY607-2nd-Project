import numpy as np
import random as rdm
import matplotlib.pyplot as plt
from scipy.stats import sem


# Required Functions
# Generating white noise. Standard for Langevin equations.
def dW(dt: float) -> float:
    return np.random.normal(loc=0.0, scale=np.sqrt(dt))


# Numerical Parameters
T = 100  # Total time
N = 1000000  # Number of points in time interval
dt = T/N  # Time step

# Initial Conditions
x0 = 0.01  # Position
v0 = 50.0  # Velocity
L = 200000  # Length of interval
p = 0  # Probability the particle is removed from interval
time = np.linspace(0.0, T, N)  # Time interval

# Physical Parameters
gamma = 0.5  # Friction coefficient
m = 1.0  # Mass
sigma = 1.5  # Noise

# Storing Variables
Variables = np.array([x0, v0], float)  # for updating
x = []  # Position Storing
v = []  # Velocity Storing

for i in time:
    x.append(Variables[0])  # Store previous position
    v.append(Variables[1])  # Store previous velocity

    # Reflectiong at |x| = L with probability p
    if L < abs(Variables[0]):
        if p < rdm.random():
            break
        else:
            Variables[1] = -Variables[1]

    # Update position, deterministic Euler method
    Variables[0] += Variables[1]*dt
    # Update velocity, stochastic Euler (- Mayurama) method
    Variables[1] += -gamma/m*Variables[1]*dt + sigma/m*dW(dt)


# Filling the storing variables with zeros to match the length of the time interval. Issues with plotting if you delete this.
if len(x) < N:
    for i in range(N - len(x)):
        x.append(0)
        v.append(0)


def compute_discretization_error(T, x0, v0, L, p, gamma, m, sigma):
    """
    Compute discretization error by comparing different time steps
    using the existing simulation code.
    """
    # List of N values to test
    N_values = [100, 200, 400, 800, 1600, 2000]
    errors = []
    dt_values = []

    # Generate reference solution with finest discretization
    np.random.seed(42)  # Fix seed for reproducibility
    x_ref, v_ref = run_simulation(
        T, max(N_values), x0, v0, L, p, gamma, m, sigma)

    # Compare with coarser discretizations
    for N in N_values[:-1]:
        np.random.seed(42)  # Same seed for fair comparison
        dt = T/N
        dt_values.append(dt)

        x, v = run_simulation(T, N, x0, v0, L, p, gamma, m, sigma)

        # Interpolate coarse solution to match reference solution time points
        t_coarse = np.linspace(0, T, len(x))
        t_fine = np.linspace(0, T, len(x_ref))
        x_interp = np.interp(t_fine, t_coarse, x)

        # Compute max absolute error
        error = np.max(np.abs(x_interp - x_ref))
        errors.append(error)

    return np.array(dt_values), np.array(errors)


def dW(dt: float) -> float:
    return np.random.normal(loc=0.0, scale=np.sqrt(dt))


def run_simulation(T, N, x0, v0, L, p, gamma, m, sigma):
    dt = T/N
    time = np.linspace(0.0, T, N)
    Variables = np.array([x0, v0], float)
    x = []
    v = []

    for i in time:
        x.append(Variables[0])
        v.append(Variables[1])

        if L < abs(Variables[0]):
            if p < rdm.random():
                break
            else:
                Variables[1] = -Variables[1]

        Variables[0] += Variables[1]*dt
        Variables[1] += -gamma/m*Variables[1]*dt + sigma/m*dW(dt)

    if len(x) < N:
        x.extend([0] * (N - len(x)))
        v.extend([0] * (N - len(v)))

    return np.array(x), np.array(v)


def compute_statistical_error(num_simulations):
    all_x = []
    for _ in range(num_simulations):
        x, _ = run_simulation(T, N, x0, v0, L, p, gamma, m, sigma)
        all_x.append(x)

    mean_x = np.mean(all_x, axis=0)
    std_error_x = sem(all_x, axis=0)

    return np.mean(std_error_x)


def run_simulation_rk4(T, N, x0, v0, L, p, gamma, m, sigma):
    dt = T/N
    time = np.linspace(0.0, T, N)
    x, v = [x0], [v0]

    for i in range(1, N):
        if L < abs(x[-1]):
            if p < np.random.random():
                break
            v[-1] = -v[-1]

        dW_val = dW(dt)

        k1x = v[-1]
        k1v = -gamma/m*v[-1] + sigma/m*dW_val/dt

        k2x = v[-1] + 0.5*dt*k1v
        k2v = -gamma/m*(v[-1] + 0.5*dt*k1v) + sigma/m*dW_val/dt

        k3x = v[-1] + 0.5*dt*k2v
        k3v = -gamma/m*(v[-1] + 0.5*dt*k2v) + sigma/m*dW_val/dt

        k4x = v[-1] + dt*k3v
        k4v = -gamma/m*(v[-1] + dt*k3v) + sigma/m*dW_val/dt

        x.append(x[-1] + dt/6*(k1x + 2*k2x + 2*k3x + k4x))
        v.append(v[-1] + dt/6*(k1v + 2*k2v + 2*k3v + k4v))

    if len(x) < N:
        x.extend([0] * (N - len(x)))
        v.extend([0] * (N - len(v)))

    return np.array(x), np.array(v)


# Plotting position
fig, ax = plt.subplots()
ax.plot(time, x)
ax.set(xlabel="t (time)", ylabel="x (position)", title="Position vs time")
fig.savefig("position.png")

# Plotting velocity
fig, ax = plt.subplots()
ax.plot(time, v)
ax.set(xlabel="t (time)", ylabel="v (velocity)", title="Velocity vs time")
fig.savefig("velocity.png")


# validation
msd = np.cumsum((np.array(x) - x0)**2) / np.arange(1,
                                                   len(x)+1)  # MSD for each time step

fit_coeffs = np.polyfit(time, msd, 1)  # Linear fit
fitted_msd = np.polyval(fit_coeffs, time)  # Evaluated linear fit

# Plotting MSD and the linear fit
plt.figure()
plt.plot(time, msd, label="MSD (simulated)", alpha=0.75)
plt.plot(time, fitted_msd,
         label=f"Linear fit: MSD = {fit_coeffs[0]:.4f}*t + {fit_coeffs[1]:.4f}", linestyle='--', color='red')
plt.xlabel("t (time)")
plt.ylabel("MSD (mean square displacement)")
plt.title("Mean Square Displacement vs Time with Linear Fit")
plt.legend()
plt.savefig("msd_fitting.png")
plt.show()

num_bins = 50
# Plotting Probability Distribution (Histogram)
plt.figure()
plt.hist(x, bins=num_bins, density=True, alpha=0.75)
plt.xlabel("x (position)")
plt.ylabel("Probability Density")
plt.title("Probability Distribution of Position")
plt.savefig("position_distribution.png")

num_simulations_list = [10, 20, 50, 100]  # reduced for the code speed
statistical_errors = []

for num_sims in num_simulations_list:
    error = compute_statistical_error(num_sims)
    statistical_errors.append(error)
    print(
        f"Number of simulations: {num_sims}, Average statistical error: {error}")

plt.figure(figsize=(10, 6))
plt.plot(num_simulations_list, statistical_errors, 'bo-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Simulations')
plt.ylabel('Average Statistical Error')
plt.title('Statistical Error vs Number of Simulations')
plt.grid(True)
theoretical_errors = statistical_errors[0] * np.sqrt(
    num_simulations_list[0]) / np.sqrt(num_simulations_list)
plt.plot(num_simulations_list, theoretical_errors,
         'r--', label='Theoretical 1/sqrt(N)')
plt.legend()

plt.savefig('statistical_error_vs_simulations.png')
plt.show()

dt_values, errors = compute_discretization_error(
    T, x0, v0, L, p, gamma, m, sigma)

plt.loglog(dt_values, errors, 'bo-', label='Numerical Error')
plt.loglog(dt_values, 100*dt_values**0.5, 'r--', label='O(√Δt) Reference')
plt.xlabel('Time Step (Δt)')
plt.ylabel('Max Absolute Error')
plt.title('Discretization Error Analysis')
plt.legend()
plt.grid(True)
plt.savefig("discretization_error.png")
plt.show()


def compute_precision_error(T, x0, v0, L, p, gamma, m, sigma, N):
    """
    Compare double precision vs single precision simulations
    to analyze finite precision effects.
    """
    np.random.seed(42)

    # Double precision (float64) simulation
    x_double, v_double = run_simulation(
        T, N, x0, v0, L, p, gamma, m, sigma
    )

    # Single precision (float32) simulation
    x_single, v_single = run_simulation(
        np.float32(T),
        N,
        np.float32(x0),
        np.float32(v0),
        np.float32(L),
        p,  # probability stays float64
        np.float32(gamma),
        np.float32(m),
        np.float32(sigma)
    )

    # Convert single precision results back to float64 for comparison
    x_single = np.float64(x_single)

    # Compute precision error
    precision_error = np.abs(x_double - x_single)

    return precision_error, x_double, x_single


def analyze_precision_effects(T, x0, v0, L, p, gamma, m, sigma):
    """
    Analyze how precision errors grow with different time steps
    """
    N_values = [100, 1000, 10000]
    results = []

    for N in N_values:
        precision_error, x_double, x_single = compute_precision_error(
            T, x0, v0, L, p, gamma, m, sigma, N
        )

        results.append({
            'N': N,
            'dt': T / N,
            'max_error': np.max(precision_error),
            'mean_error': np.mean(precision_error),
            'std_error': np.std(precision_error)
        })

    # Plot precision error over time for finest resolution
    plt.figure(figsize=(8, 5))
    time = np.linspace(0, T, len(precision_error))
    plt.semilogy(time, precision_error, 'b-', label='Precision Error')
    plt.xlabel('Time')
    plt.ylabel('|double - single|')
    plt.title('Precision Error Over Time')
    plt.grid(True)
    plt.show()

    # Plot how machine epsilon affects calculations
    eps_single = np.finfo(np.float32).eps

    # Theoretical relative rounding error accumulation
    # sqrt(n) factor due to random walk nature
    n_steps = np.array([r['N'] for r in results])
    theoretical_error = eps_single * np.sqrt(n_steps) * 8*10**6

    plt.figure(figsize=(8, 5))
    plt.loglog([r['dt'] for r in results],
               [r['max_error'] for r in results],
               'bo-', label='Observed')
    plt.loglog([r['dt'] for r in results],
               theoretical_error,
               'r--', label='Theoretical (scaled)')
    plt.xlabel('Time Step (dt)')
    plt.ylabel('Maximum Precision Error')
    plt.title('Precision Error vs Time Step')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print detailed analysis
    print("\nPrecision Error Analysis:")
    print(f"Machine epsilon (single): {eps_single:.3e}")
    print("\nError statistics for different time steps:")
    for r in results:
        print(f"\nN = {r['N']} (dt = {r['dt']:.3e}):")
        print(f"  Maximum error: {r['max_error']:.3e}")
        print(f"  Mean error: {r['mean_error']:.3e}")
        print(f"  Std error: {r['std_error']:.3e}")


analyze_precision_effects(T, x0, v0, L, p, gamma, m, sigma)
