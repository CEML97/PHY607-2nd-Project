import numpy as np
import matplotlib.pyplot as plt

class BrownianMotion:
    def __init__(self, T, N, x0, v0, L, p, gamma, m, sigma):
        self.T = T
        self.N = N
        self.dt = T / N
        self.x0 = x0
        self.v0 = v0
        self.L = L
        self.p = p
        self.gamma = gamma
        self.m = m
        self.sigma = sigma
        self.time = np.linspace(0.0, T, N)

    def dW(self) -> float:
        return np.random.normal(loc=0.0, scale=np.sqrt(self.dt))

    def run_simulation(self, method='euler'):
        if method == 'euler':
            return self._run_euler()
        elif method == 'rk4':
            return self._run_rk4()
        else:
            raise ValueError("Method not recognized. Use 'euler' or 'rk4'.")

    def _run_euler(self):
        Variables = np.array([self.x0, self.v0], float)
        x, v = [], []
        
        for _ in self.time:
            x.append(Variables[0])
            v.append(Variables[1])
            
            if self.L < abs(Variables[0]):
                if self.p < np.random.random():
                    break
                else:
                    Variables[1] = -Variables[1]

            Variables[0] += Variables[1] * self.dt
            Variables[1] += -self.gamma / self.m * Variables[1] * self.dt + self.sigma / self.m * self.dW()

        if len(x) < self.N:
            x.extend([0] * (self.N - len(x)))
            v.extend([0] * (self.N - len(v)))

        return np.array(x), np.array(v)

    def _run_rk4(self):
        x, v = [self.x0], [self.v0]
        
        for i in range(1, self.N):
            if self.L < abs(x[-1]):
                if self.p < np.random.random():
                    break
                v[-1] = -v[-1]
            
            dW_val = self.dW()
            k1x = v[-1]
            k1v = -self.gamma / self.m * v[-1] + self.sigma / self.m * dW_val / self.dt
            
            k2x = v[-1] + 0.5 * self.dt * k1v
            k2v = -self.gamma / self.m * (v[-1] + 0.5 * self.dt * k1v) + self.sigma / self.m * dW_val / self.dt
            
            k3x = v[-1] + 0.5 * self.dt * k2v
            k3v = -self.gamma / self.m * (v[-1] + 0.5 * self.dt * k2v) + self.sigma / self.m * dW_val / self.dt
            
            k4x = v[-1] + self.dt * k3v
            k4v = -self.gamma / self.m * (v[-1] + self.dt * k3v) + self.sigma / self.m * dW_val / self.dt
            
            x.append(x[-1] + self.dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x))
            v.append(v[-1] + self.dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))

        if len(x) < self.N:
            x.extend([0] * (self.N - len(x)))
            v.extend([0] * (self.N - len(v)))

        return np.array(x), np.array(v)

    def sample_inverse_cdf(self, size=1):
        lam = 1.0
        u = np.random.uniform(0, 1, size)
        samples = -np.log(1 - u) / lam
        return samples

    def sample_rejection(self, size=1):
        samples = []
        while len(samples) < size:
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(0, 0.4)
            if y < (1/np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2):
                samples.append(x)
        return np.array(samples)

class SimulationResults:
    def __init__(self, num_simulations, T, N, x0, v0, L, p, gamma, m, sigma):
        self.num_simulations = num_simulations
        self.brownian_motion = BrownianMotion(T, N, x0, v0, L, p, gamma, m, sigma)
        self.x_euler_all = []
        self.v_euler_all = []
        self.x_rk4_all = []
        self.v_rk4_all = []

    def run_all_simulations(self):
        for _ in range(self.num_simulations):
            x_euler, v_euler = self.brownian_motion.run_simulation(method='euler')
            x_rk4, v_rk4 = self.brownian_motion.run_simulation(method='rk4')
            self.x_euler_all.append(x_euler)
            self.v_euler_all.append(v_euler)
            self.x_rk4_all.append(x_rk4)
            self.v_rk4_all.append(v_rk4)

    def calculate_statistics(self):
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
        x_euler_mean = np.mean(self.x_euler_all, axis=0)
        x_rk4_mean = np.mean(self.x_rk4_all, axis=0)
        
        numerical_error = np.abs(x_euler_mean - x_rk4_mean)
        return numerical_error

    def calculate_statistical_error(self):
        x_euler_std = np.std(self.x_euler_all, axis=0)
        return x_euler_std

    def calculate_finite_precision_error(self, precisions):
        x_results = []
        for precision in precisions:
            x, _ = self.brownian_motion.run_simulation(method='euler')
            x_results.append(x.astype(np.float64))
        
        errors = []
        for i in range(len(precisions) - 1):
            diff = np.abs(x_results[i] - x_results[-1])
            errors.append(diff)
        return errors

    def plot_results(self, stats):
        x_euler_mean, v_euler_mean, x_rk4_mean, v_rk4_mean, \
        x_euler_std, v_euler_std, x_rk4_std, v_rk4_std = stats
        
        time = self.brownian_motion.time

        plt.figure(figsize=(12, 10))

        plt.subplot(2, 1, 1)
        plt.errorbar(time[::1000], x_euler_mean[::1000], yerr=x_euler_std[::1000], label='Euler', alpha=0.7)
        plt.errorbar(time[::1000], x_rk4_mean[::1000], yerr=x_rk4_std[::1000], label='RK4', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Position (x)')
        plt.title('Position vs Time: Euler and RK4 Methods')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.errorbar(time[::1000], v_euler_mean[::1000], yerr=v_euler_std[::1000], label='Euler', alpha=0.7)
        plt.errorbar(time[::1000], v_rk4_mean[::1000], yerr=v_rk4_std[::1000], label='RK4', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Velocity (v)')
        plt.title('Velocity vs Time: Euler and RK4 Methods')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Simulation parameters
T, N = 10, 1000000
x0, v0, L, p = 0.01, 50.0, 200000, 0
gamma, m, sigma = 0.5, 1.0, 1.5

# Run simulations
num_simulations = 10
simulation_results = SimulationResults(num_simulations, T, N, x0, v0, L, p, gamma, m, sigma)
simulation_results.run_all_simulations()
stats = simulation_results.calculate_statistics()
simulation_results.plot_results(stats)

# Probability Distribution Demonstration
n_samples = 1000
inverse_cdf_samples = simulation_results.brownian_motion.sample_inverse_cdf(size=n_samples)
rejection_samples = simulation_results.brownian_motion.sample_rejection(size=n_samples)

# Plotting the samples
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(inverse_cdf_samples, bins=30, density=True, alpha=0.6, color='b', label='Inverse CDF Samples')
plt.title('Samples from Inverse CDF')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(rejection_samples, bins=30, density=True, alpha=0.6, color='r', label='Rejection Sampling Samples')
plt.title('Samples from Rejection Sampling')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate errors
numerical_error = simulation_results.calculate_numerical_error()
statistical_error = simulation_results.calculate_statistical_error()
finite_precision_errors = simulation_results.calculate_finite_precision_error([np.float32, np.float64])

# Plot numerical error
plt.figure(figsize=(12, 6))
plt.plot(simulation_results.brownian_motion.time, numerical_error, label='Numerical Error', color='orange')
plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Numerical Error over Time')
plt.legend()
plt.show()

# Plot statistical error
plt.figure(figsize=(12, 6))
plt.plot(simulation_results.brownian_motion.time, statistical_error, label='Statistical Error', color='blue')
plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Statistical Error over Time')
plt.legend()
plt.show()

# Plot finite precision error
plt.figure(figsize=(12, 6))
for i, error in enumerate(finite_precision_errors):
    plt.plot(simulation_results.brownian_motion.time, error, label=f'Precision Error (Type {i})')
plt.xlabel('Time')
plt.ylabel('Absolute Error')
plt.title('Finite Precision Errors')
plt.legend()
plt.yscale('log')
plt.show()

# Print theoretical error estimates
dt = T/N
print(f"Theoretical Euler method global error: {dt}")
print(f"Theoretical RK4 method global error: {dt**4}")
print(f"Theoretical float32 precision limit: {np.finfo(np.float32).eps}")
print(f"Theoretical float64 precision limit: {np.finfo(np.float64).eps}")
