import numpy as np

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
        #"""Generate Wiener process increment."""
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
        """Implement RK4 method."""
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