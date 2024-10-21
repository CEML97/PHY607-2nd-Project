import numpy as np
import random as rdm
import matplotlib.pyplot as plt


#Required Functions
##Generating white noise. Standard for Langevin equations.
def dW(dt: float) -> float:
    return np.random.normal(loc=0.0, scale=np.sqrt(dt))


#Numerical Parameters
T = 100 #Total time
N = 1000000   #Number of points in time interval
dt = T/N    #Time step

#Initial Conditions
x0 = 0.01  #Position
v0 = 50.0  #Velocity
L = 200000  #Length of interval
p = 0#Probability the particle is removed from interval
time = np.linspace(0.0, T, N)    #Time interval

#Physical Parameters
gamma = 0.5 #Friction coefficient
m = 1.0   #Mass
sigma = 1.5 #Noise

#Storing Variables
Variables = np.array([x0, v0], float)   #for updating
x = []  #Position Storing
v = []; #Velocity Storing

for i in time:
    x.append(Variables[0])  #Store previous position
    v.append(Variables[1])  #Store previous velocity
    
    ##Reflectiong at |x| = L with probability p
    if L < abs(Variables[0]):
        if p < rdm.random():
            break
        else:
            Variables[1] = -Variables[1]
            
    Variables[0] += Variables[1]*dt  #Update position, deterministic Euler method
    Variables[1] += -gamma/m*Variables[1]*dt + sigma/m*dW(dt)    #Update velocity, stochastic Euler (- Mayurama) method



#Filling the storing variables with zeros to match the length of the time interval. Issues with plotting if you delete this.
if len(x) < N:
    for i in range(N - len(x)):
        x.append(0)
        v.append(0)

#Plotting position
fig, ax = plt.subplots()
ax.plot(time, x)
ax.set(xlabel="t (time)", ylabel="x (position)", title="Position vs time")
fig.savefig("position.png")

#Plotting velocity
fig, ax = plt.subplots()
ax.plot(time, v)
ax.set(xlabel="t (time)", ylabel="v (velocity)", title="Velocity vs time")
fig.savefig("velocity.png")


msd = np.cumsum((np.array(x) - x0)**2) / np.arange(1, len(x)+1)  # MSD for each time step

fit_coeffs = np.polyfit(time, msd, 1)  # Linear fit
fitted_msd = np.polyval(fit_coeffs, time)  # Evaluated linear fit

# Plotting MSD and the linear fit
plt.figure()
plt.plot(time, msd, label="MSD (simulated)", alpha=0.75)
plt.plot(time, fitted_msd, label=f"Linear fit: MSD = {fit_coeffs[0]:.4f}*t + {fit_coeffs[1]:.4f}", linestyle='--', color='red')
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
plt.show()