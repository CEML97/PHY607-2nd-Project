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
v0 = 5.0  #Velocity
L = 2  #Length of interval
p = 0.8 #Probability the particle is removed from interval
time = np.linspace(0.0, T, N)    #Time interval

#Physical Parameters
gamma = 2.0 #Friction coefficient
m = 1.0   #Mass
sigma = 0.5 #Noise

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