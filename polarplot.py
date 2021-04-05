import matplotlib
import matplotlib.pyplot as plt
import numpy as np

gamma = 1.4
M     = 3
sigma_min = np.arcsin(1/M)
sigma_max = 90*np.pi/180

sigma = np.linspace(sigma_min,sigma_max,20)
delta = np.arctan(((M**2*(np.sin(sigma))**2-1)/np.tan(sigma))/((gamma+1)/2*M**2-M**2*(np.sin(sigma))+1))
Pr    = (2*gamma*M**2*(np.sin(sigma))**2-(gamma-1))/(gamma+1)

plt.plot(delta,Pr)
plt.show()
