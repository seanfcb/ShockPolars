import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def oblique(Mx,sigma,gamma):
    Mxn   = Mx*np.sin(sigma)
    delta = np.arctan(((Mx**2*(np.sin(sigma))**2-1)/np.tan(sigma))/((gamma+1)/2*Mx**2-Mx**2*(np.sin(sigma))+1))
    Pr    = (2*gamma*Mx**2*(np.sin(sigma))**2-(gamma-1))/(gamma+1)
    Myn   = np.sqrt((2+(gamma-1)*Mxn**2)/(2*gamma*Mxn**2-(gamma-1)))
    My    = Myn/(np.sin(sigma-delta))
    return(delta, Pr, My)

gamma = 1.4
M1    = 1.5
sigma1_min = np.arcsin(1/M1)
sigma1_max = 90*np.pi/180

sigma1 = np.linspace(sigma1_min,sigma1_max,20)
delta1, Pr1, M2 = oblique(M1, sigma1, gamma)
delta_ramp = delta1[5]
M_ramp     = M2[5]
Pr_ramp    = Pr1[5]

sigma2_min = np.arcsin(1/M_ramp)
sigma2_max = 90*np.pi/180
sigma2 = np.linspace(sigma2_min,sigma2_max,20)
delta2, Pr2, M3 = oblique(M2, sigma2, gamma)

#
plt.plot(delta1*180/np.pi,Pr1)
#plt.plot(delta1*180/np.pi, M2)
#plt.plot((delta_ramp-delta2)*180/np.pi,Pr2*Pr_ramp)
plt.plot(sigma1, Pr1)
plt.plot(sigma2, Pr2)

plt.show()
